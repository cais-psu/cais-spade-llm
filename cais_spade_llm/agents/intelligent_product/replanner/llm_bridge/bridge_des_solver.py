"""DES solver for hybrid bridge mode — plant || safety DFA composition + BFS.

Composes an LLM-generated recovery plant automaton with pre-compiled
safety specification DFAs to produce a supervised product automaton,
then finds the shortest accepting trace via BFS.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from copy import deepcopy
from typing import Any

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_PRODUCT_STATES = 50_000

# ---------------------------------------------------------------------------
# Lightweight DFA transition helpers (self-contained, no BaseSafetyChecker)
# ---------------------------------------------------------------------------

_compiled_expr_cache: dict[str, Any] = {}
_eval_result_cache: dict[tuple[str, frozenset[str]], bool] = {}


def _eval_label(
    label: str,
    sigma: frozenset[str],
    rule_aps: list[str],
) -> bool:
    """Evaluate a boolean label expression against active APs."""
    label = label.strip()
    if label.lower() == "true":
        return True
    if not label or label.lower() == "false":
        return False

    active_aps = frozenset(ap for ap in rule_aps if ap in sigma)
    cache_key = (label, active_aps)
    cached = _eval_result_cache.get(cache_key)
    if cached is not None:
        return cached

    if label not in _compiled_expr_cache:
        expr = label.replace("&", " and ").replace("|", " or ")
        expr = expr.replace("~", " not ").replace("!", " not ")
        expr = re.sub(r"\btrue\b", "True", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bfalse\b", "False", expr, flags=re.IGNORECASE)
        try:
            _compiled_expr_cache[label] = compile(expr.strip(), "<string>", "eval")
        except Exception:
            _logger.error("Failed to compile label expression: %s", label)
            _compiled_expr_cache[label] = None

    compiled = _compiled_expr_cache.get(label)
    if compiled is None:
        _eval_result_cache[cache_key] = False
        return False

    env = {ap: (ap in active_aps) for ap in rule_aps}
    try:
        val = bool(eval(compiled, {"__builtins__": {}}, env))  # noqa: S307
    except Exception:
        _logger.error("Failed to evaluate label: %s", label)
        val = False
    _eval_result_cache[cache_key] = val
    return val


def _delta(
    dfa: dict[str, Any],
    current_state: str,
    sigma: frozenset[str],
) -> str:
    """Compute next DFA state given current state and active AP set."""
    transitions = dfa.get("transitions", {}).get(current_state, [])
    ap_symbols = dfa.get("ap_symbols", [])
    for label, dst in transitions:
        if _eval_label(label, sigma, ap_symbols):
            return dst
    return current_state  # stutter


# ---------------------------------------------------------------------------
# AP mapping for plant events
# ---------------------------------------------------------------------------


def _map_event_to_aps(
    event: dict[str, Any],
    ap_descriptors: list[dict[str, Any]],
) -> frozenset[str]:
    """Map a plant event to a set of AP labels using AP descriptor matching.

    This is a simplified version of BaseSafetyChecker._map_task_to_aps that
    works with the plant event structure directly.

    Parameters
    ----------
    event
        Plant event dict with resource_jid, action_type, part_name, etc.
    ap_descriptors
        List of AP descriptor dicts from loaded safety rules.
        Each has: label, resource (or "any"), function_name (or "any"),
                  product (or "any"), context (or "any").

    Returns
    -------
    frozenset of matching AP labels.
    """
    matched: set[str] = set()
    resource_jid = str(event.get("resource_jid") or "").strip()
    action_type = str(event.get("action_type") or "").strip()
    part_name = str(event.get("part_name") or "").strip()
    action_name = str(event.get("name") or "").strip()

    for desc in ap_descriptors:
        desc = dict(desc or {})
        ap_label = str(desc.get("label") or desc.get("ap_label") or "").strip()
        if not ap_label:
            continue

        # Resource match
        desc_resource = str(desc.get("resource") or "any").strip()
        if desc_resource != "any" and desc_resource != resource_jid:
            continue

        # Function/action match
        desc_function = str(desc.get("function_name") or desc.get("function") or "any").strip()
        if desc_function != "any":
            if desc_function != action_type and desc_function != action_name:
                continue

        # Product/part match
        desc_product = str(desc.get("product") or desc.get("part") or "any").strip()
        if desc_product != "any" and desc_product != part_name:
            continue

        # Context match (e.g., target_ref must match)
        desc_context = str(desc.get("context") or "any").strip()
        if desc_context != "any":
            target_ref = str(event.get("target_ref") or "").strip()
            if desc_context != target_ref:
                continue

        matched.add(ap_label)

    return frozenset(matched)


# ---------------------------------------------------------------------------
# Product automaton types
# ---------------------------------------------------------------------------

# Product state: (plant_state, safety_dfa_state_vector)
ProductState = tuple[str, tuple[str, ...]]


# ---------------------------------------------------------------------------
# Composition + solving
# ---------------------------------------------------------------------------


def compose_and_solve(
    plant: dict[str, Any],
    safety_dfas: dict[str, dict[str, Any]],
    ap_descriptors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compose plant with safety DFAs and find shortest recovery trace.

    Parameters
    ----------
    plant
        Compiled plant dict from ``bridge_plant_compiler``.
    safety_dfas
        ``{rule_id: dfa_dict}`` — pre-compiled safety specification DFAs.
        Each dfa_dict has keys: initial, transitions, ap_symbols,
        violation_state (optional), accepting_states (optional).
    ap_descriptors
        AP descriptor list for mapping events to atomic propositions.
        If None, events pass through all safety DFAs unchecked.

    Returns
    -------
    dict with:
        status: "solved" | "unsolvable" | "safety_blocked"
        trace: list of {event, plant_from, plant_to, safety_q}
        action_sequence: list of grounded action dicts
        blocked_transitions: events pruned by safety (if any)
        diagnostic: human-readable explanation (if not solved)
    """
    ap_descriptors = ap_descriptors or []
    rule_ids = sorted(safety_dfas.keys())
    plant_initial = str(plant.get("initial") or "").strip()
    plant_marked = set(plant.get("marked") or set())
    plant_transitions = dict(plant.get("transitions") or {})
    plant_events = dict(plant.get("events") or {})

    # Build initial product state
    dfa_q0_vec = tuple(
        str(safety_dfas[rid].get("initial") or "0")
        for rid in rule_ids
    )
    initial_ps: ProductState = (plant_initial, dfa_q0_vec)

    # BFS
    visited: set[ProductState] = set()
    queue: deque[ProductState] = deque([initial_ps])
    parent: dict[ProductState, ProductState | None] = {initial_ps: None}
    parent_edge: dict[ProductState, str | None] = {initial_ps: None}
    blocked_transitions: list[dict[str, Any]] = []
    found_goal: ProductState | None = None

    while queue and len(visited) < _MAX_PRODUCT_STATES:
        ps = queue.popleft()
        if ps in visited:
            continue
        visited.add(ps)

        plant_s, q_vec = ps

        # Check if we reached a marked plant state
        if plant_s in plant_marked:
            # Verify no safety DFA is in violation
            all_safe = True
            for i, rid in enumerate(rule_ids):
                violation_state = safety_dfas[rid].get("violation_state")
                if violation_state and q_vec[i] == violation_state:
                    all_safe = False
                    break
            if all_safe:
                found_goal = ps
                break

        # Explore transitions
        for event_name, dst_plant in plant_transitions.get(plant_s, []):
            edict = plant_events.get(event_name, {})

            # Map event to APs
            sigma = _map_event_to_aps(edict, ap_descriptors)

            # Advance all safety DFAs
            new_q_list: list[str] = []
            safety_violation = False
            violated_rule: str | None = None
            for i, rid in enumerate(rule_ids):
                dfa = safety_dfas[rid]
                q_next = _delta(dfa, q_vec[i], sigma)
                violation_state = dfa.get("violation_state")
                if violation_state and q_next == violation_state:
                    safety_violation = True
                    violated_rule = rid
                    break
                new_q_list.append(q_next)

            if safety_violation:
                blocked_transitions.append({
                    "event": event_name,
                    "from_plant": plant_s,
                    "to_plant": dst_plant,
                    "violated_rule": violated_rule,
                    "sigma": sorted(sigma),
                })
                continue

            new_q_vec = tuple(new_q_list)
            successor: ProductState = (dst_plant, new_q_vec)
            if successor not in visited:
                parent[successor] = ps
                parent_edge[successor] = event_name
                queue.append(successor)

    # Reconstruct trace
    if found_goal is not None:
        trace = _reconstruct_trace(found_goal, parent, parent_edge, plant_events)
        action_sequence = _trace_to_action_sequence(trace, plant_events)
        return {
            "status": "solved",
            "trace": trace,
            "action_sequence": action_sequence,
            "blocked_transitions": blocked_transitions,
            "product_states_explored": len(visited),
            "diagnostic": None,
        }

    # Diagnose failure
    if blocked_transitions:
        diagnostic = (
            f"Safety rules blocked {len(blocked_transitions)} transition(s). "
            f"No safe path to any marked state found after exploring "
            f"{len(visited)} product states."
        )
        status = "safety_blocked"
    else:
        diagnostic = (
            f"No path from initial to any marked state found after exploring "
            f"{len(visited)} product states. The plant may be missing required "
            f"recovery actions."
        )
        status = "unsolvable"

    return {
        "status": status,
        "trace": [],
        "action_sequence": [],
        "blocked_transitions": blocked_transitions,
        "product_states_explored": len(visited),
        "reachable_plant_states": sorted({ps[0] for ps in visited}),
        "unreachable_marked": sorted(plant_marked - {ps[0] for ps in visited}),
        "diagnostic": diagnostic,
    }


# ---------------------------------------------------------------------------
# Trace reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_trace(
    goal: ProductState,
    parent: dict[ProductState, ProductState | None],
    parent_edge: dict[ProductState, str | None],
    plant_events: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Walk parent pointers back to initial state and reverse."""
    path: list[tuple[ProductState, str | None]] = []
    current: ProductState | None = goal
    while current is not None:
        edge = parent_edge.get(current)
        path.append((current, edge))
        current = parent.get(current)
    path.reverse()

    trace: list[dict[str, Any]] = []
    for i in range(1, len(path)):
        prev_ps = path[i - 1][0]
        curr_ps = path[i][0]
        event_name = path[i][1]
        trace.append({
            "event": event_name,
            "plant_from": prev_ps[0],
            "plant_to": curr_ps[0],
            "safety_q_before": prev_ps[1],
            "safety_q_after": curr_ps[1],
        })
    return trace


def _trace_to_action_sequence(
    trace: list[dict[str, Any]],
    plant_events: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert a trace into a list of grounded action dicts."""
    actions: list[dict[str, Any]] = []
    for step in trace:
        event_name = step.get("event")
        edict = dict(plant_events.get(event_name) or {})
        action: dict[str, Any] = {
            "event_name": event_name,
            "resource_jid": str(edict.get("resource_jid") or "").strip(),
            "action_type": str(edict.get("action_type") or "").strip(),
            "description": str(
                edict.get("description") or edict.get("name") or ""
            ).strip(),
        }
        part_name = str(edict.get("part_name") or "").strip()
        if part_name:
            action["part_name"] = part_name
        target_ref = str(edict.get("target_ref") or "").strip()
        if target_ref:
            action["target_ref"] = target_ref
        pose = edict.get("pose")
        if isinstance(pose, dict):
            action["pose"] = deepcopy(pose)
        actions.append(action)
    return actions


# ---------------------------------------------------------------------------
# Diagnostic summary for LLM feedback
# ---------------------------------------------------------------------------


def solver_diagnostic_summary(result: dict[str, Any]) -> str:
    """Render solver result as human-readable feedback for the LLM."""
    status = str(result.get("status") or "").strip()
    diagnostic = str(result.get("diagnostic") or "").strip()
    lines: list[str] = []

    if status == "solved":
        actions = result.get("action_sequence") or []
        lines.append(f"Recovery plan found with {len(actions)} step(s).")
        for i, a in enumerate(actions, 1):
            desc = a.get("description") or a.get("event_name") or "action"
            lines.append(f"  {i}. {desc}")
        return "\n".join(lines)

    lines.append(f"Status: {status}")
    if diagnostic:
        lines.append(f"Reason: {diagnostic}")

    blocked = result.get("blocked_transitions") or []
    if blocked:
        lines.append(f"\nBlocked transitions ({len(blocked)}):")
        for bt in blocked:
            lines.append(
                f"  - {bt.get('event')}: blocked by rule {bt.get('violated_rule')}"
            )

    unreachable = result.get("unreachable_marked") or []
    if unreachable:
        lines.append(f"\nGoal states unreachable: {', '.join(unreachable)}")

    reachable = result.get("reachable_plant_states") or []
    if reachable:
        lines.append(f"Reachable plant states: {', '.join(reachable)}")

    return "\n".join(lines)


__all__ = [
    "compose_and_solve",
    "solver_diagnostic_summary",
]
