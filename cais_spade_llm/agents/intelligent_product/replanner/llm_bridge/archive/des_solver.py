"""Archived DES solver for legacy plant || safety DFA composition + BFS."""

from __future__ import annotations

import logging
import re
from collections import deque
from copy import deepcopy
from typing import Any
from urllib.parse import parse_qsl

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_PRODUCT_STATES = 50_000


def _string_token(value: Any) -> str:
    return str(value or "").strip()


def event_location_ref(event: dict[str, Any]) -> str:
    """Return the normalized direction-neutral grounded anchor for an event."""
    return _string_token(event.get("location_ref") or event.get("target_ref"))

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


def _norm_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _resource_short_name(resource_jid: Any) -> str:
    token = _norm_token(resource_jid)
    if "@" in token:
        token = token.split("@", 1)[0]
    return token


def _parse_ap_full(full: Any) -> dict[str, str]:
    """Parse AP strings like ap_event/assembly/mcp/ur5e/place_approach/destination=x."""
    token = str(full or "").strip()
    segments = [segment.strip() for segment in token.split("/") if segment.strip()]
    if len(segments) < 5:
        return {}
    parsed = {
        "prefix": segments[0],
        "process": segments[1] if len(segments) > 1 else "",
        "product": segments[2] if len(segments) > 2 else "",
        "resource": segments[3] if len(segments) > 3 else "",
        "function_name": segments[4] if len(segments) > 4 else "",
        "context": segments[5] if len(segments) > 5 else "",
    }
    if parsed["prefix"] not in {"ap", "ap_event"}:
        return {}
    return parsed


def _descriptor_value(desc: dict[str, Any], *names: str) -> str:
    for name in names:
        value = str(desc.get(name) or "").strip()
        if value:
            return value
    selector = dict(desc.get("selector") or {})
    for name in names:
        value = str(selector.get(name) or "").strip()
        if value:
            return value
    parsed = _parse_ap_full(desc.get("full"))
    for name in names:
        value = str(parsed.get(name) or "").strip()
        if value:
            return value
    return ""


def _event_context_values(event: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "location_ref",
        "target_ref",
        "target_location",
        "destination_location",
        "destination",
        "location",
        "context",
    ):
        value = event.get(key)
        if value not in (None, "") and not isinstance(value, (dict, list, tuple, set)):
            values.add(str(value).strip())
    action_target = event.get("action_target")
    if isinstance(action_target, dict):
        for key in ("target_location", "destination_location", "destination", "location"):
            value = action_target.get(key)
            if value not in (None, ""):
                values.add(str(value).strip())
    params = event.get("params")
    if isinstance(params, dict):
        for key in ("target_location", "destination_location", "destination", "location"):
            value = params.get(key)
            if value not in (None, ""):
                values.add(str(value).strip())
    return {value for value in values if value}


def _context_matches(desc_context: str, event: dict[str, Any]) -> bool:
    token = str(desc_context or "").strip()
    if not token or _norm_token(token) == "any":
        return True

    event_values = _event_context_values(event)
    normalized_values = {_norm_token(value) for value in event_values}
    try:
        pairs = parse_qsl(token, keep_blank_values=True, strict_parsing=False)
    except Exception:
        pairs = []
    if not pairs and "=" in token:
        key, value = token.split("=", 1)
        pairs = [(key, value)]
    if not pairs:
        return _norm_token(token) in normalized_values

    for key, value in pairs:
        normalized_value = _norm_token(value)
        if not normalized_value or normalized_value == "any":
            continue
        direct_value = event.get(str(key).strip())
        if direct_value not in (None, "") and _norm_token(direct_value) == normalized_value:
            continue
        if normalized_value in normalized_values:
            continue
        return False
    return True


def _descriptor_selector(desc: dict[str, Any]) -> dict[str, Any]:
    selector = dict(desc.get("selector") or {})
    if selector:
        return selector

    parsed = _parse_ap_full(desc.get("full"))
    if not parsed or parsed.get("prefix") != "ap_event":
        return {}

    context = str(parsed.get("context") or "").strip()
    destination = ""
    if context:
        try:
            destination = dict(parse_qsl(context, keep_blank_values=True)).get("destination", "")
        except Exception:
            destination = ""
        if not destination and "=" in context:
            key, value = context.split("=", 1)
            if key.strip() == "destination":
                destination = value.strip()
    part_name = str(parsed.get("product") or "").strip()
    return {
        "mode": "move_part_to_destination" if part_name and part_name.lower() != "any" else "resource_move_to_destination",
        "part": part_name or "any",
        "resource": str(parsed.get("resource") or "any").strip() or "any",
        "destination": destination,
    }


def _semantic_selector_matches(selector: dict[str, Any], event: dict[str, Any]) -> bool:
    if not selector:
        return False

    witness = dict(event.get("semantic_witness") or {})
    mode = _norm_token(selector.get("mode"))
    resource_selector = _norm_token(selector.get("resource") or "any")
    part_selector = _norm_token(selector.get("part") or "any")
    destination = str(selector.get("destination") or "").strip()

    resource_jid = str(event.get("resource_jid") or "").strip()
    resource_short = _resource_short_name(resource_jid)
    if (
        resource_selector not in {"", "any", "robot"}
        and resource_selector not in {_norm_token(resource_jid), resource_short}
    ):
        return False

    part_name = str(event.get("part_name") or "").strip()
    if part_selector not in {"", "any"} and part_selector != _norm_token(part_name):
        return False

    after_resource = dict(witness.get("resource_after") or {})
    after_part = dict(witness.get("part_after") or {})
    destination_tokens = {
        token
        for token in (
            event_location_ref(event),
            str(witness.get("location_ref") or "").strip(),
            str(after_resource.get("current_location") or after_resource.get("location") or "").strip(),
            str(after_part.get("current_location") or after_part.get("location") or "").strip(),
        )
        if token
    }
    if destination and destination not in destination_tokens:
        return False

    category = _norm_token(witness.get("category"))
    if mode == "move_part_to_destination":
        return bool(part_name and category in {"acquisition", "release", "transfer", "carry"})
    if mode == "resource_move_to_destination":
        return bool(destination_tokens and category in {"resource_only_transition", "carry", "release", "transfer", "acquisition"})
    return False


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
    resource_short = _resource_short_name(resource_jid)
    part_name = str(event.get("part_name") or "").strip()
    action_name = str(event.get("name") or "").strip()
    function_candidates = {
        _norm_token(event.get("function_name")),
        _norm_token(action_name),
    }
    function_candidates.discard("")

    for desc in ap_descriptors:
        desc = dict(desc or {})
        ap_label = str(desc.get("label") or desc.get("ap_label") or "").strip()
        if not ap_label:
            continue

        selector = _descriptor_selector(desc)
        if _semantic_selector_matches(selector, event):
            matched.add(ap_label)
            continue

        # Resource match
        desc_resource = _descriptor_value(desc, "resource") or "any"
        desc_resource_norm = _norm_token(desc_resource)
        if (
            desc_resource_norm not in {"any", "robot"}
            and desc_resource_norm != _norm_token(resource_jid)
            and desc_resource_norm != resource_short
        ):
            continue

        # Function/action match
        desc_function = (
            _descriptor_value(desc, "function_name", "function", "event", "symbol")
            or "any"
        )
        desc_function_norm = _norm_token(desc_function)
        if desc_function_norm != "any" and desc_function_norm not in function_candidates:
            continue

        # Product/part match
        desc_product = _descriptor_value(desc, "product", "part") or "any"
        if _norm_token(desc_product) != "any" and _norm_token(desc_product) != _norm_token(part_name):
            continue

        # Context match (e.g., target_ref must match)
        desc_context = _descriptor_value(desc, "context", "destination") or "any"
        if not _context_matches(desc_context, event):
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
    max_states: int | None = None,
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

    state_cap = int(max_states or _MAX_PRODUCT_STATES)

    while queue and len(visited) < state_cap:
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
    if len(visited) >= state_cap and found_goal is None:
        diagnostic = (
            f"Reached the explored-state cap ({state_cap}) before finding any accepting "
            "product state. The recovery search budget may be too small for this plant."
        )
        return {
            "status": "unsolvable",
            "trace": [],
            "action_sequence": [],
            "blocked_transitions": blocked_transitions,
            "product_states_explored": len(visited),
            "reachable_plant_states": sorted({ps[0] for ps in visited}),
            "unreachable_marked": sorted(plant_marked - {ps[0] for ps in visited}),
            "diagnostic": diagnostic,
        }

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
            "description": str(
                edict.get("description") or edict.get("name") or ""
            ).strip(),
        }
        part_name = str(edict.get("part_name") or "").strip()
        if part_name:
            action["part_name"] = part_name
        location_ref = event_location_ref(edict)
        if location_ref:
            action["location_ref"] = location_ref
        pose = edict.get("pose")
        if isinstance(pose, dict):
            action["pose"] = deepcopy(pose)
        projected_effect = edict.get("projected_effect")
        if isinstance(projected_effect, dict):
            action["projected_effect"] = deepcopy(projected_effect)
        semantic_witness = edict.get("semantic_witness")
        if isinstance(semantic_witness, dict):
            action["semantic_witness"] = deepcopy(semantic_witness)
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
