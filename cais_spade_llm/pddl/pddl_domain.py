"""
PDDL domain generator for manufacturing recovery planning.

Generates the domain automatically from the tools catalog (tools.json).
If tools.json gains new functions or state names, the domain updates automatically.

State semantics (how state names map to PDDL predicate types):
  resource_only – predicate involves only the resource (?r)
  resource_part – predicate involves resource + part (?r ?p)
  part_location – predicate involves part + location (?p ?l)

State transition patterns → action types:
  resource_only → resource_only : pure resource movement (e.g. move_to_pick_location)
  part_location → resource_part : pick action (resource grabs part from location)
  resource_part → resource_part : carry action (resource moves with part to destination)
  resource_part → part_location : place action (resource releases part at location)

Two recovery actions are always added regardless of tools.json:
  stage-part       – puts held part at staging location (no ordering check)
  unlock-placement – fires when p1 placed, allows p2 to be placed (ordering)
"""

from __future__ import annotations

DOMAIN_NAME = "manufacturing-recovery"

# ---------------------------------------------------------------------------
# Semantic mapping: tools.json state name → (kind, pddl_predicate_name)
# ---------------------------------------------------------------------------
# Extend this dict when new state names are introduced in tools.json.
STATE_SEMANTICS: dict[str, tuple[str, str]] = {
    "idle":            ("resource_only", "idle"),
    "at_pick":         ("resource_only", "at-source"),
    "at_destination":  ("resource_only", "at-destination"),
    "ready":           ("resource_only", "idle"),          # printer/cnc ready state
    "picked":          ("resource_part", "carrying"),
    "positioned":      ("resource_part", "resource-positioned"),
    "processing":      ("resource_part", "resource-processing"),  # generic mid-operation
    "placed":          ("part_location", "part-placed"),
    "verified":        ("part_location", "part-placed"),
    "printed":         ("part_location", "part-at"),
    "available":       ("part_location", "part-at"),
    "misplaced":       ("part_location", "part-at"),
    "machined":        ("part_location", "part-at"),
}

# Always-present predicates (not derived from state names)
_FIXED_PREDICATES = """\
    ; Capability
    (resource-available ?r - resource)          ; resource is operational (not broken/down)
    (reachable ?r - resource ?l - location)     ; resource can physically reach this location

    ; Safety ordering
    (placement-allowed ?p - part)               ; this part may be finally placed
    (placed-flag ?p - part)                     ; set when part is finally placed"""

# Always-present recovery actions
_STAGE_PART_ACTION = """\
  ; Put held part at an intermediate location (frees gripper, no ordering check)
  (:action stage-part
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (resource-available ?r)
      (carrying ?r ?p)
      (reachable ?r ?l)
    )
    :effect (and
      (part-at ?p ?l)
      (idle ?r)
      (not (carrying ?r ?p))
    )
  )"""

_UNLOCK_PLACEMENT_ACTION = """\
  ; Unlock final placement of ?p2 once ?p1 is placed (safety ordering)
  (:action unlock-placement
    :parameters (?p1 - part ?p2 - part)
    :precondition (placed-flag ?p1)
    :effect (placement-allowed ?p2)
  )"""


# ---------------------------------------------------------------------------
# Predicate rendering helpers
# ---------------------------------------------------------------------------

def _predicate_str(state: str, r: str = "?r", p: str = "?p", l: str = "?l") -> str | None:
    """Render a state name as a PDDL predicate string. Returns None if unknown."""
    sem = STATE_SEMANTICS.get(state)
    if sem is None:
        return None
    kind, pred = sem
    if kind == "resource_only":
        return f"({pred} {r})"
    elif kind == "resource_part":
        return f"({pred} {r} {p})"
    else:  # part_location
        return f"({pred} {p} {l})"


def _state_kind(state: str) -> str:
    sem = STATE_SEMANTICS.get(state)
    return sem[0] if sem else "unknown"


# ---------------------------------------------------------------------------
# Action generator
# ---------------------------------------------------------------------------

def _generate_action(fn: str, in_state: str, out_state: str, pick_prereq_state: str | None) -> str:
    """
    Generate a PDDL action block for a tools.json function.

    pick_prereq_state: for pick-type actions (part_location → resource_part),
    the resource must already be in this resource_only state (e.g. 'at_pick').
    """
    action_name = fn.replace("_", "-")
    in_kind  = _state_kind(in_state)
    out_kind = _state_kind(out_state)

    in_pred  = _predicate_str(in_state)
    out_pred = _predicate_str(out_state)

    transition = (in_kind, out_kind)

    # ------------------------------------------------------------------
    # Pattern 1: resource_only → resource_only  (pure movement)
    # ------------------------------------------------------------------
    if transition == ("resource_only", "resource_only"):
        return f"""\
  (:action {action_name}
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (resource-available ?r)
      {in_pred}
      (reachable ?r ?l)
    )
    :effect (and
      {out_pred}
      (not {in_pred})
    )
  )"""

    # ------------------------------------------------------------------
    # Pattern 2: part_location → resource_part  (pick)
    # ------------------------------------------------------------------
    if transition == ("part_location", "resource_part"):
        in_part_pred = _predicate_str(in_state)
        prereq_pred  = _predicate_str(pick_prereq_state) if pick_prereq_state else None
        prereq_line  = f"\n      {prereq_pred}" if prereq_pred else ""
        return f"""\
  (:action {action_name}
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (resource-available ?r){prereq_line}
      {in_part_pred}
      (reachable ?r ?l)
    )
    :effect (and
      {out_pred}
      (not {in_part_pred}){f'{chr(10)}      (not {prereq_pred})' if prereq_pred else ''}
    )
  )"""

    # ------------------------------------------------------------------
    # Pattern 3: resource_part → resource_part  (carry / move-loaded)
    # ------------------------------------------------------------------
    if transition == ("resource_part", "resource_part"):
        return f"""\
  (:action {action_name}
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (resource-available ?r)
      {in_pred}
      (reachable ?r ?l)
    )
    :effect (and
      {out_pred}
      (not {in_pred})
    )
  )"""

    # ------------------------------------------------------------------
    # Pattern 4: resource_part → part_location  (final place)
    # ------------------------------------------------------------------
    if transition == ("resource_part", "part_location"):
        in_res_pred   = _predicate_str(in_state)
        out_part_pred = _predicate_str(out_state)
        return f"""\
  (:action {action_name}
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (resource-available ?r)
      {in_res_pred}
      (reachable ?r ?l)
      (placement-allowed ?p)
    )
    :effect (and
      {out_part_pred}
      (placed-flag ?p)
      (idle ?r)
      (not {in_res_pred})
    )
  )"""

    return f"  ; WARNING: no action template for {fn} ({in_state} → {out_state})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_domain(tools_catalog: list[dict], domain_name: str = DOMAIN_NAME) -> str:
    """
    Generate a PDDL domain string from the tools catalog.

    Args:
        tools_catalog: list of tool dicts from tools.json
        domain_name:   PDDL domain name

    Returns:
        Full PDDL domain string ready to pass to the planner.
    """
    # --- Deduplicate by function name ---
    seen: dict[str, dict] = {}
    for tool in tools_catalog:
        fn = tool.get("function")
        if fn and fn not in seen:
            seen[fn] = tool
    unique_tools = list(seen.values())

    # --- Collect all state names used ---
    all_in_states:  set[str] = set()
    all_out_states: set[str] = set()
    for tool in unique_tools:
        if tool.get("in_state"):  all_in_states.add(tool["in_state"])
        if tool.get("out_state"): all_out_states.add(tool["out_state"])

    # --- Find pick-prereq state: resource_only non-idle out_state ---
    pick_prereq_state: str | None = None
    for s in all_out_states:
        sem = STATE_SEMANTICS.get(s)
        if sem and sem[0] == "resource_only" and s != "idle" and s != "ready":
            pick_prereq_state = s
            break

    # --- Collect unique PDDL predicates from all state names ---
    pddl_predicates: dict[str, str] = {}
    for state in all_in_states | all_out_states:
        sem = STATE_SEMANTICS.get(state)
        if sem is None:
            continue
        kind, pred = sem
        if pred in pddl_predicates:
            continue
        if kind == "resource_only":
            pddl_predicates[pred] = f"({pred} ?r - resource)"
        elif kind == "resource_part":
            pddl_predicates[pred] = f"({pred} ?r - resource ?p - part)"
        else:
            pddl_predicates[pred] = f"({pred} ?p - part ?l - location)"

    predicates_block = (
        "    ; Resource / part states\n    "
        + "\n    ".join(pddl_predicates.values())
        + "\n"
        + _FIXED_PREDICATES
    )

    # --- Generate actions ---
    action_blocks: list[str] = []
    for tool in unique_tools:
        in_state  = tool.get("in_state", "")
        out_state = tool.get("out_state", "")
        if not in_state or not out_state:
            continue
        action_blocks.append(_generate_action(
            tool["function"], in_state, out_state, pick_prereq_state
        ))

    action_blocks.append(_STAGE_PART_ACTION)
    action_blocks.append(_UNLOCK_PLACEMENT_ACTION)

    actions_text = "\n\n".join(action_blocks)

    return f"""\
(define (domain {domain_name})
  (:requirements :strips :typing :negative-preconditions)

  (:types resource part location - object)

  (:predicates
{predicates_block}
  )

{actions_text}
)
"""


def get_domain(tools_catalog: list[dict] | None = None) -> str:
    """Return PDDL domain. If tools_catalog is None, loads from default tools.json."""
    if tools_catalog is None:
        import json
        from pathlib import Path
        tools_path = Path("cais_spade_llm/initialization/tools.json")
        tools_catalog = json.loads(tools_path.read_text())
    return generate_domain(tools_catalog)
