"""
PDDL problem generator for assembly recovery planning.

Builds a PDDL problem file from:
  - system_state   : robot states, part locations (from CCA + ProductAgent)
  - resource_infos : robot static_capabilities (workspace_boundaries, staging_areas, reachability)
  - pending_goals  : list of {part, destination} dicts — what still needs to be placed
  - safety_rules   : from cca_safety_logic.json — used to derive ordering constraints

Reachability is computed two ways:
  1. Named locations (printers, assembly board): use robot's `reachability` list
  2. Coordinate locations (misplaced parts): check workspace_boundaries x/y/z ranges
"""

from __future__ import annotations
from typing import Any
from cais_spade_llm.pddl.pddl_domain import DOMAIN_NAME


# ---------------------------------------------------------------------------
# Reachability helpers
# ---------------------------------------------------------------------------

def _in_workspace(bounds: dict, x: float, y: float, z: float) -> bool:
    """Return True if (x, y, z) is within the robot's workspace_boundaries."""
    xr = bounds.get("x_range", [-9999, 9999])
    yr = bounds.get("y_range", [-9999, 9999])
    zr = bounds.get("z_range", [-9999, 9999])
    return xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1] and zr[0] <= z <= zr[1]


def _pddl_name(s: str) -> str:
    """Sanitise a string to a valid PDDL identifier (lowercase, hyphens)."""
    return s.lower().replace("_", "-").replace("@", "-").replace(".", "-")


# ---------------------------------------------------------------------------
# Ordering constraint extraction
# ---------------------------------------------------------------------------

def extract_ordering_constraints(safety_rules: list[dict]) -> list[tuple[str, str]]:
    """
    Parse safety rules for ordering constraints.
    Returns list of (prerequisite_part, dependent_part) pairs.

    Handles constraint_type == "ordering_place_before":
      product[0] must be placed before product[1].
    """
    ordering: list[tuple[str, str]] = []
    for rule in safety_rules:
        if rule.get("constraint_type") == "ordering_place_before":
            products = rule.get("product") or []
            if len(products) >= 2:
                ordering.append((products[0].lower(), products[1].lower()))
    return ordering


# ---------------------------------------------------------------------------
# Main problem builder
# ---------------------------------------------------------------------------

def build_problem(
    *,
    system_state: dict[str, Any],
    resource_infos: list[dict],
    pending_goals: list[dict],        # [{"part": "SG", "destination": "assembly_board-v1"}, ...]
    safety_rules: list[dict],
    problem_name: str = "recovery-problem",
) -> str:
    """
    Build a PDDL problem string for assembly recovery.

    Args:
        system_state:   Combined state from CCA + ProductAgent.
        resource_infos: List of {jid, static_capabilities} dicts.
        pending_goals:  Parts that still need to be placed and where.
        safety_rules:   Parsed safety rules from cca_safety_logic.json.
        problem_name:   Name for the PDDL problem instance.
    """

    robots_state: dict = system_state.get("robots") or {}
    parts_state: dict  = system_state.get("parts") or {}

    ordering_constraints = extract_ordering_constraints(safety_rules)

    # ------------------------------------------------------------------
    # 1. Collect all objects
    # ------------------------------------------------------------------

    resource_names = [_pddl_name(jid) for jid in robots_state]
    part_names   = list({_pddl_name(p) for p in parts_state})

    # Named locations: from reachability lists + staging areas
    named_locations: set[str] = set()
    for ri in resource_infos:
        caps = ri.get("static_capabilities") or {}
        for loc in caps.get("reachability", []):
            named_locations.add(_pddl_name(loc))
        for zone in (caps.get("staging_areas") or {}).keys():
            named_locations.add(_pddl_name(zone))

    # Coordinate locations: one per misplaced part
    coord_locations: dict[str, dict] = {}   # pddl_name → {x, y, z}
    for part_id, ps in parts_state.items():
        pos = ps.get("position")
        if pos and ps.get("state") in ("misplaced", "untracked", "lost"):
            loc_name = f"failed-loc-{_pddl_name(part_id)}"
            coord_locations[loc_name] = pos

    all_locations = sorted(named_locations | set(coord_locations.keys()))

    # ------------------------------------------------------------------
    # 2. Compute reachability facts
    # ------------------------------------------------------------------

    reachability_facts: list[str] = []

    for ri in resource_infos:
        robot_jid  = ri.get("jid", "")
        robot_pddl = _pddl_name(robot_jid)
        caps       = ri.get("static_capabilities") or {}
        reachable_named = [_pddl_name(l) for l in caps.get("reachability", [])]
        staging_names   = [_pddl_name(z) for z in (caps.get("staging_areas") or {}).keys()]
        bounds          = caps.get("workspace_boundaries") or {}

        for loc in reachable_named + staging_names:
            reachability_facts.append(f"(reachable {robot_pddl} {loc})")

        for loc_name, pos in coord_locations.items():
            if _in_workspace(bounds, pos["x"], pos["y"], pos["z"]):
                reachability_facts.append(f"(reachable {robot_pddl} {loc_name})")

    # ------------------------------------------------------------------
    # 3. Build :init facts
    # ------------------------------------------------------------------

    init_facts: list[str] = list(reachability_facts)

    # Robot states — broken/unavailable robots do NOT get (resource-available)
    BROKEN_STATES = {"broken", "offline", "error", "unavailable", "disconnected"}

    for jid, rs in robots_state.items():
        robot_pddl = _pddl_name(jid)
        held       = rs.get("held_part")
        state      = rs.get("current_state", "idle")

        if state in BROKEN_STATES:
            # Robot is broken — omit resource-available so no actions fire for it.
            # If it was carrying a part, that part is now stuck at the robot's
            # last known position; caller should add it as a coordinate location.
            continue

        # Robot is operational
        init_facts.append(f"(resource-available {robot_pddl})")

        if held:
            held_pddl = _pddl_name(held)
            init_facts.append(f"(carrying {robot_pddl} {held_pddl})")
        else:
            # recovery_required / positioned / idle all treated as idle —
            # the planner will replan from scratch for this robot.
            init_facts.append(f"(idle {robot_pddl})")

    for part_id, ps in parts_state.items():
        part_pddl = _pddl_name(part_id)
        pstate    = ps.get("state", "")
        pos       = ps.get("position")

        if pstate in ("misplaced", "untracked", "lost") and pos:
            # Part dropped — at its coordinate location
            loc_name = f"failed-loc-{part_pddl}"
            init_facts.append(f"(part-at {part_pddl} {loc_name})")

        elif pstate in ("picked", "in_transit"):
            # Part is held by a robot — no part-at fact needed (it's in carrying)
            pass

        elif pstate == "verified":
            # Already successfully placed — add placed-flag so it can unlock dependents
            # Find where it was placed
            loc = ps.get("location") or ps.get("last_known_location")
            if loc:
                loc_pddl = _pddl_name(loc)
                init_facts.append(f"(part-placed {part_pddl} {loc_pddl})")
                init_facts.append(f"(placed-flag {part_pddl})")

        else:
            # printed / available — at its last known location
            loc = ps.get("location") or ps.get("last_known_location")
            if loc:
                loc_pddl = _pddl_name(loc)
                init_facts.append(f"(part-at {part_pddl} {loc_pddl})")

    # Placement-allowed: parts with no ordering prerequisite are allowed immediately.
    # Parts that have a prerequisite are NOT allowed until unlock-placement fires.
    dependent_parts: set[str] = {dep for _, dep in ordering_constraints}
    for part_id in parts_state:
        part_pddl = _pddl_name(part_id)
        if part_pddl not in dependent_parts:
            init_facts.append(f"(placement-allowed {part_pddl})")

    # Parts already placed also satisfy placed-flag for ordering purposes
    # (already added above in the "verified" branch)

    # ------------------------------------------------------------------
    # 4. Build :goal
    # ------------------------------------------------------------------

    goal_parts: list[str] = []
    for g in pending_goals:
        part_pddl = _pddl_name(g["part"])
        dest_pddl = _pddl_name(g["destination"])
        goal_parts.append(f"(part-placed {part_pddl} {dest_pddl})")

    goal_str = "\n    ".join(goal_parts)

    # ------------------------------------------------------------------
    # 5. Assemble PDDL
    # ------------------------------------------------------------------

    objects_block = (
        f"    {' '.join(resource_names)} - resource\n"
        f"    {' '.join(part_names)} - part\n"
        f"    {' '.join(all_locations)} - location"
    )

    init_block = "\n    ".join(f"({f})" if not f.startswith("(") else f for f in init_facts)

    return f"""\
(define (problem {problem_name})
  (:domain {DOMAIN_NAME})

  (:objects
{objects_block}
  )

  (:init
    {init_block}
  )

  (:goal
    (and
    {goal_str}
    )
  )
)
"""