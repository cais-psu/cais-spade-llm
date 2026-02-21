"""
Build the PDDL problem for manufacturing recovery replanning.

State tokens (resource-state / part-state) are collected dynamically from tools.json
so the domain never needs to change when new resource types or tools are added.

Build order:
  1. build_objects() — declares all typed objects (resources, parts, contexts, states)
  2. build_init()    — asserts facts true at recovery start
  3. build_goal()    — specifies what must be true at end
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TOOLS_JSON_PATH = Path("cais_spade_llm/initialization/tools.json")

DOMAIN_NAME = "manufacturing-recovery"
PROBLEM_NAME = "recovery"


def pddl_name(raw: str) -> str:
    """Convert a JID or identifier to a PDDL-safe name (no @, ., spaces)."""
    return raw.replace("@", "-").replace(".", "-").replace(" ", "-")


def build_objects(
    tools: list[dict[str, Any]],
    resource_jids: list[str],
    part_names: list[str],
    context_names: list[str] | None = None,
) -> str:
    """
    Build the (:objects ...) block.

    Args:
        tools:         Parsed tools.json — used to collect state tokens.
        resource_jids: JIDs of all resources in the system (e.g. "ur5e@localhost").
        part_names:    Names of all parts involved (e.g. ["SG", "MCP"]).
        context_names: Named locations used as origins/destinations
                       (e.g. ["prusa-mk4-1", "assembly-board-v1", "ur5e-region"]).
                       If None or empty, no context objects are emitted.

    Returns:
        PDDL (:objects ...) string.
    """
    resource_states, part_states = collect_states_from_tools(tools)

    resources = " ".join(pddl_name(jid) for jid in resource_jids)
    parts = " ".join(pddl_name(p) for p in part_names)
    rstates = " ".join(resource_states)
    pstates = " ".join(part_states)

    lines = [
        "  (:objects",
        f"    {resources} - resource",
        f"    {parts} - part",
        f"    {rstates} - resource-state",
        f"    {pstates} - part-state",
    ]

    if context_names:
        contexts = " ".join(pddl_name(c) for c in context_names)
        lines.append(f"    {contexts} - context")

    lines.append("  )")
    return "\n".join(lines)


def build_init(
    tools: list[dict[str, Any]],
    resource_states: dict[str, str],
    part_states: dict[str, str],
    reachability: dict[str, list[str]] | None = None,
    part_locations: dict[str, str] | None = None,
) -> str:
    """
    Build the (:init ...) block from the current system state at recovery start.

    Only emits facts for state tokens declared in (:objects ...).
    Resources whose current_state is not a known PDDL token (e.g. "recovery_required")
    are silently skipped — the planner cannot use them, which is correct behaviour
    for a broken or unavailable resource.

    Args:
        tools:           Parsed tools.json — used to validate known state tokens.
        resource_states: JID → current resource state token.
                         e.g. {"ur5e@localhost": "idle", "xarm6@localhost": "recovery_required"}
        part_states:     Part name → current part state token.
                         e.g. {"SG": "ready", "MCP": "in_transit"}
        reachability:    JID → list of context names the resource can reach.
                         e.g. {"ur5e@localhost": ["prusa-mk4-2", "ur5e-region", "assembly-board-v1"]}
                         These become (reachable ?r ?c) facts.
        part_locations:  Part name → context name where the part currently resides.
                         e.g. {"SG": "ur5e-region", "MCP": "prusa-mk4-2"}
                         These become (part-at ?p ?c) facts.
                         Omit parts that are in-gripper (no physical location).

    Returns:
        PDDL (:init ...) string.
    """
    valid_resource_states, valid_part_states = collect_states_from_tools(tools)
    valid_rstates = set(valid_resource_states)
    valid_pstates = set(valid_part_states)

    facts: list[str] = []

    for jid, state in resource_states.items():
        if state in valid_rstates:
            facts.append(f"    (resource-in-state {pddl_name(jid)} {state})")
        # else: unknown state (e.g. recovery_required) → resource skipped by planner

    for part, state in part_states.items():
        if state in valid_pstates:
            facts.append(f"    (part-in-state {pddl_name(part)} {state})")
        # else: unknown part state → no fact asserted

    if reachability:
        for jid, contexts in reachability.items():
            for ctx in contexts:
                facts.append(f"    (reachable {pddl_name(jid)} {pddl_name(ctx)})")

    if part_locations:
        for part, ctx in part_locations.items():
            facts.append(f"    (part-at {pddl_name(part)} {pddl_name(ctx)})")

    lines = ["  (:init", *facts, "  )"]
    return "\n".join(lines)


def collect_states_from_tools(
    tools: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """
    Scan tools.json entries and return all unique state token names.

    Returns:
        (resource_states, part_states)
        resource_states: unique values from in_state / out_state fields
        part_states: unique values from part_in_state and part_transition outcome states
    """
    rstates: set[str] = set()
    pstates: set[str] = set()

    for tool in tools:
        if "in_state" in tool:
            rstates.add(tool["in_state"])
        if "out_state" in tool:
            rstates.add(tool["out_state"])
        if "part_in_state" in tool:
            pstates.add(tool["part_in_state"])
        for outcome in tool.get("part_transition", {}).values():
            if "state" in outcome:
                pstates.add(outcome["state"])

    return sorted(rstates), sorted(pstates)


def build_goal(
    part_names: list[str],
    target_state: str = "assembled",
) -> str:
    """
    Build the (:goal ...) block.

    Args:
        part_names:   Names of all parts that must reach the target state.
        target_state: The part-state token all parts must be in at plan end.
                      Defaults to "assembled" (the final state in tools.json).

    Returns:
        PDDL (:goal ...) string.
    """
    conjuncts = [
        f"      (part-in-state {pddl_name(p)} {target_state})"
        for p in part_names
    ]
    lines = [
        "  (:goal",
        "    (and",
        *conjuncts,
        "    )",
        "  )",
    ]
    return "\n".join(lines)


def build_problem(
    objects: str,
    init: str,
    goal: str,
    problem_name: str = PROBLEM_NAME,
    domain_name: str = DOMAIN_NAME,
) -> str:
    """
    Assemble the full PDDL problem string from its three blocks.

    Args:
        objects:      Output of build_objects().
        init:         Output of build_init().
        goal:         Output of build_goal().
        problem_name: PDDL problem name (default "recovery").
        domain_name:  PDDL domain name (default "manufacturing-recovery").

    Returns:
        Complete PDDL problem string ready for the solver.
    """
    lines = [
        f"(define (problem {problem_name})",
        f"  (:domain {domain_name})",
        objects,
        init,
        goal,
        ")",
    ]
    return "\n".join(lines)


def load_tools(tools_path: Path = TOOLS_JSON_PATH) -> list[dict[str, Any]]:
    with tools_path.open() as f:
        return json.load(f)
