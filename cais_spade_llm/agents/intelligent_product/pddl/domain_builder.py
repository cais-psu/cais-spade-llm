"""
Build the PDDL domain for manufacturing recovery replanning.

Actions are generated per-tool from tools.json so the domain stays in sync
with the actual tool catalogue.

Types:
  resource       - robots, printers, CNC machines — any active agent
  part           - physical parts being processed or assembled
  resource-state - state token: values from tools.json in_state/out_state
  part-state     - state token: values from tools.json part_in_state/part_transition
  context        - named location (origin/destination for tools with required_context_keys)

Predicates:
  (resource-in-state ?r - resource  ?s - resource-state)
  (part-in-state     ?p - part      ?s - part-state)
  (part-at           ?p - part      ?c - context)   ; part is at a physical location
  (reachable         ?r - resource  ?c - context)   ; resource can reach a context

Per-tool action generation:
  - Each unique (resource_type, function) pair → one PDDL action
  - Tools with required_context_keys get a ?c - context parameter
  - origin context: requires (reachable ?r ?c) AND (part-at ?p ?c); removes (part-at ?p ?c)
  - destination context: requires (reachable ?r ?c); asserts (part-at ?p ?c) on completion
"""

from __future__ import annotations

from typing import Any


DOMAIN_NAME = "manufacturing-recovery"


def build_domain(tools: list[dict[str, Any]]) -> str:
    """
    Build and return the PDDL domain string.

    Args:
        tools: Parsed contents of tools.json.
    """
    action_lines = _build_actions(tools)

    lines: list[str] = [
        f"(define (domain {DOMAIN_NAME})",
        "  (:requirements :strips :typing :negative-preconditions)",
        "",
        "  (:types",
        "    resource       ; robots, printers, CNC machines — any active agent",
        "    part           ; physical parts being processed or assembled",
        "    resource-state ; named resource state — values from tools.json in_state/out_state",
        "    part-state     ; named part state     — values from tools.json part_in_state/part_transition",
        "    context        ; named location/context — origin or destination",
        "  )",
        "",
        "  (:predicates",
        "    (resource-in-state  ?r - resource  ?s - resource-state)",
        "    (part-in-state      ?p - part       ?s - part-state)",
        "    (part-at            ?p - part       ?c - context)",
        "    (reachable          ?r - resource   ?c - context)",
        "  )",
        "",
        "  ; Actions derived from tools.json",
        "",
        *action_lines,
        ")",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_actions(tools: list[dict[str, Any]]) -> list[str]:
    """
    Generate PDDL actions from tools, deduplicating by (resource_type, function).

    Each unique function within a resource_type gets one PDDL action.
    Resources of the same type share actions — spatial reachability is
    captured by (reachable ?r ?c) facts in the problem init block.
    """
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for tool in tools:
        key = (tool.get("resource_type", "unknown"), tool["function"])
        if key not in seen:
            seen.add(key)
            deduped.append(tool)

    lines: list[str] = []
    current_rtype: str | None = None
    for tool in deduped:
        rtype = tool.get("resource_type", "unknown")
        if rtype != current_rtype:
            lines.append(f"  ; --- {rtype} actions ---")
            lines.append("")
            current_rtype = rtype
        lines.extend(_build_action(tool))

    return lines


def _build_action(tool: dict[str, Any]) -> list[str]:
    """Build a single PDDL action from one tool entry."""
    action_name = tool["function"].replace("_", "-")
    r_pre = tool["in_state"]
    r_post = tool["out_state"]
    context_keys: list[str] = tool.get("required_context_keys", [])
    has_context = bool(context_keys)
    has_part = "part_in_state" in tool

    # Parameters
    if has_context and has_part:
        params = "(?r - resource ?p - part ?c - context)"
    elif has_context:
        params = "(?r - resource ?c - context)"
    elif has_part:
        params = "(?r - resource ?p - part)"
    else:
        params = "(?r - resource)"

    # Preconditions
    precond_facts = [f"(resource-in-state ?r {r_pre})"]
    if has_part:
        precond_facts.append(f"(part-in-state ?p {tool['part_in_state']})")
    if has_context:
        precond_facts.append("(reachable ?r ?c)")
        # Origin pick: the part must be physically at that context location
        if "origin" in context_keys and has_part:
            precond_facts.append("(part-at ?p ?c)")

    if len(precond_facts) == 1:
        precond = precond_facts[0]
    else:
        inner = "\n               ".join(precond_facts)
        precond = f"(and {inner})"

    # Effects
    p_pre = tool.get("part_in_state")
    p_post = tool.get("part_transition", {}).get("completed", {}).get("state")

    effect_facts = [
        f"(resource-in-state ?r {r_post})",
        f"(not (resource-in-state ?r {r_pre}))",
    ]
    if has_part and p_pre and p_post:
        effect_facts.append(f"(part-in-state ?p {p_post})")
        effect_facts.append(f"(not (part-in-state ?p {p_pre}))")
        # Origin pick removes part-at; destination place/assemble adds part-at
        if "origin" in context_keys:
            effect_facts.append("(not (part-at ?p ?c))")
        elif "destination" in context_keys:
            effect_facts.append("(part-at ?p ?c)")

    if len(effect_facts) == 1:
        effect = effect_facts[0]
    else:
        inner = "\n              ".join(effect_facts)
        effect = f"(and {inner})"

    return [
        f"  (:action {action_name}",
        f"    :parameters {params}",
        f"    :precondition {precond}",
        f"    :effect {effect}",
        "  )",
        "",
    ]
