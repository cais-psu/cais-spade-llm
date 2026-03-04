"""Convert compiled global FSA JSON to Mermaid flowchart syntax."""

from __future__ import annotations

import re
from typing import Any


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _escape(text: str) -> str:
    return text.replace('"', "'")


def _collect_states(
    automaton: dict[str, Any],
    transitions: list[dict[str, Any]],
    current_state: str | None,
) -> list[str]:
    states_ordered: list[str] = []
    seen_states: set[str] = set()

    def _push_state(value: Any) -> None:
        state = str(value or "")
        if state and state not in seen_states:
            seen_states.add(state)
            states_ordered.append(state)

    for state in automaton.get("X") or []:
        _push_state(state)
    for tr in transitions:
        _push_state(tr.get("from"))
        _push_state(tr.get("to"))

    x0 = str(automaton.get("x0") or "")
    marked = {str(s) for s in (automaton.get("Xm") or [])}
    if x0:
        _push_state(x0)
    if current_state:
        _push_state(current_state)
    for state in marked:
        _push_state(state)

    return states_ordered


def _edge_label(transition: dict[str, Any]) -> str:
    event = str(transition.get("event") or "")
    readable = str(transition.get("readable_event") or "")
    function_name = str(transition.get("function_name") or "")
    phase = event.rsplit(".", 1)[-1] if "." in event else ""

    if not function_name and readable:
        # readable_event example: place_approach(LCP).start
        function_name = readable.split("(", 1)[0].split(".", 1)[0]
    core = f"{function_name}.{phase}" if function_name and phase else (readable or event)
    core = _trim(core, 26)

    resource = str(transition.get("resource_jid") or "")
    if resource:
        resource = resource.split("@", 1)[0]
        # Force a clean two-line label instead of random single-letter wrapping.
        return f"{resource}<br/>{core}"
    return core


def _compact_state_text(state: str) -> str:
    text = str(state or "")
    entries: list[str] = []
    pattern = re.compile(r"([^\s=,]+)=\(k=(\d+),(?:run=([^:,\)]+):([^)]+)|idle)\)")
    for match in pattern.finditer(text):
        resource = match.group(1).split("@", 1)[0]
        k = match.group(2)
        run_task = match.group(3)
        run_fn = match.group(4)
        if run_task and run_fn:
            entries.append(f"{resource}:k{k} run {run_task}/{run_fn}")
        else:
            entries.append(f"{resource}:k{k} idle")

    if entries:
        return _trim(" | ".join(entries), 62)
    return _trim(text.replace("@localhost", ""), 62)


def fsa_state_index_text(
    fsa: dict[str, Any] | None,
    *,
    current_state: str | None = None,
    max_states: int = 500,
) -> str:
    """Build a text index mapping S<n> node IDs back to full FSA state strings."""
    automaton = (fsa or {}).get("A") or {}
    transitions = automaton.get("Tr") or []
    if not automaton and not transitions:
        return "No global FSA loaded"

    states_ordered = _collect_states(automaton, transitions, current_state)
    if not states_ordered:
        return "No global FSA states"

    x0 = str(automaton.get("x0") or "")
    marked = {str(s) for s in (automaton.get("Xm") or [])}

    lines: list[str] = []
    for i, state in enumerate(states_ordered[:max_states]):
        tags: list[str] = []
        if state == x0:
            tags.append("x0")
        if state in marked:
            tags.append("Xm")
        if current_state and state == current_state:
            tags.append("current")
        tag_text = f" [{' | '.join(tags)}]" if tags else ""
        lines.append(f"S{i}{tag_text}: {state}")

    if len(states_ordered) > max_states:
        lines.append(f"... {len(states_ordered) - max_states} more states hidden")

    return "\n".join(lines)


def fsa_to_mermaid(
    fsa: dict[str, Any] | None,
    *,
    current_state: str | None = None,
    max_edges: int = 220,
) -> str:
    """Build Mermaid graph text from a compiled global FSA payload."""
    automaton = (fsa or {}).get("A") or {}
    transitions = automaton.get("Tr") or []
    if not automaton and not transitions:
        return "graph TB\n    empty[No global FSA loaded]"

    states_ordered = _collect_states(automaton, transitions, current_state)
    if not states_ordered:
        return "graph TB\n    empty[No global FSA states]"

    state_to_id = {state: f"S{i}" for i, state in enumerate(states_ordered)}
    x0 = str(automaton.get("x0") or "")
    marked = {str(s) for s in (automaton.get("Xm") or [])}

    lines = ["graph TB"]
    lines.append("    classDef normal fill:#f8fafc,color:#0f172a,stroke:#94a3b8,stroke-width:1.5px")
    lines.append("    classDef initial fill:#bfdbfe,color:#1e3a8a,stroke:#3b82f6,stroke-width:2.5px")
    lines.append("    classDef marked fill:#bbf7d0,color:#14532d,stroke:#22c55e,stroke-width:2.5px")
    lines.append("    classDef current fill:#fed7aa,color:#7c2d12,stroke:#f97316,stroke-width:3.5px")
    lines.append("    classDef current_marked fill:#fde68a,color:#78350f,stroke:#f59e0b,stroke-width:3.5px")
    lines.append("    classDef note fill:#e2e8f0,color:#334155,stroke:#94a3b8,stroke-dasharray: 3 3")
    lines.append("    linkStyle default stroke:#64748b,stroke-width:2.2px")

    for state in states_ordered:
        node_id = state_to_id[state]
        state_text = _compact_state_text(state)
        label = _escape(f"{node_id}<br/>{state_text}")
        if state == current_state and state in marked:
            node_class = "current_marked"
        elif state == current_state:
            node_class = "current"
        elif state in marked:
            node_class = "marked"
        elif state == x0:
            node_class = "initial"
        else:
            node_class = "normal"
        lines.append(f'    {node_id}["{label}"]:::{node_class}')

    for tr in transitions[:max_edges]:
        frm = str(tr.get("from") or "")
        to = str(tr.get("to") or "")
        if not frm or not to or frm not in state_to_id or to not in state_to_id:
            continue
        label = _escape(_edge_label(tr))
        lines.append(f'    {state_to_id[frm]} -- "{label}" --> {state_to_id[to]}')

    if len(transitions) > max_edges:
        hidden = len(transitions) - max_edges
        lines.append(f'    trunc["{hidden} transitions hidden (graph limit)"]:::note')

    return "\n".join(lines)
