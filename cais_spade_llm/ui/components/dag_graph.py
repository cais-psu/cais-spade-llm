"""Convert plan nodes to Mermaid flowchart syntax for DAG visualization."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


# Mermaid class-name → status mapping.
_STATUS_CLASS = {
    "pending": "pending",
    "dispatched": "dispatched",
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "blocked": "blocked",
}


def _safe_mermaid_id(task_id: str) -> str:
    return str(task_id).replace("-", "_").replace(".", "_")


def _short_resource_label(value: Any) -> str:
    token = str(value or "").strip()
    if "@" in token:
        token = token.split("@", 1)[0]
    return token.strip()


def _first_non_empty_string(*values: Any) -> str:
    for value in values:
        if not isinstance(value, str):
            continue
        token = value.strip()
        if token:
            return token
    return ""


def _task_part_label(node: dict[str, Any]) -> str:
    params = node.get("params")
    params = params if isinstance(params, dict) else {}
    return _first_non_empty_string(
        params.get("part_name"),
        params.get("part"),
        node.get("part_name"),
        node.get("part"),
        node.get("product"),
    )


def _task_resource_label(node: dict[str, Any]) -> str:
    params = node.get("params")
    params = params if isinstance(params, dict) else {}
    return _short_resource_label(
        _first_non_empty_string(
            node.get("resource_jid"),
            params.get("resource_jid"),
            node.get("resource"),
            params.get("resource"),
            node.get("function_owner_agent"),
        )
    )


def _task_metadata_label(node: dict[str, Any]) -> str:
    parts: list[str] = []
    part = _task_part_label(node)
    resource = _task_resource_label(node)
    if part:
        parts.append(f"part: {part}")
    if resource:
        parts.append(f"robot: {resource}")
    return " | ".join(parts)


def build_fsa_dag_overlay(
    nodes: list[dict[str, Any]],
    global_fsa: dict[str, Any] | None,
) -> dict[str, dict[str, str]]:
    """Build a UI-only overlay describing which DAG roots are FSA-startable."""
    tasks = [
        node for node in (nodes or [])
        if isinstance(node, dict) and str(node.get("id") or node.get("task_id") or "").strip()
    ]
    if not tasks or not isinstance(global_fsa, dict):
        return {}

    task_lookup = {
        str(node.get("id") or node.get("task_id") or "").strip(): node
        for node in tasks
    }
    root_ids = {
        task_id
        for task_id, node in task_lookup.items()
        if not list(node.get("predecessors", []) or [])
    }

    automaton = global_fsa.get("A") or {}
    transitions = automaton.get("Tr") or []
    x0 = str(automaton.get("x0") or "").strip()
    if not x0 or not isinstance(transitions, list):
        return {}

    startable_ids: list[str] = []
    startable_by_resource: dict[str, list[str]] = defaultdict(list)
    for transition in transitions:
        if not isinstance(transition, dict):
            continue
        if str(transition.get("from") or "").strip() != x0:
            continue
        if not str(transition.get("event") or "").strip().endswith(".start"):
            continue
        task_id = str(transition.get("task_id") or "").strip()
        if not task_id or task_id not in task_lookup or task_id in startable_ids:
            continue
        startable_ids.append(task_id)
        resource_jid = str(transition.get("resource_jid") or "").strip()
        if resource_jid:
            startable_by_resource[resource_jid].append(task_id)

    if not startable_ids:
        return {}

    resource_block_labels: dict[str, list[str]] = {}
    raw_blocks = (global_fsa.get("meta") or {}).get("resource_requirement_blocks") or {}
    if isinstance(raw_blocks, dict):
        for resource_jid, blocks in raw_blocks.items():
            if not isinstance(blocks, list):
                continue
            labels: list[str] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                label = str(
                    block.get("label")
                    or block.get("requirement_id")
                    or block.get("block_id")
                    or ""
                ).strip()
                if label.startswith("task:"):
                    label = label.split(":", 1)[1]
                if label:
                    labels.append(label)
            if labels:
                resource_block_labels[str(resource_jid)] = labels

    overlay: dict[str, dict[str, str]] = {}
    for task_id in startable_ids:
        overlay[task_id] = {
            "label_suffix": "FSA-startable",
            "style": "stroke:#0f766e,stroke-width:3px",
        }

    for task_id in sorted(root_ids):
        if task_id in overlay:
            continue
        node = task_lookup.get(task_id) or {}
        resource_jid = str(node.get("resource_jid") or "").strip()
        if not resource_jid or not startable_by_resource.get(resource_jid):
            continue
        label_suffix = f"resource-blocked:{resource_jid}"
        block_order = resource_block_labels.get(resource_jid, [])
        if block_order:
            label_suffix += f" [{' -> '.join(block_order)}]"
        overlay[task_id] = {
            "label_suffix": label_suffix,
            "style": "stroke:#b45309,stroke-width:3px,stroke-dasharray: 6 3",
        }

    return overlay


def nodes_to_mermaid(
    nodes: list[dict[str, Any]],
    task_states: dict[str, str] | None = None,
    overlay: dict[str, dict[str, str]] | None = None,
) -> str:
    """Build a Mermaid flowchart string from plan nodes.

    Each node dict is expected to have at least:
      - task_id: str
      - function_name or instruction: str  (label)
      - predecessors: list[str]  (upstream task_ids)
      - status: str  (optional, overridden by task_states if provided)
    """
    if not nodes:
        return "graph LR\n    empty[No plan loaded]"

    task_states = task_states or {}
    overlay = overlay or {}
    lines = ["graph LR"]

    # Define style classes.
    lines.append("    classDef pending fill:#9e9e9e,color:#fff")
    lines.append("    classDef dispatched fill:#42a5f5,color:#fff")
    lines.append("    classDef running fill:#ffa726,color:#fff")
    lines.append("    classDef completed fill:#66bb6a,color:#fff")
    lines.append("    classDef failed fill:#ef5350,color:#fff")
    lines.append("    classDef blocked fill:#ff7043,color:#fff")

    for node in nodes:
        tid = node.get("id") or node.get("task_id", "?")
        func = node.get("function_name") or node.get("instruction", "")
        if isinstance(func, dict):
            func = func.get("function_name", str(func))
        label = f"{tid}<br/>{func}" if func else tid
        metadata_label = _task_metadata_label(node)
        if metadata_label:
            label = f"{label}<br/>{metadata_label}"
        overlay_entry = overlay.get(str(tid), {})
        label_suffix = str(overlay_entry.get("label_suffix", "")).strip()
        if label_suffix:
            label = f"{label}<br/>{label_suffix}"
        # Truncate long labels.
        if len(str(label)) > 170:
            label = str(label)[:167] + "..."
        # Sanitize for Mermaid.
        safe_label = str(label).replace('"', "'")
        safe_tid = _safe_mermaid_id(str(tid))

        # Determine status.
        status = task_states.get(tid, node.get("status", "pending")).lower()
        for key in _STATUS_CLASS:
            if key in status:
                status = key
                break
        else:
            status = "pending"

        lines.append(f'    {safe_tid}["{safe_label}"]:::{status}')

    # Edges from predecessors.
    for node in nodes:
        tid = _safe_mermaid_id(str(node.get("id") or node.get("task_id", "?")))
        for pred_id in node.get("predecessors", []):
            safe_pred = _safe_mermaid_id(str(pred_id))
            lines.append(f"    {safe_pred} --> {tid}")

    for task_id, overlay_entry in overlay.items():
        style = str(overlay_entry.get("style", "")).strip()
        if style:
            lines.append(f"    style {_safe_mermaid_id(task_id)} {style}")

    return "\n".join(lines)
