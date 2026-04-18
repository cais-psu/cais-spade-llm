"""Convert plan nodes to Mermaid flowchart syntax for DAG visualization."""

from __future__ import annotations

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


def nodes_to_mermaid(nodes: list[dict[str, Any]], task_states: dict[str, str] | None = None) -> str:
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
        # Truncate long labels.
        if len(str(label)) > 40:
            label = str(label)[:37] + "..."
        # Sanitize for Mermaid.
        safe_label = str(label).replace('"', "'")
        safe_tid = tid.replace("-", "_").replace(".", "_")

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
        tid = (node.get("id") or node.get("task_id", "?")).replace("-", "_").replace(".", "_")
        for pred_id in node.get("predecessors", []):
            safe_pred = pred_id.replace("-", "_").replace(".", "_")
            lines.append(f"    {safe_pred} --> {tid}")

    return "\n".join(lines)
