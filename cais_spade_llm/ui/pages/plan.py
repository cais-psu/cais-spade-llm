"""Plan page: task DAG visualization + execution timeline."""

from __future__ import annotations

import json

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.dag_graph import nodes_to_mermaid


def render(bridge: SystemBridge) -> None:
    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-6"):
        ui.label("Plan View").classes("text-2xl font-bold")

        # ── DAG Visualization ────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Task DAG").classes("text-lg font-semibold mb-2")
            mermaid = ui.mermaid("graph TD\n    empty[No plan loaded]").classes("w-full")

            def _refresh_dag():
                nodes = bridge.get_plan_nodes()
                task_states = bridge.get_task_states()
                mermaid.content = nodes_to_mermaid(nodes, task_states)

            ui.timer(2.0, _refresh_dag)

        # ── Task States Table ────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Task States").classes("text-lg font-semibold mb-2")
            task_table = ui.table(
                columns=[
                    {"name": "task_id", "label": "Task ID", "field": "task_id", "sortable": True},
                    {"name": "status", "label": "Status", "field": "status", "sortable": True},
                ],
                rows=[],
            ).classes("w-full")

            def _refresh_tasks():
                ts = bridge.get_task_states()
                task_table.rows = [{"task_id": k, "status": v} for k, v in ts.items()]

            ui.timer(2.0, _refresh_tasks)

        # ── Execution Timeline ───────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Execution Timeline").classes("text-lg font-semibold mb-2")
            timeline_table = ui.table(
                columns=[
                    {"name": "timestamp", "label": "Time", "field": "timestamp", "sortable": True},
                    {"name": "task_id", "label": "Task", "field": "task_id", "sortable": True},
                    {"name": "status", "label": "Status", "field": "status", "sortable": True},
                    {"name": "resource_jid", "label": "Resource", "field": "resource_jid"},
                ],
                rows=[],
            ).classes("w-full")

            def _refresh_timeline():
                tl = bridge.get_execution_timeline()
                timeline_table.rows = tl[-50:]  # Show last 50 events.

            ui.timer(3.0, _refresh_timeline)

        # ── Node Detail ──────────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Node Detail (click a task row above)").classes("text-lg font-semibold mb-2")
            detail_pre = ui.code("Select a task to view details", language="json").classes("w-full")

            def _on_task_click(e):
                row = e.args.get("row", {}) if isinstance(e.args, dict) else {}
                tid = row.get("task_id", "")
                nodes = bridge.get_plan_nodes()
                node = next((n for n in nodes if n.get("task_id") == tid), None)
                if node:
                    detail_pre.content = json.dumps(node, indent=2, default=str)
                else:
                    detail_pre.content = f"No node found for {tid}"

            task_table.on("rowClick", _on_task_click)
