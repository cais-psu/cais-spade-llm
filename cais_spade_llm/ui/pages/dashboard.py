"""Read-only overview of the current manufacturing system."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from nicegui import context, ui

from cais_spade_llm.ui.bridge import SystemBridge


def read_status(bridge: SystemBridge) -> dict[str, Any]:
    """Read current status without selecting a configuration or starting work."""
    return {
        "system_running": bridge.system_running,
        "starting": bridge._starting,
        "stopping": bridge._stopping,
        "execution_mode": bridge.execution_mode,
        "selected_product": bridge.selected_product,
        "selected_product_order_file": bridge.selected_product_order_file,
        "agents": bridge.get_agent_statuses(),
        "resources": bridge.get_robot_states(),
        "tasks": bridge.get_task_states(),
        "safety": bridge.get_safety_state(),
        "events": bridge.get_execution_timeline()[-50:],
        "last_error": bridge.last_error,
    }


def render(bridge: SystemBridge) -> None:
    """Render live status and links to the project workflows."""
    client = context.client
    with ui.column().classes("w-full gap-4 p-6"):
        ui.label("dashboard").classes("text-2xl font-bold")
        with ui.row().classes("gap-4"):
            ui.link("recovery-framework", "/recovery-framework")
            ui.link("spec2primitives", "/spec2primitives")
        with ui.card().classes("w-full"):
            ui.label("Current system").classes("text-lg font-semibold")
            system = ui.label()
            product = ui.label().classes("break-all")
            order = ui.label().classes("break-all")
            error = ui.label().classes("text-red-700 whitespace-pre-wrap")
        with ui.card().classes("w-full"):
            ui.label("Task progress").classes("text-lg font-semibold")
            progress = ui.label()
            tasks = ui.table(
                columns=[
                    {"name": "task_id", "field": "task_id", "label": "Task"},
                    {"name": "status", "field": "status", "label": "Status"},
                ],
                rows=[],
                row_key="task_id",
                pagination=10,
            ).classes("w-full")
        with ui.card().classes("w-full"):
            ui.label("Agent status").classes("text-lg font-semibold")
            agents = ui.table(
                columns=[
                    {"name": key, "field": key, "label": key}
                    for key in ("name", "jid", "type", "alive")
                ],
                rows=[],
                row_key="jid",
                pagination=10,
            ).classes("w-full")
        with ui.card().classes("w-full"):
            ui.label("Resource status").classes("text-lg font-semibold")
            resources = ui.table(
                columns=[
                    {"name": "resource", "field": "resource", "label": "Resource"},
                    {"name": "state", "field": "state", "label": "Current state"},
                ],
                rows=[],
                row_key="resource",
                pagination=10,
            ).classes("w-full")
        with ui.card().classes("w-full"):
            ui.label("Safety alerts").classes("text-lg font-semibold")
            safety = ui.code("{}", language="json").classes("w-full")
        with ui.card().classes("w-full"):
            ui.label("Recent events").classes("text-lg font-semibold")
            events = ui.table(
                columns=[
                    {"name": key, "field": key, "label": key}
                    for key in ("timestamp", "task_id", "status", "resource_jid")
                ],
                rows=[],
                row_key="row_id",
                pagination=10,
            ).classes("w-full")

    def refresh() -> None:
        if getattr(client, "_deleted", False):
            return
        status = read_status(bridge)
        state = "Running" if status["system_running"] else "Stopped"
        if status["starting"]:
            state = "Starting..."
        if status["stopping"]:
            state = "Stopping..."
        system.text = f"{state} · {status['execution_mode']}"
        product.text = f"Product: {status['selected_product'] or 'not selected'}"
        order.text = f"Order: {status['selected_product_order_file'] or 'not selected'}"
        error.text = status["last_error"] or ""
        counts = Counter(status["tasks"].values())
        progress.text = (
            " · ".join(f"{state}: {count}" for state, count in counts.items()) or "No tasks"
        )
        tasks.rows = [{"task_id": key, "status": value} for key, value in status["tasks"].items()]
        agents.rows = status["agents"]
        resources.rows = [
            {"resource": key, "state": json.dumps(value, ensure_ascii=False, default=str)}
            for key, value in status["resources"].items()
        ]
        safety.content = json.dumps(status["safety"], indent=2, ensure_ascii=False, default=str)
        events.rows = [{**event, "row_id": index} for index, event in enumerate(status["events"])]

    refresh()
    timer = ui.timer(2.0, refresh)
    client.on_delete(lambda: timer.cancel(with_current_invocation=True))
