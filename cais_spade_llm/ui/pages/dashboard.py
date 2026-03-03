"""Dashboard page: system control panel, agent overview, plan DAG, and execution timeline."""

from __future__ import annotations

import asyncio
import json

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.agent_chat import render_chat
from cais_spade_llm.ui.components.dag_graph import nodes_to_mermaid


# Execution mode labels → internal values.
_MODE_MAP = {
    "Dry Run": "simulate",
    "Simulation": "ros2",
    "Physical": "real",
}
_MODE_LABELS = list(_MODE_MAP.keys())


def render(bridge: SystemBridge) -> None:
    ui.label("Dashboard").classes("text-2xl font-bold px-6 pt-6")

    with ui.row().classes("w-full px-6 gap-6 items-start"):
      # ── Left column: existing dashboard content ───────────
      with ui.column().classes("flex-grow gap-6 min-w-0"):

        # ── System Control Card ──────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("System Control").classes("text-lg font-semibold mb-2")

            with ui.row().classes("items-end gap-4 flex-wrap"):
                # Product selector.
                product_files = bridge.list_product_files()
                product_options = {f: f.split("/")[-1] for f in product_files}
                product_select = ui.select(
                    product_options,
                    value=product_files[0] if product_files else None,
                    label="Product Specification",
                ).classes("w-64")

                # Execution mode.
                mode_select = ui.radio(
                    _MODE_LABELS,
                    value="Simulation",
                ).props("inline")
                bridge.execution_mode = _MODE_MAP.get(mode_select.value, "ros2")
                bridge.robot_env = "gazebo" if bridge.execution_mode in ("simulate", "ros2") else "real"

            # Prerequisite banner.
            prereq_banner = ui.column().classes("w-full mt-3")

            with ui.row().classes("gap-4 mt-4"):
                async def _start():
                    internal = _MODE_MAP[mode_select.value]
                    bridge.execution_mode = internal
                    bridge.robot_env = "gazebo" if internal in ("simulate", "ros2") else "real"
                    bridge.selected_product = product_select.value or ""
                    await bridge.start_system()

                async def _stop():
                    await bridge.stop_system()

                start_btn = ui.button("Start System", on_click=_start, icon="play_arrow").props("color=green")
                stop_btn = ui.button("Stop System", on_click=_stop, icon="stop").props("color=red")

            # Error display.
            error_label = ui.label("").classes("text-red-500 text-sm mt-2")
            hw_probe = {"busy": False}

            def _update_controls():
                internal = _MODE_MAP.get(mode_select.value, "simulate")
                if internal == "real" and hasattr(bridge, "hardware_connection_statuses"):
                    if not hw_probe["busy"]:
                        async def _probe_hw():
                            hw_probe["busy"] = True
                            try:
                                await asyncio.to_thread(bridge.hardware_connection_statuses)
                            finally:
                                hw_probe["busy"] = False
                        asyncio.create_task(_probe_hw())
                prereqs_met = _check_prerequisites(bridge, internal, prereq_banner)

                can_start = prereqs_met and not bridge.system_running and not bridge._starting
                start_btn.set_enabled(can_start)
                stop_btn.set_enabled(bridge.system_running and not bridge._stopping)
                error_label.text = bridge.last_error or ""

            ui.timer(1.0, _update_controls)

        # ── Agent Overview Grid ──────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Agent Status").classes("text-lg font-semibold mb-2")
            agent_container = ui.row().classes("gap-4 flex-wrap")

        def _refresh_agents():
            agent_container.clear()
            if not bridge.system_running:
                with agent_container:
                    ui.label("System not running").classes("text-slate-400 italic")
                return

            statuses = bridge.get_agent_statuses()
            with agent_container:
                for agent in statuses:
                    with ui.card().classes("w-56"):
                        with ui.row().classes("items-center gap-2"):
                            color = "green" if agent["alive"] else "red"
                            ui.icon("circle", color=color).classes("text-xs")
                            ui.label(agent["name"]).classes("font-semibold")
                        ui.label(agent["type"]).classes("text-xs text-slate-500 uppercase")
                        ui.label(agent["jid"]).classes("text-xs text-slate-400 truncate")

        ui.timer(2.0, _refresh_agents)

        # ── Quick Stats ──────────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Quick Stats").classes("text-lg font-semibold mb-2")
            stats_row = ui.row().classes("gap-8")

        def _refresh_stats():
            stats_row.clear()
            with stats_row:
                task_states = bridge.get_task_states()
                total = len(task_states)
                completed = sum(1 for v in task_states.values() if "completed" in v.lower()) if task_states else 0
                failed = sum(1 for v in task_states.values() if "failed" in v.lower()) if task_states else 0

                _stat_card("Tasks Completed", f"{completed}/{total}", "task_alt")
                _stat_card("Tasks Failed", str(failed), "error_outline")

                robot_states = bridge.get_robot_states()
                active = sum(1 for r in robot_states.values() if r.get("current_state", "idle") != "idle")
                _stat_card("Active Robots", str(active), "precision_manufacturing")

                safety = bridge.get_safety_state()
                blocked = len(safety.get("blocked_tasks", {}))
                _stat_card("Safety Blocks", str(blocked), "shield")

        ui.timer(2.0, _refresh_stats)

        # ── Task DAG ────────────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Task DAG").classes("text-lg font-semibold mb-2")
            mermaid = ui.mermaid("graph TD\n    empty[No plan loaded]").classes("w-full")

            def _refresh_dag():
                nodes = bridge.get_plan_nodes()
                task_states = bridge.get_task_states()
                mermaid.content = nodes_to_mermaid(nodes, task_states)

            ui.timer(2.0, _refresh_dag)

        # ── Task States Table ───────────────────────────────────────
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

        # ── Execution Timeline ──────────────────────────────────────
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
                timeline_table.rows = tl[-50:]

            ui.timer(3.0, _refresh_timeline)

        # ── Node Detail ─────────────────────────────────────────────
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

      # ── Right column: chat panel ──────────────────────────
      with ui.column().classes("w-96 shrink-0 sticky top-20 self-start"):
        render_chat(
            bridge,
            agent_options={
                "auto": "Auto-route",
                "product": "Product Agent",
                "cca": "Central Controller Agent",
                "xarm6": "xArm6",
                "ur5e": "UR5e",
            },
            title="System Chat",
        )


def _check_prerequisites(bridge: SystemBridge, mode: str, banner: ui.column) -> bool:
    """Check if prerequisites are met for the selected mode. Updates the banner. Returns True if OK."""
    banner.clear()

    if bridge.system_running:
        return True  # Already running, don't block.

    if mode == "simulate":
        # Dry Run has no prerequisites.
        with banner:
            with ui.row().classes("items-center gap-2 text-green-600"):
                ui.icon("check_circle").classes("text-sm")
                ui.label("Dry Run mode — no prerequisites required.").classes("text-sm")
        return True

    if mode == "ros2":
        # Simulation needs Gazebo running.
        statuses = bridge.ros2_all_statuses()
        gazebo_running = any(
            statuses.get(k) == "running"
            for k in ("gazebo_dual", "gazebo_xarm6", "gazebo_ur5e")
        )
        with banner:
            if gazebo_running:
                with ui.row().classes("items-center gap-2 text-green-600"):
                    ui.icon("check_circle").classes("text-sm")
                    ui.label("Gazebo is running — ready to start.").classes("text-sm")
            else:
                with ui.row().classes("items-center gap-2 text-amber-600 bg-amber-50 p-3 rounded"):
                    ui.icon("warning").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label("Gazebo is not running.").classes("text-sm font-semibold")
                        with ui.row().classes("items-center gap-1"):
                            ui.label("Go to").classes("text-sm")
                            ui.link("Control", "/control").classes("text-sm font-semibold")
                            ui.label("to launch Gazebo + MoveIt first.").classes("text-sm")
        return gazebo_running

    if mode == "real":
        # Physical mode — show connectivity hints but allow manual override.
        if hasattr(bridge, "hardware_connection_statuses_cached"):
            hw = bridge.hardware_connection_statuses_cached()
        elif hasattr(bridge, "hardware_connection_statuses"):
            hw = bridge.hardware_connection_statuses()
        else:
            hw = {}
        xarm = hw.get("xarm6", {})
        ur5e = hw.get("ur5e", {})

        def _line(name: str, entry: dict) -> str:
            ip = entry.get("ip", "?")
            if entry.get("reachable"):
                latency = entry.get("latency_ms")
                if latency is None:
                    return f"{name}: {ip} reachable"
                return f"{name}: {ip} reachable ({latency:.1f} ms)"
            return f"{name}: {ip} unreachable ({entry.get('message', 'no reply')})"

        with banner:
            with ui.row().classes("items-center gap-2 text-blue-600 bg-blue-50 p-3 rounded"):
                ui.icon("info").classes("text-lg")
                with ui.column().classes("gap-1"):
                    ui.label("Physical mode — ensure robots are powered on and controllers are running.").classes("text-sm")
                    ui.label(_line("xArm6", xarm)).classes("text-xs")
                    ui.label(_line("UR5e", ur5e)).classes("text-xs")
        return True

    return True


def _stat_card(label: str, value: str, icon: str) -> None:
    with ui.column().classes("items-center"):
        ui.icon(icon).classes("text-2xl text-slate-500")
        ui.label(value).classes("text-xl font-bold")
        ui.label(label).classes("text-xs text-slate-500")
