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
    "Dry Run": "dry_run",
    "Simulation": "simulation",
    "Physical": "physical",
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
                bridge.execution_mode = _MODE_MAP.get(mode_select.value, "simulation")
                bridge.robot_env = "gazebo" if bridge.execution_mode in ("dry_run", "simulation") else "real"

            # Prerequisite banner.
            prereq_banner = ui.column().classes("w-full mt-3")

            with ui.column().classes("w-full gap-2 mt-4"):
                banner_hide_task: asyncio.Task | None = None
                start_click_state = {"locked": False}
                start_task: asyncio.Task | None = None

                def _set_action_banner(kind: str, message: str, *, auto_hide_s: float | None = None) -> None:
                    nonlocal banner_hide_task
                    style_map = {
                        "info": ("info", "text-blue-700 bg-blue-50"),
                        "success": ("check_circle", "text-green-700 bg-green-50"),
                        "warning": ("warning", "text-amber-700 bg-amber-50"),
                        "error": ("error", "text-red-700 bg-red-50"),
                    }
                    icon_name, color_classes = style_map.get(kind, style_map["info"])

                    action_banner_icon.name = icon_name
                    action_banner.classes(remove="text-blue-700 bg-blue-50 text-green-700 bg-green-50 text-amber-700 bg-amber-50 text-red-700 bg-red-50")
                    action_banner.classes(add=color_classes)
                    action_banner_label.text = message
                    action_banner.style("display:flex;")

                    if banner_hide_task and not banner_hide_task.done():
                        banner_hide_task.cancel()
                        banner_hide_task = None

                    if auto_hide_s is not None and auto_hide_s > 0:
                        async def _hide_later():
                            try:
                                await asyncio.sleep(auto_hide_s)
                            except asyncio.CancelledError:
                                return
                            action_banner.style("display:none;")

                        banner_hide_task = asyncio.create_task(_hide_later())

                async def _start():
                    nonlocal start_task
                    if hasattr(bridge, "_diag_emit"):
                        bridge._diag_emit(
                            f"dashboard start click running={bridge.system_running} "
                            f"starting={bridge._starting} locked={start_click_state['locked']}"
                        )
                    if bridge.system_running:
                        _set_action_banner("warning", "System is already running.", auto_hide_s=4.0)
                        return
                    if start_click_state["locked"]:
                        _set_action_banner("warning", "Start already in progress...")
                        return
                    if bridge._starting:
                        _set_action_banner("warning", "Start already in progress...")
                        return
                    internal = _MODE_MAP[mode_select.value]
                    bridge.execution_mode = internal
                    bridge.robot_env = "gazebo" if internal in ("dry_run", "simulation") else "real"
                    bridge.selected_product = product_select.value or ""
                    if not _check_prerequisites(bridge, internal, prereq_banner):
                        detail = f" {bridge.last_error}" if bridge.last_error else ""
                        _set_action_banner(
                            "warning",
                            "Startup is not done yet." + detail,
                            auto_hide_s=6.0,
                        )
                        return
                    start_click_state["locked"] = True
                    _update_controls()
                    _set_action_banner("info", "Start System clicked. Starting agents...")
                    if start_task is None or start_task.done():
                        async def _run_start_in_background() -> None:
                            if hasattr(bridge, "_diag_emit"):
                                bridge._diag_emit("dashboard start background task begin")
                            try:
                                await bridge.start_system()
                                if bridge.system_running:
                                    _set_action_banner("success", "System started successfully.", auto_hide_s=5.0)
                                else:
                                    reason = bridge.last_error or "unknown error"
                                    _set_action_banner("error", f"Start failed: {reason}", auto_hide_s=8.0)
                            finally:
                                if hasattr(bridge, "_diag_emit"):
                                    bridge._diag_emit(
                                        "dashboard start background task end "
                                        f"running={bridge.system_running} error={bridge.last_error or ''}"
                                    )
                                if not bridge.system_running:
                                    start_click_state["locked"] = False
                                _update_controls()

                        start_task = asyncio.create_task(_run_start_in_background())

                async def _stop():
                    if bridge._stopping:
                        _set_action_banner("warning", "Stop already in progress...")
                        return
                    if not bridge.system_running:
                        _set_action_banner("warning", "System is already stopped.", auto_hide_s=4.0)
                        start_click_state["locked"] = False
                        _update_controls()
                        return
                    _set_action_banner("info", "Stop System clicked. Stopping agents...")
                    try:
                        await bridge.stop_system()
                        if not bridge.system_running:
                            _set_action_banner("success", "System stopped.", auto_hide_s=5.0)
                            start_click_state["locked"] = False
                        else:
                            _set_action_banner("error", "Stop failed. System is still running.", auto_hide_s=8.0)
                    finally:
                        _update_controls()

                async def _reset_gazebo():
                    if bridge._starting:
                        _set_action_banner("warning", "Start is in progress. Wait before reset.", auto_hide_s=4.0)
                        return
                    if bridge._stopping:
                        _set_action_banner("warning", "Stop is in progress. Wait before reset.", auto_hide_s=4.0)
                        return

                    _set_action_banner("info", "Reset Gazebo clicked. Preparing reset...")
                    try:
                        # Keep agent/world state consistent: stop system first if needed.
                        if bridge.system_running:
                            _set_action_banner("info", "Stopping system before Gazebo reset...")
                            await bridge.stop_system()
                            if bridge.system_running:
                                _set_action_banner(
                                    "error",
                                    "Reset aborted: failed to stop system first.",
                                    auto_hide_s=8.0,
                                )
                                return
                            start_click_state["locked"] = False

                        ok, msg = await asyncio.to_thread(bridge.ros2_reset_gazebo_environment)
                        if ok:
                            start_click_state["locked"] = False
                            _set_action_banner(
                                "success",
                                "Gazebo scene reset to initial state. Click Start System.",
                                auto_hide_s=6.0,
                            )
                        else:
                            _set_action_banner("warning", f"Gazebo reset skipped: {msg}", auto_hide_s=8.0)
                    finally:
                        _update_controls()

                with ui.row().classes("gap-4"):
                    start_btn = ui.button("Start System", on_click=_start, icon="play_arrow").props("color=green")
                    stop_btn = ui.button("Stop System", on_click=_stop, icon="stop").props("color=red")
                    reset_btn = ui.button("Reset Gazebo", on_click=_reset_gazebo, icon="restart_alt").props("color=blue")
                action_banner = ui.row().classes(
                    "w-full mt-2 items-center gap-2 rounded p-3 text-sm text-blue-700 bg-blue-50"
                )
                action_banner.style("display:none;")
                with action_banner:
                    action_banner_icon = ui.icon("info")
                    action_banner_label = ui.label("")

            # Error display.
            error_label = ui.label("").classes("text-red-500 text-sm mt-2")
            hw_probe = {"busy": False}

            def _update_controls():
                try:
                    internal = _MODE_MAP.get(mode_select.value, "dry_run")
                    if internal == "physical" and hasattr(bridge, "hardware_connection_statuses"):
                        if not hw_probe["busy"]:
                            async def _probe_hw():
                                hw_probe["busy"] = True
                                try:
                                    await asyncio.to_thread(bridge.hardware_connection_statuses)
                                finally:
                                    hw_probe["busy"] = False
                            asyncio.create_task(_probe_hw())

                    if bridge._starting and not bridge.system_running:
                        prereqs_met = False
                    else:
                        prereqs_met = _check_prerequisites(bridge, internal, prereq_banner)

                    can_start = (
                        prereqs_met
                        and not bridge.system_running
                        and not bridge._starting
                        and not start_click_state["locked"]
                    )
                    gazebo_running = any(
                        bridge.ros2_proc_status(name) == "running"
                        for name in ("gazebo_dual", "gazebo_xarm6", "gazebo_ur5e")
                    )
                    start_btn.set_enabled(can_start)
                    stop_btn.set_enabled(bridge.system_running and not bridge._stopping)
                    reset_btn.set_enabled(
                        (bridge.system_running or gazebo_running)
                        and not bridge._starting
                        and not bridge._stopping
                    )
                    error_label.text = bridge.last_error or ""
                except Exception as exc:
                    if hasattr(bridge, "_diag_emit"):
                        bridge._diag_emit(f"dashboard update_controls exception: {exc}")
                    start_btn.set_enabled(False)
                    stop_btn.set_enabled(False)
                    reset_btn.set_enabled(False)
                    error_label.text = f"Dashboard control update failed: {exc}"

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
                row_key="task_id",
                selection="multiple",
            ).classes("w-full")

            selected_task_ids: list[str] = []

            def _set_node_details(task_ids: list[str]) -> None:
                """Populate the detail table with one or more selected nodes."""
                if not task_ids:
                    detail_label.text = "Node Detail"
                    detail_table.rows = []
                    return
                nodes = bridge.get_plan_nodes()
                all_rows: list[dict] = []
                row_idx = 0
                for tid in task_ids:
                    node = next((n for n in nodes if (n.get("id") or n.get("task_id")) == tid), None)
                    if not node:
                        continue
                    if all_rows:
                        all_rows.append({"row_id": f"sep_{row_idx}", "key": "———", "value": "———"})
                        row_idx += 1
                    for k, v in node.items():
                        display = json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
                        all_rows.append({"row_id": f"{tid}_{k}", "key": k, "value": display})
                        row_idx += 1
                detail_label.text = "Node Detail — " + ", ".join(task_ids)
                detail_table.rows = all_rows

            def _refresh_tasks():
                ts = bridge.get_task_states()
                task_table.rows = [{"task_id": k, "status": v} for k, v in ts.items()]
                if selected_task_ids:
                    _set_node_details(selected_task_ids)
                elif ts:
                    # Auto-show the currently running task, or the last active one.
                    active = next((k for k, v in ts.items() if "running" in v.lower()), None)
                    if not active:
                        active = next(
                            (k for k, v in reversed(ts.items())
                             if any(s in v.lower() for s in ("dispatched", "accepted", "completed", "failed"))),
                            None,
                        )
                    if active:
                        _set_node_details([active])

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
            detail_label = ui.label("Node Detail").classes("text-lg font-semibold mb-2")
            detail_table = ui.table(
                columns=[
                    {"name": "key", "label": "Field", "field": "key"},
                    {"name": "value", "label": "Value", "field": "value"},
                ],
                rows=[],
                row_key="row_id",
            ).classes("w-full")

            def _on_task_select(e):
                selected_task_ids.clear()
                for row in (e.selection or []):
                    tid = row.get("task_id")
                    if tid:
                        selected_task_ids.append(tid)
                _set_node_details(selected_task_ids)

            task_table.on_select(_on_task_select)

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

    if bridge._starting:
        with banner:
            with ui.row().classes("items-center gap-2 text-blue-600 bg-blue-50 p-3 rounded"):
                ui.icon("hourglass_top").classes("text-lg")
                ui.label("System startup in progress...").classes("text-sm font-semibold")
        return False

    if mode == "dry_run":
        # Dry Run has no prerequisites.
        with banner:
            with ui.row().classes("items-center gap-2 text-green-600"):
                ui.icon("check_circle").classes("text-sm")
                ui.label("Dry Run mode — no prerequisites required.").classes("text-sm")
        return True

    if mode == "simulation":
        # Simulation needs Gazebo stack plus MoveIt/services ready.
        statuses = bridge.ros2_all_statuses()
        gazebo_running = any(
            statuses.get(k) == "running"
            for k in ("gazebo_dual", "gazebo_xarm6", "gazebo_ur5e")
        )
        sim_ready = False
        sim_reason = ""
        if gazebo_running and hasattr(bridge, "simulation_start_ready"):
            sim_ready, sim_reason = bridge.simulation_start_ready()
        elif gazebo_running:
            sim_ready = True

        with banner:
            if gazebo_running and sim_ready:
                with ui.row().classes("items-center gap-2 text-green-600"):
                    ui.icon("check_circle").classes("text-sm")
                    ui.label("Gazebo + MoveIt are ready — safe to start.").classes("text-sm")
            elif gazebo_running:
                with ui.row().classes("items-center gap-2 text-amber-600 bg-amber-50 p-3 rounded"):
                    ui.icon("warning").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label("Startup is not done yet.").classes("text-sm font-semibold")
                        ui.label(sim_reason or "Waiting for ROS services/prewarm to complete...").classes("text-xs")
            else:
                with ui.row().classes("items-center gap-2 text-amber-600 bg-amber-50 p-3 rounded"):
                    ui.icon("warning").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label("Gazebo is not running.").classes("text-sm font-semibold")
                        with ui.row().classes("items-center gap-1"):
                            ui.label("Go to").classes("text-sm")
                            ui.link("Control", "/control").classes("text-sm font-semibold")
                            ui.label("to launch Gazebo + MoveIt first.").classes("text-sm")
        return gazebo_running and sim_ready

    if mode == "physical":
        # Physical mode requires real perception backend availability.
        if hasattr(bridge, "hardware_connection_statuses_cached"):
            hw = bridge.hardware_connection_statuses_cached()
        elif hasattr(bridge, "hardware_connection_statuses"):
            hw = bridge.hardware_connection_statuses()
        else:
            hw = {}
        xarm = hw.get("xarm6", {})
        ur5e = hw.get("ur5e", {})
        ready, reason = bridge.physical_perception_ready()

        def _line(name: str, entry: dict) -> str:
            ip = entry.get("ip", "?")
            if entry.get("reachable"):
                latency = entry.get("latency_ms")
                if latency is None:
                    return f"{name}: {ip} reachable"
                return f"{name}: {ip} reachable ({latency:.1f} ms)"
            return f"{name}: {ip} unreachable ({entry.get('message', 'no reply')})"

        with banner:
            if not ready:
                with ui.row().classes("items-center gap-2 text-red-700 bg-red-50 p-3 rounded"):
                    ui.icon("error").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label("Physical mode is blocked.").classes("text-sm font-semibold")
                        ui.label(reason).classes("text-xs")
                        ui.label(_line("xArm6", xarm)).classes("text-xs")
                        ui.label(_line("UR5e", ur5e)).classes("text-xs")
            else:
                with ui.row().classes("items-center gap-2 text-blue-600 bg-blue-50 p-3 rounded"):
                    ui.icon("info").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label("Physical mode — ensure robots are powered on and controllers are running.").classes("text-sm")
                        ui.label(_line("xArm6", xarm)).classes("text-xs")
                        ui.label(_line("UR5e", ur5e)).classes("text-xs")
        return ready

    return True


def _stat_card(label: str, value: str, icon: str) -> None:
    with ui.column().classes("items-center"):
        ui.icon(icon).classes("text-2xl text-slate-500")
        ui.label(value).classes("text-xl font-bold")
        ui.label(label).classes("text-xs text-slate-500")
