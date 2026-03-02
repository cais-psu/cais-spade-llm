"""Dashboard page: system control panel + agent overview grid."""

from __future__ import annotations

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge


def render(bridge: SystemBridge) -> None:
    with ui.column().classes("w-full max-w-6xl mx-auto p-6 gap-6"):
        ui.label("Dashboard").classes("text-2xl font-bold")

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
                    ["simulate", "ros2", "real"],
                    value=bridge.execution_mode,
                ).props("inline")

                # Robot environment.
                env_select = ui.radio(
                    ["gazebo", "real"],
                    value=bridge.robot_env,
                ).props("inline")

            with ui.row().classes("gap-4 mt-4"):
                async def _start():
                    bridge.execution_mode = mode_select.value
                    bridge.robot_env = env_select.value
                    bridge.selected_product = product_select.value or ""
                    await bridge.start_system()

                async def _stop():
                    await bridge.stop_system()

                start_btn = ui.button("Start System", on_click=_start, icon="play_arrow").props("color=green")
                stop_btn = ui.button("Stop System", on_click=_stop, icon="stop").props("color=red")

            # Error display.
            error_label = ui.label("").classes("text-red-500 text-sm mt-2")

            def _update_controls():
                start_btn.set_enabled(not bridge.system_running and not bridge._starting)
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


def _stat_card(label: str, value: str, icon: str) -> None:
    with ui.column().classes("items-center"):
        ui.icon(icon).classes("text-2xl text-slate-500")
        ui.label(value).classes("text-xl font-bold")
        ui.label(label).classes("text-xs text-slate-500")
