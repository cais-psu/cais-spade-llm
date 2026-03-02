"""Robot status page: phase pipeline, state cards, config editor."""

from __future__ import annotations

import json

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.status_badge import status_badge


_PHASES = ["idle", "at_pick", "picked", "positioned", "placed"]


def render(bridge: SystemBridge) -> None:
    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-6"):
        ui.label("Robot Status").classes("text-2xl font-bold")

        robot_container = ui.column().classes("w-full gap-6")

        def _refresh():
            robot_container.clear()
            states = bridge.get_robot_states()

            if not states:
                with robot_container:
                    ui.label("No robots available").classes("text-slate-400 italic")
                return

            with robot_container:
                for name, state in states.items():
                    _robot_card(bridge, name, state)

        ui.timer(2.0, _refresh)


def _robot_card(bridge: SystemBridge, name: str, state: dict) -> None:
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center gap-4"):
            ui.label(name).classes("text-lg font-bold")
            current = state.get("current_state", "idle")
            status_badge(current)

        # Phase pipeline visualization.
        with ui.row().classes("gap-1 mt-2"):
            current_phase = state.get("current_state", "idle")
            for phase in _PHASES:
                is_current = phase == current_phase
                color = "bg-blue-500 text-white" if is_current else "bg-slate-200 text-slate-600"
                ui.label(phase).classes(f"px-3 py-1 rounded text-xs font-mono {color}")

        # State details.
        with ui.row().classes("gap-8 mt-3 flex-wrap"):
            _detail("Execution Mode", state.get("execution_mode", "?"))
            _detail("Controller Ready", str(state.get("controller_ready", "?")))
            _detail("Held Part", state.get("held_part") or "None")
            _detail("Gripper", state.get("gripper_state", "?"))

        pos = state.get("position", {})
        if pos:
            with ui.row().classes("gap-4 mt-2"):
                for axis in ("x", "y", "z"):
                    val = pos.get(axis, 0)
                    ui.label(f"{axis}: {val:.3f}" if isinstance(val, float) else f"{axis}: {val}").classes("text-xs font-mono text-slate-500")

        # Config editor (expandable).
        with ui.expansion("Configuration", icon="settings").classes("w-full mt-2"):
            res_files = bridge.list_resource_files()
            matching = [f for f in res_files if name in f]
            if matching:
                config_path = matching[0]
                try:
                    config_data = bridge.load_config(config_path)
                except Exception:
                    config_data = {}

                editor = ui.textarea(
                    value=json.dumps(config_data, indent=2),
                    label=f"Config: {config_path.split('/')[-1]}",
                ).classes("w-full font-mono text-xs").props("rows=15")

                def _save(p=config_path, e=editor):
                    try:
                        data = json.loads(e.value)
                        bridge.save_config(p, data)
                        ui.notify("Configuration saved", type="positive")
                    except json.JSONDecodeError as exc:
                        ui.notify(f"Invalid JSON: {exc}", type="negative")

                ui.button("Save Config", on_click=_save, icon="save").classes("mt-2")
            else:
                ui.label("No config file found").classes("text-slate-400")


def _detail(label: str, value: str) -> None:
    with ui.column().classes("gap-0"):
        ui.label(label).classes("text-xs text-slate-500")
        ui.label(value).classes("text-sm font-semibold")
