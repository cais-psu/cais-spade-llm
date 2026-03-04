"""Resources page: robot configurations, specifications, and live state."""

from __future__ import annotations

import json

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.agent_chat import render_chat
from cais_spade_llm.ui.components.status_badge import status_badge


_PHASES = ["idle", "at_pick", "picked", "positioned", "placed"]


def render(bridge: SystemBridge) -> None:
    ui.label("Resources").classes("text-2xl font-bold px-6 pt-6")

    with ui.row().classes("w-full px-6 gap-6 items-start"):
      with ui.column().classes("flex-grow gap-6 min-w-0"):

        # ── Resource Configuration Files ─────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Resources").classes("text-lg font-semibold mb-2")
            ui.label("JSON manifests that define robot capabilities, controllers, and motion parameters.").classes(
                "text-xs text-slate-500 mb-3"
            )

            res_files = bridge.list_resource_files()
            if not res_files:
                ui.label("No resource files found").classes("text-slate-400 italic")
            else:
                for path in res_files:
                    filename = path.split("/")[-1]
                    with ui.expansion(filename, icon="description").classes("w-full"):
                        try:
                            config_data = bridge.load_config(path)
                        except Exception:
                            config_data = {}

                        editor = ui.textarea(
                            value=json.dumps(config_data, indent=2),
                            label=filename,
                        ).classes("w-full font-mono text-xs").props("rows=20")

                        def _save(p=path, e=editor):
                            try:
                                data = json.loads(e.value)
                                bridge.save_config(p, data)
                                ui.notify("Configuration saved", type="positive")
                            except json.JSONDecodeError as exc:
                                ui.notify(f"Invalid JSON: {exc}", type="negative")

                        ui.button("Save", on_click=_save, icon="save").classes("mt-2")

        # ── Live Robot Status ────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Live Robot Status").classes("text-lg font-semibold mb-2")
            robot_container = ui.column().classes("w-full gap-4")

        def _refresh():
            robot_container.clear()
            states = bridge.get_robot_states()

            if not states:
                with robot_container:
                    ui.label("No robots available — start the system first").classes("text-slate-400 italic")
                return

            with robot_container:
                for name, state in states.items():
                    _robot_status_card(name, state)

        ui.timer(2.0, _refresh)

      # ── Right column: chat panel ──────────────────────────
      with ui.column().classes("w-96 shrink-0 sticky top-20 self-start"):
        render_chat(
            bridge,
            agent_options={"xarm6": "xArm6", "ur5e": "UR5e"},
            title="Resource Agent Chat",
        )


def _robot_status_card(name: str, state: dict) -> None:
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center gap-4"):
            ui.label(name).classes("text-lg font-bold")
            current = state.get("current_state", "idle")
            status_badge(current)

        # Phase pipeline.
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
                    ui.label(f"{axis}: {val:.3f}" if isinstance(val, float) else f"{axis}: {val}").classes(
                        "text-xs font-mono text-slate-500"
                    )


def _detail(label: str, value: str) -> None:
    with ui.column().classes("gap-0"):
        ui.label(label).classes("text-xs text-slate-500")
        ui.label(value).classes("text-sm font-semibold")
