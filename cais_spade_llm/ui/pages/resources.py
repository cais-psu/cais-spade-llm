"""Resources page: robot configurations, specifications, and live state."""

from __future__ import annotations

import json

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.agent_chat import render_chat
from cais_spade_llm.ui.components.robot_status_card import render_robot_status_card


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
                    render_robot_status_card(name, state)

        ui.timer(2.0, _refresh)

      # ── Right column: chat panel ──────────────────────────
      with ui.column().classes("w-96 shrink-0 sticky top-20 self-start"):
        render_chat(
            bridge,
            agent_options={"xarm6": "xArm6", "ur5e": "UR5e"},
            title="Resource Agent Chat",
        )
