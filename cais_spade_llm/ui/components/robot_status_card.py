"""Reusable live robot status card."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from cais_spade_llm.ui.components.status_badge import status_badge

_PHASES = ["idle", "at_pick", "picked", "positioned", "placed"]


def render_robot_status_card(name: str, state: dict[str, Any]) -> None:
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center gap-4"):
            ui.label(name).classes("text-lg font-bold")
            current = state.get("current_state", "idle")
            status_badge(current)

        with ui.row().classes("gap-1 mt-2"):
            current_phase = state.get("current_state", "idle")
            for phase in _PHASES:
                is_current = phase == current_phase
                color = "bg-blue-500 text-white" if is_current else "bg-slate-200 text-slate-600"
                ui.label(phase).classes(f"px-3 py-1 rounded text-xs font-mono {color}")

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
                    text = f"{axis}: {val:.3f}" if isinstance(val, float) else f"{axis}: {val}"
                    ui.label(text).classes("text-xs font-mono text-slate-500")


def _detail(label: str, value: str) -> None:
    with ui.column().classes("gap-0"):
        ui.label(label).classes("text-xs text-slate-500")
        ui.label(value).classes("text-sm font-semibold")
