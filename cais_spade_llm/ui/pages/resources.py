"""Resources page: nominal capabilities and live robot status."""

from __future__ import annotations

import asyncio
from copy import deepcopy

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.refresh import PageRefresh
from cais_spade_llm.ui.components.nominal_resource_des import render_nominal_resource_des
from cais_spade_llm.ui.components.robot_status_card import render_robot_status_card


def render(bridge: SystemBridge) -> None:
    """Render resource capability models and a separate live status panel.

    Args:
        bridge: Existing public UI-to-runtime surface.
    """
    polling = PageRefresh()
    ui.label("Resources").classes("text-2xl font-bold px-6 pt-6")
    with ui.row().classes("px-6 items-center gap-4"):
        ui.label(
            "Inspect configured capabilities here; choose permitted resources and failure scenarios in project setup."
        ).classes("text-sm text-slate-600")
        ui.link("recovery-framework setup", "/recovery-framework?tab=setup")

    with ui.column().classes("w-full px-6 gap-6 min-w-0"):
        refresh_capabilities = render_nominal_resource_des(bridge)
        with ui.card().classes("w-full"):
            ui.label("Live Robot Status").classes("text-lg font-semibold mb-2")
            robot_container = ui.column().classes("w-full gap-4")

        previous = None

        async def _refresh() -> None:
            nonlocal previous
            if refresh_capabilities is not None:
                await refresh_capabilities()
            states = await asyncio.to_thread(bridge.get_robot_states)
            if not polling.active() or states == previous:
                return
            previous = deepcopy(states)
            robot_container.clear()
            with robot_container:
                if not states:
                    ui.label("No robots available — start the system first").classes(
                        "text-slate-400 italic"
                    )
                else:
                    for name, state in states.items():
                        render_robot_status_card(name, state)

        polling.timer(2.0, _refresh)
