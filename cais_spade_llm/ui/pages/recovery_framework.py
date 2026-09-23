"""recovery-framework setup, existing workflow, and saved evidence."""

from __future__ import annotations

import json
from pathlib import Path

from nicegui import context, ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.gazebo_delivery_run import render_gazebo_delivery_runs
from cais_spade_llm.ui.components.recovery_setup import render_setup
from cais_spade_llm.ui.pages import recovery_run
from cais_spade_llm.ui.recovery_results import NOT_RECORDED, read_artifact, render_results

_ROOT = Path(__file__).resolve().parents[3]
_PAPER = _ROOT / "writing/Journal Paper 2 (recovery framework)/README.md"


def _render_setup(bridge: SystemBridge) -> None:
    render_setup(bridge, root=_ROOT)
    path = "cais_spade_llm/initialization/recovery_framework_gazebo.json"
    with ui.expansion("Configured plant layout", icon="description").classes("w-full"):
        ui.label(path).classes("text-xs break-all")
        ui.code(json.dumps(read_artifact(_ROOT, path), indent=2), language="json").classes("w-full")
    ui.label("Planned paper experiments").classes("text-lg font-semibold")
    ui.label("The protocol below describes planned comparisons, not completed trials.").classes(
        "text-sm text-slate-600"
    )
    try:
        protocol = _PAPER.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        ui.label(f"Paper protocol is unavailable: {exc}").classes("text-amber-700")
        return
    for heading in ("Working Thesis", "Experiment Plan", "Neurosymbolic Selection Method"):
        _, found, section = protocol.partition(f"## {heading}\n")
        with ui.expansion(heading, icon="science", value=heading == "Experiment Plan").classes(
            "w-full"
        ):
            ui.markdown(section.split("\n## ", 1)[0] if found else NOT_RECORDED)
    ui.button("Download paper protocol", icon="download", on_click=lambda: ui.download.file(_PAPER))


def render(bridge: SystemBridge) -> None:
    """Open run by default and construct its controls once per page client."""
    try:
        request = context.client.request
    except RuntimeError:
        request = None
    initial_tab = request.query_params.get("tab", "run") if request is not None else "run"
    if initial_tab not in {"run", "setup", "results"}:
        initial_tab = "run"
    ui.add_head_html("""<style>
        .recovery-panels, .recovery-panels > .q-panel { overflow: visible !important; }
    </style>""")
    with ui.column().classes("w-full gap-4 p-6"):
        ui.label("recovery-framework").classes("text-2xl font-bold")
        with ui.tabs().props("no-caps").classes("w-full") as tabs:
            for name in ("run", "setup", "results"):
                ui.tab(name).props("no-caps")
        with ui.tab_panels(tabs, value=initial_tab, animated=False, keep_alive=True).classes(
            "w-full recovery-panels"
        ):
            panels = {}
            for name in ("run", "setup", "results"):
                with ui.tab_panel(name).classes("p-0") as panel:
                    panels[name] = panel
        built: set[str] = set()

        def show_tab() -> None:
            name = tabs.value
            if name not in panels or name in built:
                return
            with panels[name]:
                if name == "setup":
                    _render_setup(bridge)
                elif name == "run":
                    recovery_run.render(bridge, is_active=lambda: tabs.value == "run")
                else:
                    ui.label(
                        "Saved results retain their recorded inputs; the current setup is not applied to historical runs."
                    ).classes("text-sm text-slate-600")
                    render_gazebo_delivery_runs()
                    render_gazebo_delivery_runs(
                        _ROOT / 'cais_spade_llm/monitor/environment_runs', environment=True,
                    )
                    render_results()
            built.add(name)

        tabs.on_value_change(show_tab)
        show_tab()
