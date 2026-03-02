"""Safety monitor page: rules table, DFA states, blocked tasks, decisions feed."""

from __future__ import annotations

import json

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge


def render(bridge: SystemBridge) -> None:
    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-6"):
        ui.label("Safety Monitor").classes("text-2xl font-bold")

        # ── Safety Rules ─────────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Safety Rules").classes("text-lg font-semibold mb-2")
            rules_table = ui.table(
                columns=[
                    {"name": "id", "label": "Rule ID", "field": "id", "sortable": True},
                    {"name": "raw_text", "label": "Rule Text", "field": "raw_text"},
                    {"name": "constraint_type", "label": "Type", "field": "constraint_type"},
                    {"name": "ltlf", "label": "LTLf Formula", "field": "ltlf"},
                ],
                rows=[],
            ).classes("w-full")

            def _refresh_rules():
                rules = bridge.get_safety_rules()
                rows = []
                for i, r in enumerate(rules):
                    rows.append({
                        "id": r.get("id", f"R{i}"),
                        "raw_text": r.get("raw_text", r.get("text", str(r))),
                        "constraint_type": r.get("constraint_type", ""),
                        "ltlf": r.get("ltlf", r.get("formula", "")),
                    })
                rules_table.rows = rows

            ui.timer(5.0, _refresh_rules)

        # ── DFA / FSA State ──────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Runtime Safety State").classes("text-lg font-semibold mb-2")
            state_pre = ui.code("{}", language="json").classes("w-full")

            def _refresh_state():
                ss = bridge.get_safety_state()
                state_pre.content = json.dumps(ss, indent=2, default=str) if ss else "{}"

            ui.timer(2.0, _refresh_state)

        # ── Blocked Tasks ────────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Blocked Tasks").classes("text-lg font-semibold mb-2")
            blocked_container = ui.column().classes("w-full gap-2")

            def _refresh_blocked():
                blocked_container.clear()
                ss = bridge.get_safety_state()
                blocked = ss.get("blocked_tasks", {})
                if not blocked:
                    with blocked_container:
                        ui.label("No blocked tasks").classes("text-slate-400 italic")
                    return
                with blocked_container:
                    for tid, info in blocked.items():
                        with ui.card().classes("w-full bg-red-50"):
                            ui.label(f"Task: {tid}").classes("font-semibold")
                            ui.label(f"Violated rule: {info.get('violated_rule', 'unknown')}").classes("text-sm text-red-600")

            ui.timer(3.0, _refresh_blocked)

        # ── Safety Requirements Editor ───────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Safety Requirements (read-only when running)").classes("text-lg font-semibold mb-2")

            safety_path = "cais_spade_llm/specification/safety/safety_requirements.txt"
            try:
                with open(safety_path) as f:
                    safety_text = f.read()
            except FileNotFoundError:
                safety_text = ""

            editor = ui.textarea(value=safety_text, label="Safety Requirements").classes("w-full").props("rows=10")

            def _save_safety():
                if bridge.system_running:
                    ui.notify("Cannot edit while system is running", type="warning")
                    return
                with open(safety_path, "w") as f:
                    f.write(editor.value)
                ui.notify("Safety requirements saved", type="positive")

            with ui.row():
                ui.button("Save", on_click=_save_safety, icon="save")

            def _update_readonly():
                editor.props(f"readonly={str(bridge.system_running).lower()}")

            ui.timer(2.0, _update_readonly)
