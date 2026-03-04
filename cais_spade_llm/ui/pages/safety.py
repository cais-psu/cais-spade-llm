"""Safety page: safety requirements CRUD, rules table, runtime state, blocked tasks."""

from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui, events

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.agent_chat import render_chat


_SAFETY_DIR = Path("cais_spade_llm/specification/safety")


def render(bridge: SystemBridge) -> None:
    ui.label("Safety").classes("text-2xl font-bold px-6 pt-6")

    with ui.row().classes("w-full px-6 gap-6 items-start flex-nowrap"):
      with ui.column().classes("flex-grow gap-6 min-w-0"):

        # ── Safety Requirements (top; same pattern as Product Agent page) ──
        _render_safety_requirements_card(bridge)

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

        # ── Runtime Safety State ─────────────────────────────────────
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

      # ── Right column: chat panel ──────────────────────────
      with ui.column().classes("w-96 shrink-0 sticky top-20 self-start"):
        render_chat(bridge, agent_jid="cca", title="Central Controller Agent Chat")


def _render_safety_requirements_card(bridge: SystemBridge) -> None:
    with ui.card().classes("w-full"):
        ui.label("Safety Requirements").classes("text-lg font-semibold mb-2")
        ui.label("Text files that define safety constraints for the system.").classes(
            "text-xs text-slate-500 mb-3"
        )

        _SAFETY_DIR.mkdir(parents=True, exist_ok=True)
        status_label = ui.label("").classes("text-sm")

        req_select = ui.select(
            {},
            label="Requirement File",
        ).classes("w-64")

        req_editor = ui.textarea(label="Edit requirements").classes(
            "w-full font-mono"
        ).props("outlined autogrow")

        def _refresh_file_list(select_path: str | None = None):
            req_files = sorted(_SAFETY_DIR.glob("*.txt"))
            file_map = {str(f): f.stem for f in req_files}
            req_select.options = file_map
            req_select.update()
            if select_path and select_path in file_map:
                req_select.value = select_path
            elif req_files:
                req_select.value = str(req_files[0])
            else:
                req_select.value = None
                req_editor.value = ""
            _load_req()

        def _load_req():
            if not req_select.value:
                req_editor.value = ""
                status_label.text = ""
                return
            path = Path(req_select.value)
            if path.exists():
                req_editor.value = path.read_text()
                status_label.text = ""

        def _save_req():
            if bridge.system_running:
                ui.notify("Cannot edit while system is running", type="warning")
                return
            if not req_select.value:
                status_label.text = "No file selected"
                status_label.classes(replace="text-sm text-amber-600")
                return
            path = Path(req_select.value)
            path.write_text(req_editor.value)
            status_label.text = f"Saved {path.name}"
            status_label.classes(replace="text-sm text-green-600")

        def _delete_req():
            if bridge.system_running:
                ui.notify("Cannot edit while system is running", type="warning")
                return
            if not req_select.value:
                return
            path = Path(req_select.value)
            name = path.name
            if path.exists():
                path.unlink()
            status_label.text = f"Deleted {name}"
            status_label.classes(replace="text-sm text-red-600")
            _refresh_file_list()

        async def _handle_upload(e: events.UploadEventArguments):
            if bridge.system_running:
                ui.notify("Cannot edit while system is running", type="warning")
                return
            content = e.content.read()
            name = e.name if e.name.endswith(".txt") else e.name + ".txt"
            dest = _SAFETY_DIR / name
            dest.write_bytes(content)
            status_label.text = f"Uploaded {name}"
            status_label.classes(replace="text-sm text-green-600")
            _refresh_file_list(str(dest))

        async def _create_new():
            if bridge.system_running:
                ui.notify("Cannot edit while system is running", type="warning")
                return
            name_input.value = name_input.value.strip()
            if not name_input.value:
                status_label.text = "Enter a file name first"
                status_label.classes(replace="text-sm text-amber-600")
                return
            name = name_input.value if name_input.value.endswith(".txt") else name_input.value + ".txt"
            dest = _SAFETY_DIR / name
            if dest.exists():
                status_label.text = f"{name} already exists — select it to edit"
                status_label.classes(replace="text-sm text-amber-600")
                return
            dest.write_text("[Safety Requirements]\n- ")
            status_label.text = f"Created {name}"
            status_label.classes(replace="text-sm text-green-600")
            name_input.value = ""
            _refresh_file_list(str(dest))

        req_select.on_value_change(_load_req)

        with ui.row().classes("gap-2 mt-1 items-end flex-wrap"):
            ui.button("Save", on_click=_save_req, icon="save").props("color=primary")
            ui.button("Reload", on_click=_load_req, icon="refresh").props("flat")
            ui.button("Delete", on_click=_delete_req, icon="delete").props("flat color=red")

        with ui.row().classes("gap-2 mt-3 items-end flex-wrap"):
            name_input = ui.input(label="New file name", placeholder="e.g. safety_case3").classes("w-48")
            ui.button("Create", on_click=_create_new, icon="add").props("flat")
            ui.upload(
                label="Upload .txt",
                auto_upload=True,
                on_upload=_handle_upload,
                max_file_size=1_000_000,
            ).props("accept=.txt flat dense").classes("w-40")

        _refresh_file_list()

        def _update_readonly():
            req_editor.props(f"readonly={str(bridge.system_running).lower()}")

        ui.timer(2.0, _update_readonly)
