"""Products page: editable requirements (CRUD + upload), spec viewer, part tracker, geometry."""

from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui, events

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.agent_chat import render_chat


_SPEC_DIR = Path("cais_spade_llm/specification/products")
_REQ_DIR = _SPEC_DIR / "requirements"
_GEO_DIR = _SPEC_DIR / "geometry"


def render(bridge: SystemBridge) -> None:
    ui.label("Product Agent").classes("text-2xl font-bold px-6 pt-6")

    with ui.row().classes("w-full px-6 gap-6 items-start"):
      with ui.column().classes("flex-grow gap-6 min-w-0"):

        # ── Editable Assembly Requirements (top) ──────────────────
        with ui.card().classes("w-full"):
            ui.label("Assembly Requirements").classes("text-lg font-semibold mb-2")

            _REQ_DIR.mkdir(parents=True, exist_ok=True)
            status_label = ui.label("").classes("text-sm")

            # State: track which file is selected.
            req_select = ui.select(
                {},
                label="Requirement File",
            ).classes("w-64")

            req_editor = ui.textarea(label="Edit requirements").classes(
                "w-full font-mono"
            ).props("outlined autogrow")

            def _refresh_file_list(select_path: str | None = None):
                """Re-scan the requirements directory and update the selector."""
                req_files = sorted(_REQ_DIR.glob("*.txt"))
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
                if not req_select.value:
                    status_label.text = "No file selected"
                    status_label.classes(replace="text-sm text-amber-600")
                    return
                path = Path(req_select.value)
                path.write_text(req_editor.value)
                status_label.text = f"Saved {path.name}"
                status_label.classes(replace="text-sm text-green-600")

            def _delete_req():
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
                content = e.content.read()
                name = e.name if e.name.endswith(".txt") else e.name + ".txt"
                dest = _REQ_DIR / name
                dest.write_bytes(content)
                status_label.text = f"Uploaded {name}"
                status_label.classes(replace="text-sm text-green-600")
                _refresh_file_list(str(dest))

            async def _create_new():
                name_input.value = name_input.value.strip()
                if not name_input.value:
                    status_label.text = "Enter a file name first"
                    status_label.classes(replace="text-sm text-amber-600")
                    return
                name = name_input.value if name_input.value.endswith(".txt") else name_input.value + ".txt"
                dest = _REQ_DIR / name
                if dest.exists():
                    status_label.text = f"{name} already exists — select it to edit"
                    status_label.classes(replace="text-sm text-amber-600")
                    return
                dest.write_text("[Assembly Requirements]\n- ")
                status_label.text = f"Created {name}"
                status_label.classes(replace="text-sm text-green-600")
                name_input.value = ""
                _refresh_file_list(str(dest))

            req_select.on_value_change(_load_req)

            # Action buttons row.
            with ui.row().classes("gap-2 mt-1 items-end flex-wrap"):
                ui.button("Save", on_click=_save_req, icon="save").props("color=primary")
                ui.button("Reload", on_click=_load_req, icon="refresh").props("flat")
                ui.button("Delete", on_click=_delete_req, icon="delete").props("flat color=red")

            # Create new / upload row.
            with ui.row().classes("gap-2 mt-3 items-end flex-wrap"):
                name_input = ui.input(label="New file name", placeholder="e.g. my_product").classes("w-48")
                ui.button("Create", on_click=_create_new, icon="add").props("flat")
                ui.upload(
                    label="Upload .txt",
                    auto_upload=True,
                    on_upload=_handle_upload,
                    max_file_size=1_000_000,
                ).props("accept=.txt flat dense").classes("w-40")

            # Initial load.
            _refresh_file_list()

        # ── Product Spec Selector ────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Product Specifications").classes("text-lg font-semibold mb-2")

            product_files = bridge.list_product_files()
            if not product_files:
                ui.label("No product files found").classes("text-slate-400 italic")
                return

            product_select = ui.select(
                {f: Path(f).stem for f in product_files},
                value=product_files[0],
                label="Select Product",
            ).classes("w-64")

            config_display = ui.code("{}", language="json").classes("w-full mt-2")

            def _load_product():
                try:
                    data = bridge.load_config(product_select.value)
                    config_display.content = json.dumps(data, indent=2)
                except Exception as e:
                    config_display.content = str(e)

            product_select.on_value_change(_load_product)
            _load_product()

        # ── Part Tracker ─────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Part Tracker").classes("text-lg font-semibold mb-2")
            part_table = ui.table(
                columns=[
                    {"name": "part", "label": "Part", "field": "part", "sortable": True},
                    {"name": "state", "label": "State", "field": "state", "sortable": True},
                    {"name": "location", "label": "Location", "field": "location"},
                    {"name": "last_task", "label": "Last Task", "field": "last_task"},
                ],
                rows=[],
            ).classes("w-full")

            def _refresh_parts():
                pt = bridge.get_part_tracker()
                rows = []
                for part_name, info in pt.items():
                    rows.append({
                        "part": part_name,
                        "state": info.get("state", "unknown"),
                        "location": info.get("location") or info.get("last_known_location", "?"),
                        "last_task": info.get("last_successful_task", ""),
                    })
                part_table.rows = rows

            ui.timer(3.0, _refresh_parts)

        # ── Geometry ─────────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Part Geometry").classes("text-lg font-semibold mb-2")
            geo_container = ui.column().classes("w-full")

            if _GEO_DIR.exists():
                for f in sorted(_GEO_DIR.glob("*.json")):
                    try:
                        data = json.loads(f.read_text())
                        with geo_container:
                            ui.label(f.name).classes("font-semibold text-sm")
                            ui.table(
                                columns=[
                                    {"name": "part", "label": "Part", "field": "part"},
                                    {"name": "x", "label": "X", "field": "x"},
                                    {"name": "y", "label": "Y", "field": "y"},
                                    {"name": "z", "label": "Z", "field": "z"},
                                ],
                                rows=[
                                    {"part": k, **{ax: v.get(ax, 0) for ax in "xyz"}}
                                    for k, v in data.items()
                                    if isinstance(v, dict)
                                ],
                            ).classes("w-full mb-4")
                    except Exception:
                        pass

      # ── Right column: chat panel ──────────────────────────
      with ui.column().classes("w-96 shrink-0 sticky top-20 self-start"):
        render_chat(bridge, agent_jid="product", title="Product Agent Chat")
