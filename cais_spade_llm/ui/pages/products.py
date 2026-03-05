"""Products page: editable requirements (CRUD + upload), spec viewer, part tracker, geometry."""

from __future__ import annotations

import json
import re
from pathlib import Path

from nicegui import ui, events

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.agent_chat import render_chat


_SPEC_DIR = Path("cais_spade_llm/specification/products")
_REQ_DIR = _SPEC_DIR / "requirements"
_GEO_DIR = _SPEC_DIR / "geometry"


def render(bridge: SystemBridge) -> None:
    ui.add_head_html(
        """
        <style>
        .compact-upload { min-height: auto !important; }
        .compact-upload .q-uploader__list { display: none !important; }
        .compact-upload .q-uploader__header { min-height: auto !important; padding: 6px 8px !important; }
        </style>
        """
    )
    ui.label("Products").classes("text-2xl font-bold px-6 pt-6")

    with ui.row().classes("w-full px-6 gap-6 items-start"):
      with ui.column().classes("flex-grow gap-6 min-w-0"):

        # ── Editable Product Requirements (top) ───────────────────
        with ui.card().classes("w-full"):
            ui.label("Product Requirements").classes("text-lg font-semibold mb-2")

            _REQ_DIR.mkdir(parents=True, exist_ok=True)
            status_label = ui.label("").classes("text-sm")

            # State: track which file is selected.
            with ui.row().classes("w-full gap-2 items-end flex-nowrap"):
                req_select = ui.select(
                    {},
                    label="Requirement File",
                ).classes("w-56 shrink-0")
                name_input = ui.input(label="New file", placeholder="e.g. my_product").classes("w-56 shrink-0")
                create_btn = ui.button("Create", icon="add").props("flat")

            upload_widget = None
            with ui.row().classes("w-full items-end"):
                upload_widget = ui.upload(
                    label="Upload .txt",
                    auto_upload=True,
                    on_upload=lambda e: _handle_upload(e),
                    max_file_size=1_000_000,
                ).props("accept=.txt flat dense max-files=1 hide-upload-progress").classes("w-40 compact-upload")
            upload_info_label = ui.label("").classes("text-xs text-slate-600")

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
                linked_ok, linked_info = _sync_requirement_into_selected_product_spec(path)
                if linked_ok:
                    status_label.text = f"Saved {path.name} and updated {linked_info}"
                    status_label.classes(replace="text-sm text-green-600")
                else:
                    status_label.text = f"Saved {path.name} (spec link unchanged: {linked_info})"
                    status_label.classes(replace="text-sm text-amber-600")
                _refresh_file_list(str(path))

            def _sync_requirement_into_selected_product_spec(requirement_path: Path) -> tuple[bool, str]:
                try:
                    selected_product_file = str(product_select.value or "").strip()
                except NameError:
                    return False, "no product specification selected"
                if not selected_product_file:
                    return False, "no product specification selected"
                try:
                    raw = bridge.load_config(selected_product_file)
                    if not isinstance(raw, dict) or not raw:
                        return False, "invalid product specification format"
                    product_key = next(iter(raw.keys()))
                    product_meta = raw.get(product_key)
                    if not isinstance(product_meta, dict):
                        return False, "invalid product specification entry"

                    product_meta["product_specification_file"] = str(requirement_path)
                    bridge.save_config(selected_product_file, raw)
                    try:
                        _load_product()
                    except Exception:
                        pass
                    return True, Path(selected_product_file).name
                except Exception as exc:
                    return False, str(exc)

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

            _REQ_FILENAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*\.txt$")

            def _normalized_upload_name(raw_name: str) -> str:
                base_name = Path(str(raw_name or "").strip()).name
                if not base_name:
                    base_name = "uploaded"
                stem = Path(base_name).stem if Path(base_name).suffix else base_name
                stem = stem.lower()
                stem = re.sub(r"\s+", "_", stem)
                stem = re.sub(r"[^a-z0-9_-]", "_", stem)
                stem = re.sub(r"_+", "_", stem).strip("_-")
                if not stem:
                    stem = "uploaded"
                normalized = f"{stem}.txt"
                if not _REQ_FILENAME_RE.fullmatch(normalized):
                    normalized = "uploaded.txt"
                return normalized

            def _next_available_upload_path(filename: str) -> Path:
                candidate = _REQ_DIR / filename
                if not candidate.exists():
                    return candidate
                stem = candidate.stem or "uploaded"
                suffix = candidate.suffix or ".txt"
                idx = 1
                while True:
                    alt = _REQ_DIR / f"{stem}_{idx}{suffix}"
                    if not alt.exists():
                        return alt
                    idx += 1

            async def _handle_upload(e: events.UploadEventArguments):
                content = await e.file.read()
                original_name = str(getattr(e.file, "name", "") or "uploaded.txt")
                uploaded_name = _normalized_upload_name(original_name)
                dest = _next_available_upload_path(uploaded_name)
                dest.write_bytes(content)
                status_label.text = f"Uploaded as new file: {dest.name}"
                status_label.classes(replace="text-sm text-green-600")
                _refresh_file_list(str(dest))
                req_select.value = str(dest)
                _load_req()
                req_editor.update()
                if dest.name != original_name:
                    upload_info_label.text = f"Uploaded: {original_name} -> {dest.name}"
                else:
                    upload_info_label.text = f"Uploaded: {dest.name}"
                if upload_widget is not None:
                    try:
                        upload_widget.reset()
                    except Exception:
                        pass

            async def _create_new():
                name_input.value = name_input.value.strip()
                if not name_input.value:
                    status_label.text = "Enter a file name first"
                    status_label.classes(replace="text-sm text-amber-600")
                    return
                original_name = name_input.value
                name = _normalized_upload_name(original_name)
                dest = _REQ_DIR / name
                if dest.exists():
                    status_label.text = f"{name} already exists — select it to edit"
                    status_label.classes(replace="text-sm text-amber-600")
                    return
                dest.write_text("[Product Requirements]\n- ")
                if name != original_name and name != f"{original_name}.txt":
                    status_label.text = f"Created {name} (normalized from \"{original_name}\")"
                else:
                    status_label.text = f"Created {name}"
                status_label.classes(replace="text-sm text-green-600")
                name_input.value = ""
                _refresh_file_list(str(dest))

            req_select.on_value_change(_load_req)
            create_btn.on_click(_create_new)

            # Action buttons row.
            with ui.row().classes("gap-2 mt-1 items-end flex-wrap"):
                ui.button("Save", on_click=_save_req, icon="save").props("color=primary")
                ui.button("Delete", on_click=_delete_req, icon="delete").props("flat color=red")

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
            ui.label("Product Geometry").classes("text-lg font-semibold mb-1")
            ui.label(
                "Expand a geometry file to inspect Gazebo/Real slot coordinates and part metadata."
            ).classes("text-xs text-slate-500 mb-2")

            if not _GEO_DIR.exists():
                ui.label("No geometry files found.").classes("text-slate-400 italic")
            else:
                for geo_file in sorted(_GEO_DIR.glob("*.json")):
                    try:
                        data = json.loads(geo_file.read_text(encoding="utf-8"))
                    except Exception as exc:
                        with ui.expansion(geo_file.name, icon="warning").classes("w-full"):
                            ui.label(f"Failed to load geometry: {exc}").classes("text-red-700 text-sm")
                        continue

                    if not isinstance(data, dict):
                        with ui.expansion(geo_file.name, icon="warning").classes("w-full"):
                            ui.label("Invalid format: expected JSON object.").classes("text-red-700 text-sm")
                        continue

                    with ui.expansion(geo_file.name, icon="category").classes("w-full"):
                        if not data:
                            ui.label("No geometry environments found.").classes("text-slate-400 italic")
                            continue

                        for env_name, env_payload_raw in data.items():
                            env_payload = env_payload_raw if isinstance(env_payload_raw, dict) else {}
                            board_raw = env_payload.get("assembly_board", {})
                            parts_raw = env_payload.get("parts", {})
                            board = board_raw if isinstance(board_raw, dict) else {}
                            parts = parts_raw if isinstance(parts_raw, dict) else {}
                            center_raw = board.get("center", {})
                            center = center_raw if isinstance(center_raw, dict) else {}
                            slots_raw = board.get("slots", {})
                            slots = slots_raw if isinstance(slots_raw, dict) else {}
                            model_map_raw = parts.get("model_map", {})
                            heights_raw = parts.get("heights_m", {})
                            model_map = model_map_raw if isinstance(model_map_raw, dict) else {}
                            heights = heights_raw if isinstance(heights_raw, dict) else {}

                            with ui.expansion(
                                f"{str(env_name).upper()} Details",
                                value=str(env_name).lower() == "gazebo",
                            ).classes("w-full ml-2"):
                                board_rows = [
                                    {"field": "center_x_m", "value": center.get("x", "")},
                                    {"field": "center_y_m", "value": center.get("y", "")},
                                    {"field": "center_z_m", "value": center.get("z", "")},
                                    {"field": "slot_floor_z_m", "value": board.get("slot_floor_z_m", "")},
                                    {"field": "thickness_m", "value": board.get("thickness_m", "")},
                                ]
                                ui.label("Board").classes("text-sm font-semibold")
                                ui.table(
                                    columns=[
                                        {"name": "field", "label": "Field", "field": "field"},
                                        {"name": "value", "label": "Value", "field": "value"},
                                    ],
                                    rows=board_rows,
                                ).classes("w-full mb-3")

                                slot_rows = []
                                for part_name, xy in sorted(slots.items()):
                                    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                                        slot_rows.append(
                                            {
                                                "part": part_name,
                                                "x": xy[0],
                                                "y": xy[1],
                                            }
                                        )
                                ui.label("Slot Coordinates (relative XY)").classes("text-sm font-semibold")
                                if slot_rows:
                                    ui.table(
                                        columns=[
                                            {"name": "part", "label": "Part", "field": "part"},
                                            {"name": "x", "label": "X", "field": "x"},
                                            {"name": "y", "label": "Y", "field": "y"},
                                        ],
                                        rows=slot_rows,
                                    ).classes("w-full mb-3")
                                else:
                                    ui.label("No slot coordinates available.").classes("text-slate-400 italic text-sm")

                                part_rows = []
                                part_names = sorted(set(model_map.keys()) | set(heights.keys()))
                                for part_name in part_names:
                                    part_rows.append(
                                        {
                                            "part": part_name,
                                            "model": model_map.get(part_name, ""),
                                            "height_m": heights.get(part_name, ""),
                                        }
                                    )
                                ui.label("Part Metadata").classes("text-sm font-semibold")
                                if part_rows:
                                    ui.table(
                                        columns=[
                                            {"name": "part", "label": "Part", "field": "part"},
                                            {"name": "model", "label": "Model", "field": "model"},
                                            {"name": "height_m", "label": "Height (m)", "field": "height_m"},
                                        ],
                                        rows=part_rows,
                                    ).classes("w-full")
                                else:
                                    ui.label("No part metadata available.").classes("text-slate-400 italic text-sm")

      # ── Right column: chat panel ──────────────────────────
      with ui.column().classes("w-96 shrink-0 sticky top-20 self-start"):
        render_chat(bridge, agent_jid="product", title="Product Agent Chat")
