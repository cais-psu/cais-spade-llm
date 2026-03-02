"""Products page: spec viewer, part tracker, execution timeline."""

from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge


_SPEC_DIR = Path("cais_spade_llm/specification/products")
_REQ_DIR = _SPEC_DIR / "requirements"
_GEO_DIR = _SPEC_DIR / "geometry"


def render(bridge: SystemBridge) -> None:
    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-6"):
        ui.label("Products").classes("text-2xl font-bold")

        # ── Product Spec Selector ────────────────────────────────────
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

        # ── Requirements Text ────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Assembly Requirements").classes("text-lg font-semibold mb-2")
            req_container = ui.column().classes("w-full")

            def _load_requirements():
                req_container.clear()
                with req_container:
                    if _REQ_DIR.exists():
                        for f in sorted(_REQ_DIR.glob("*.txt")):
                            ui.label(f.name).classes("font-semibold text-sm mt-2")
                            ui.label(f.read_text()).classes("text-sm whitespace-pre-wrap font-mono bg-slate-50 p-3 rounded")
                    else:
                        ui.label("No requirements found").classes("text-slate-400 italic")

            _load_requirements()

        # ── Part Tracker ─────────────────────────────────────────────
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

        # ── Geometry ─────────────────────────────────────────────────
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
