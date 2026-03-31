"""Products page: product config, editable requirements (CRUD + upload), part tracker, geometry."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from nicegui import ui, events

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.agent_chat import render_chat


_SPEC_DIR = Path("cais_spade_llm/specification/products")
_REQ_DIR = _SPEC_DIR / "requirements"
_GEO_DIR = _SPEC_DIR / "geometry"
_INIT_DIR = Path("cais_spade_llm/initialization/products")
_PRODUCT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_DEFAULT_PRODUCT_INSTRUCTIONS = "This product requires to be printed and assembled as specified."

_PRODUCT_CONFIG_ORDER = [
    "type",
    "jid",
    "password",
    "domain",
    "functions",
    "instructions",
    "product_specification_file",
    "product_geometry_file",
]
_READONLY_PRODUCT_CONFIG_KEYS = {"type"}
_CONTROLLED_PRODUCT_CONFIG_KEYS = {
    "type",
    "jid",
    "password",
    "domain",
    "instructions",
    "product_geometry_file",
}

# Geometry template matching assembly_board-v1 structure.
_GEO_TEMPLATE = {
    "gazebo": {
        "assembly_board": {
            "center": {"x": 0.0, "y": 0.0, "z": 1.02},
            "thickness_m": 0.01,
            "slot_floor_z_m": 1.025,
            "slots": {
                "PART_A": [0.0, 0.0],
            },
        },
        "parts": {
            "model_map": {"PART_A": "model_name"},
            "heights_m": {"PART_A": 0.01},
        },
    },
    "real": {
        "assembly_board": {
            "center": {"x": 0.0, "y": 0.0, "z": 1.02},
            "thickness_m": 0.01,
            "slot_floor_z_m": 1.025,
            "slots": {
                "PART_A": [0.0, 0.0],
            },
        },
        "parts": {
            "model_map": {"PART_A": "model_name"},
            "heights_m": {"PART_A": 0.01},
        },
    },
}


def _editable_product_fields() -> list[str]:
    """Fields the user can edit in the product configuration table."""
    return ["instructions", "product_geometry_file"]


def _default_product_meta(product_name: str = "") -> dict[str, Any]:
    name = str(product_name or "").strip()
    return {
        "type": "product",
        "jid": f"{name}@localhost" if name else "",
        "password": "none",
        "domain": "localhost",
        "functions": [],
        "instructions": _DEFAULT_PRODUCT_INSTRUCTIONS,
        "product_specification_file": str(_REQ_DIR / f"{name}.txt") if name else "",
        "product_geometry_file": "",
    }


def _normalize_product_meta(meta: dict[str, Any] | None, *, product_name: str = "") -> dict[str, Any]:
    out = dict(_default_product_meta(product_name))
    for key, value in dict(meta or {}).items():
        if str(key).strip() in {"cad_path", "replan_mode"}:
            continue
        out[str(key)] = value
    if not isinstance(out.get("functions"), list):
        out["functions"] = []
    return out


def _ordered_product_config_keys(meta: dict[str, Any]) -> list[str]:
    keys = [key for key in _PRODUCT_CONFIG_ORDER if key in meta and key != "cad_path"]
    extras = sorted(
        key for key in meta.keys()
        if key not in _PRODUCT_CONFIG_ORDER and key != "cad_path"
    )
    return keys + extras


def _format_product_config_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, (dict, list, bool, int, float)):
        return json.dumps(value, indent=2)
    return str(value)


def _parse_product_config_value(text: str) -> Any:
    raw = str(text or "")
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return raw


def _validate_geometry_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "top-level geometry JSON must be an object"

    env_keys = [key for key in ("gazebo", "real") if key in payload]
    if not env_keys:
        return "geometry JSON must define at least one of 'gazebo' or 'real'"

    for env_key in env_keys:
        env_payload = payload.get(env_key)
        if not isinstance(env_payload, dict):
            return f"{env_key} must be an object"

        board = env_payload.get("assembly_board")
        parts = env_payload.get("parts")
        if not isinstance(board, dict):
            return f"{env_key}.assembly_board must be an object"
        if not isinstance(parts, dict):
            return f"{env_key}.parts must be an object"

        center = board.get("center")
        slots = board.get("slots")
        if not isinstance(center, dict):
            return f"{env_key}.assembly_board.center must be an object"
        for axis in ("x", "y", "z"):
            if axis not in center:
                return f"{env_key}.assembly_board.center must include {axis}"
        if "thickness_m" not in board:
            return f"{env_key}.assembly_board must include thickness_m"
        if "slot_floor_z_m" not in board:
            return f"{env_key}.assembly_board must include slot_floor_z_m"
        if not isinstance(slots, dict):
            return f"{env_key}.assembly_board.slots must be an object"
        for part_name, xy in slots.items():
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                return f"{env_key}.assembly_board.slots.{part_name} must be a 2-element [x, y] array"

        model_map = parts.get("model_map")
        heights_m = parts.get("heights_m")
        if not isinstance(model_map, dict):
            return f"{env_key}.parts.model_map must be an object"
        if not isinstance(heights_m, dict):
            return f"{env_key}.parts.heights_m must be an object"

    return ""


def _paths_match(path_a: str | Path | None, path_b: str | Path | None) -> bool:
    raw_a = str(path_a or "").strip()
    raw_b = str(path_b or "").strip()
    if not raw_a or not raw_b:
        return False
    try:
        return Path(raw_a).resolve() == Path(raw_b).resolve()
    except Exception:
        return Path(raw_a) == Path(raw_b)


def _unlink_geometry_from_products(bridge: SystemBridge, geometry_path: str | Path) -> list[str]:
    target = str(geometry_path or "").strip()
    if not target:
        return []
    cleared_products: list[str] = []
    for product_file in bridge.list_product_files():
        data = bridge.load_config(product_file)
        changed = False
        if not isinstance(data, dict):
            continue
        for product_name, meta in list(data.items()):
            if not isinstance(meta, dict):
                continue
            current_geometry = str(meta.get("product_geometry_file", "") or "").strip()
            if not _paths_match(current_geometry, target):
                continue
            normalized = _normalize_product_meta(meta, product_name=str(product_name))
            normalized["product_geometry_file"] = ""
            data[str(product_name)] = normalized
            cleared_products.append(str(product_name))
            changed = True
        if changed:
            bridge.save_config(str(product_file), data)
    return cleared_products


def _delete_product_manifest(product_path: str | Path | None) -> bool:
    raw = str(product_path or "").strip()
    if not raw:
        return False
    path = Path(raw)
    if not path.exists() or path.suffix != ".json":
        return False
    path.unlink()
    return True


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

        # ── Product Configuration ───────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Product Configuration").classes("text-lg font-semibold mb-2")

            # --- Product selector ---
            product_files = bridge.list_product_files()

            product_select = ui.select(
                {f: Path(f).stem for f in product_files},
                value=product_files[0] if product_files else None,
                label="Product",
            ).classes("w-64")

            product_status_label = ui.label("").classes("text-sm")

            geo_options = {"": "(none)"}

            def _refresh_geo_options() -> None:
                geo_options.clear()
                geo_options[""] = "(none)"
                if not _GEO_DIR.exists():
                    return
                for gf in sorted(_GEO_DIR.glob("*.json")):
                    geo_options[str(gf)] = gf.stem

            _refresh_geo_options()

            config_display_ref: dict[str, Any] = {"widget": None}

            def _refresh_product_options(select_path: str | None = None) -> None:
                files = bridge.list_product_files()
                product_select.options = {f: Path(f).stem for f in files}
                if select_path and select_path in product_select.options:
                    product_select.value = select_path
                elif files:
                    product_select.value = files[0]
                else:
                    product_select.value = None
                product_select.update()

            def _refresh_json_viewer():
                widget = config_display_ref.get("widget")
                if widget is None:
                    return
                if _product_state.get("path"):
                    try:
                        data = bridge.load_config(_product_state["path"])
                        widget.content = json.dumps(data, indent=2)
                    except Exception:
                        widget.content = "{}"
                else:
                    widget.content = "{}"

            def _collect_product_meta(
                row_controls: list[dict[str, Any]],
                *,
                product_name: str,
            ) -> tuple[dict[str, Any] | None, str]:
                product_name = str(product_name or "").strip()
                meta: dict[str, Any] = {}
                seen_keys: set[str] = set()
                for row in row_controls:
                    key_input = row.get("key_input")
                    value_widget = row.get("value_widget")
                    key = str(getattr(key_input, "value", "") or "").strip()
                    if not key:
                        return None, "Config keys cannot be empty."
                    if key == "cad_path":
                        continue
                    if key in seen_keys:
                        return None, f"Duplicate config key: {key}"
                    seen_keys.add(key)

                    if key == "product_geometry_file":
                        value = str(getattr(value_widget, "value", "") or "").strip()
                    elif key in {"type", "jid", "password", "domain", "instructions"}:
                        value = str(getattr(value_widget, "value", "") or "")
                    else:
                        value = _parse_product_config_value(str(getattr(value_widget, "value", "") or ""))
                    meta[key] = value

                normalized = _normalize_product_meta(meta, product_name=product_name)
                if not str(normalized.get("jid", "")).strip():
                    normalized["jid"] = f"{product_name}@localhost" if product_name else ""
                if not isinstance(normalized.get("functions"), list):
                    return None, "functions must be a JSON array."
                return normalized, ""

            def _render_config_rows(
                container,
                row_controls: list[dict[str, Any]],
                meta: dict[str, Any],
                *,
                product_name: str,
            ) -> None:
                container.clear()
                row_controls.clear()
                ordered_keys = _ordered_product_config_keys(meta)
                with container:
                    with ui.row().classes("w-full gap-3 items-center text-xs font-semibold text-slate-500"):
                        ui.label("Key").classes("w-44")
                        ui.label("Value").classes("flex-1")
                        ui.label("Action").classes("w-16")

                    def _remove_extra_key(extra_key: str) -> None:
                        meta.pop(extra_key, None)
                        _render_config_rows(
                            container,
                            row_controls,
                            meta,
                            product_name=product_name,
                        )

                    for key in ordered_keys:
                        value = meta.get(key, "")
                        with ui.row().classes("w-full gap-3 items-start"):
                            key_input = ui.input(label="Key", value=key).classes("w-44")
                            if key in _CONTROLLED_PRODUCT_CONFIG_KEYS:
                                key_input.props("readonly")

                            if key == "product_geometry_file":
                                value_widget = ui.select(
                                    geo_options,
                                    value=str(value or ""),
                                    label="Value",
                                ).classes("flex-1")
                            elif key in {"type", "jid", "password", "domain"}:
                                value_widget = ui.input(
                                    label="Value",
                                    value=str(value or ""),
                                ).classes("flex-1")
                                if key in _READONLY_PRODUCT_CONFIG_KEYS:
                                    value_widget.props("readonly")
                            else:
                                rows = 2 if key in {"instructions", "functions"} else 1
                                value_widget = ui.textarea(
                                    label="Value",
                                    value=_format_product_config_value(value),
                                ).classes("flex-1 font-mono").props(
                                    f"outlined autogrow rows={rows}"
                                )

                            if key in _PRODUCT_CONFIG_ORDER:
                                ui.label("").classes("w-16")
                            else:
                                ui.button(
                                    icon="delete",
                                    on_click=lambda _=None, extra_key=key: _remove_extra_key(extra_key),
                                ).props("flat color=red").classes("w-16")

                            row_controls.append(
                                {
                                    "key_input": key_input,
                                    "value_widget": value_widget,
                                }
                            )

            with ui.expansion("Edit Configuration", icon="settings", value=False).classes("w-full"):
                ui.label(
                    "Edit the product JSON as table rows. Extra keys are allowed; cad_path is ignored and removed."
                ).classes("text-xs text-slate-500")
                selected_config_rows = ui.column().classes("w-full gap-2")
                selected_row_controls: list[dict[str, Any]] = []
                selected_add_key = None
                selected_add_value = None

                def _add_selected_config_row() -> None:
                    key = str(selected_add_key.value or "").strip()
                    if not key:
                        product_status_label.text = "Enter a config key to add."
                        product_status_label.classes(replace="text-sm text-amber-600")
                        return
                    if key == "cad_path":
                        product_status_label.text = "cad_path is disabled and will not be added."
                        product_status_label.classes(replace="text-sm text-amber-600")
                        return
                    if key in _product_state["meta"]:
                        product_status_label.text = f"{key} already exists in this product."
                        product_status_label.classes(replace="text-sm text-amber-600")
                        return
                    _product_state["meta"][key] = _parse_product_config_value(selected_add_value.value or "")
                    selected_add_key.value = ""
                    selected_add_value.value = ""
                    _render_config_rows(
                        selected_config_rows,
                        selected_row_controls,
                        _product_state["meta"],
                        product_name=str(_product_state.get("name", "") or ""),
                    )
                    product_status_label.text = ""
                    product_status_label.classes(replace="text-sm")

                with ui.row().classes("w-full gap-3 items-end flex-wrap"):
                    selected_add_key = ui.input(
                        label="New config key",
                        placeholder="e.g. product_specification_file",
                    ).classes("w-64")
                    selected_add_value = ui.textarea(
                        label="Value (JSON or text)",
                        placeholder='e.g. "cais_spade_llm/specification/products/requirements/assembly1.txt"',
                    ).classes("flex-1 font-mono").props("outlined autogrow rows=1")
                    ui.button("Add Config Row", on_click=_add_selected_config_row, icon="add").props("flat")

            # Product data state (mutable dict shared by load/save).
            _product_state: dict = {"name": "", "meta": {}, "path": ""}

            def _load_product():
                if not product_select.value:
                    product_status_label.text = ""
                    _product_state.update(name="", meta={}, path="")
                    _render_config_rows(
                        selected_config_rows,
                        selected_row_controls,
                        _default_product_meta(""),
                        product_name="",
                    )
                    _refresh_json_viewer()
                    return
                try:
                    data = bridge.load_config(product_select.value)
                    if isinstance(data, dict) and data:
                        name = next(iter(data.keys()), "")
                        meta = data.get(name, {})
                    else:
                        name, meta = "", {}

                    _product_state.update(
                        name=name,
                        meta=_normalize_product_meta(meta, product_name=name),
                        path=str(product_select.value),
                    )
                    geo_val = str(_product_state["meta"].get("product_geometry_file", "") or "")
                    if geo_val and geo_val not in geo_options:
                        geo_options[geo_val] = Path(geo_val).stem
                    _render_config_rows(
                        selected_config_rows,
                        selected_row_controls,
                        _product_state["meta"],
                        product_name=name,
                    )
                    product_status_label.text = ""
                    product_status_label.classes(replace="text-sm")
                    _refresh_json_viewer()
                except Exception as e:
                    product_status_label.text = f"Error: {e}"
                    product_status_label.classes(replace="text-sm text-red-600")

            def _save_product():
                path_str = _product_state.get("path", "")
                name = _product_state.get("name", "")
                if not path_str or not name:
                    product_status_label.text = "No product selected"
                    product_status_label.classes(replace="text-sm text-amber-600")
                    return
                try:
                    data = bridge.load_config(path_str)
                    meta, error = _collect_product_meta(selected_row_controls, product_name=name)
                    if error:
                        product_status_label.text = error
                        product_status_label.classes(replace="text-sm text-amber-600")
                        return

                    data[name] = meta
                    bridge.save_config(path_str, data)
                    _product_state["meta"] = meta

                    product_status_label.text = f"Saved {name}"
                    product_status_label.classes(replace="text-sm text-green-600")
                    _refresh_json_viewer()
                except Exception as e:
                    product_status_label.text = f"Save failed: {e}"
                    product_status_label.classes(replace="text-sm text-red-600")

            def _delete_product() -> None:
                path_str = str(_product_state.get("path", "") or "").strip()
                name = str(_product_state.get("name", "") or "").strip()
                if not path_str or not name:
                    product_status_label.text = "No product selected"
                    product_status_label.classes(replace="text-sm text-amber-600")
                    return
                deleted = _delete_product_manifest(path_str)
                if not deleted:
                    product_status_label.text = f"Product file missing: {Path(path_str).name}"
                    product_status_label.classes(replace="text-sm text-amber-600")
                    _refresh_product_options()
                    _load_product()
                    _refresh_requirement_file_list()
                    return
                product_status_label.text = f"Deleted {name}"
                product_status_label.classes(replace="text-sm text-green-600")
                _refresh_product_options()
                _load_product()
                _refresh_requirement_file_list()
                _refresh_json_viewer()

            with ui.row().classes("gap-2 mt-1"):
                ui.button("Save Product", on_click=_save_product, icon="save").props("color=primary")
                ui.button("Delete Product", on_click=_delete_product, icon="delete").props("flat color=red")

            product_select.on_value_change(_load_product)

            ui.separator().classes("my-2")

            # --- Create New Product (collapsible) ---
            with ui.expansion("Create New Product", icon="add_circle_outline", value=False).classes("w-full"):
                create_status = ui.label("").classes("text-sm")
                new_name_input = ui.input(
                    label="Product name",
                    placeholder="e.g. assembly_board-v2",
                ).classes("w-64")
                ui.label(
                    "Create the product JSON as table rows. Upload geometry below and select it here."
                ).classes("text-xs text-slate-500")
                new_product_meta: dict[str, Any] = _default_product_meta("")
                new_name_state = {"value": ""}
                create_config_rows = ui.column().classes("w-full gap-2")
                create_row_controls: list[dict[str, Any]] = []
                create_add_key = None
                create_add_value = None

                def _render_new_product_config() -> None:
                    _render_config_rows(
                        create_config_rows,
                        create_row_controls,
                        new_product_meta,
                        product_name=str(new_name_input.value or "").strip(),
                    )

                def _sync_new_name_defaults() -> None:
                    current_name = str(new_name_input.value or "").strip().lower().replace(" ", "_")
                    previous_name = str(new_name_state["value"] or "")
                    previous_default_jid = f"{previous_name}@localhost" if previous_name else ""
                    previous_default_req = str(_REQ_DIR / f"{previous_name}.txt") if previous_name else ""
                    current_jid = str(new_product_meta.get("jid", "") or "")
                    current_req = str(new_product_meta.get("product_specification_file", "") or "")
                    if not current_jid or current_jid == previous_default_jid:
                        new_product_meta["jid"] = f"{current_name}@localhost" if current_name else ""
                    if not current_req or current_req == previous_default_req:
                        new_product_meta["product_specification_file"] = (
                            str(_REQ_DIR / f"{current_name}.txt") if current_name else ""
                        )
                    new_name_state["value"] = current_name
                    _render_new_product_config()

                def _add_create_config_row() -> None:
                    key = str(create_add_key.value or "").strip()
                    if not key:
                        create_status.text = "Enter a config key to add."
                        create_status.classes(replace="text-sm text-amber-600")
                        return
                    if key == "cad_path":
                        create_status.text = "cad_path is disabled and will not be added."
                        create_status.classes(replace="text-sm text-amber-600")
                        return
                    if key in new_product_meta:
                        create_status.text = f"{key} already exists in the new product config."
                        create_status.classes(replace="text-sm text-amber-600")
                        return
                    new_product_meta[key] = _parse_product_config_value(create_add_value.value or "")
                    create_add_key.value = ""
                    create_add_value.value = ""
                    _render_new_product_config()
                    create_status.text = ""
                    create_status.classes(replace="text-sm")

                with ui.row().classes("w-full gap-3 items-end flex-wrap"):
                    create_add_key = ui.input(
                        label="New config key",
                        placeholder="e.g. product_specification_file",
                    ).classes("w-64")
                    create_add_value = ui.textarea(
                        label="Value (JSON or text)",
                        placeholder='e.g. "cais_spade_llm/specification/products/requirements/my_product.txt"',
                    ).classes("flex-1 font-mono").props("outlined autogrow rows=1")
                    ui.button("Add Config Row", on_click=_add_create_config_row, icon="add").props("flat")
                _render_new_product_config()
                new_name_input.on_value_change(lambda _: _sync_new_name_defaults())

                def _create_product():
                    raw_name = (new_name_input.value or "").strip()
                    if not raw_name:
                        create_status.text = "Enter a product name"
                        create_status.classes(replace="text-sm text-amber-600")
                        return
                    name = raw_name.lower().replace(" ", "_")
                    if not _PRODUCT_NAME_RE.fullmatch(name):
                        create_status.text = "Invalid name (use lowercase, digits, hyphens, underscores)"
                        create_status.classes(replace="text-sm text-amber-600")
                        return

                    _INIT_DIR.mkdir(parents=True, exist_ok=True)
                    dest = _INIT_DIR / f"{name}.json"
                    if dest.exists():
                        create_status.text = f"{name} already exists"
                        create_status.classes(replace="text-sm text-amber-600")
                        return

                    meta, error = _collect_product_meta(create_row_controls, product_name=name)
                    if error:
                        create_status.text = error
                        create_status.classes(replace="text-sm text-amber-600")
                        return
                    product_data = {name: meta}
                    dest.write_text(json.dumps(product_data, indent=2), encoding="utf-8")

                    # Refresh product dropdown.
                    _refresh_product_options(str(dest))

                    create_status.text = f"Created {name}"
                    create_status.classes(replace="text-sm text-green-600")
                    new_name_input.value = ""
                    new_product_meta.clear()
                    new_product_meta.update(_default_product_meta(""))
                    new_name_state["value"] = ""
                    _render_new_product_config()
                    _refresh_json_viewer()

                ui.button("Create Product", on_click=_create_product, icon="add").props("color=primary")

            # --- Collapsible product JSON viewer ---
            with ui.expansion("Product JSON (raw)", icon="data_object", value=False).classes("w-full"):
                config_display = ui.code("{}", language="json").classes("w-full")
                config_display_ref["widget"] = config_display

                # Refresh on product change.
                product_select.on_value_change(lambda _: _refresh_json_viewer())
                _refresh_json_viewer()

            ui.separator().classes("my-2")

            # --- Geometry: download template + upload ---
            ui.label("Geometry Assets For Create / Edit").classes("text-base font-semibold mb-1")
            ui.label(
                "Download an assembly_board-v1-style geometry template or upload a validated geometry .json file."
            ).classes("text-xs text-slate-500 mb-1")

            geo_status = ui.label("").classes("text-sm")

            with ui.row().classes("gap-2 items-end flex-wrap"):
                ui.button(
                    "Download Template",
                    icon="download",
                    on_click=lambda: ui.download(
                        json.dumps(_GEO_TEMPLATE, indent=2).encode(),
                        "geometry_template.json",
                    ),
                ).props("flat")

                geo_upload_widget = None
                geo_upload_widget = ui.upload(
                    label="Upload .json",
                    auto_upload=True,
                    on_upload=lambda e: _handle_geo_upload(e),
                    max_file_size=2_000_000,
                ).props("accept=.json flat dense max-files=1 hide-upload-progress").classes(
                    "w-44 compact-upload"
                )

            async def _handle_geo_upload(e: events.UploadEventArguments):
                content = await e.file.read()
                original_name = str(getattr(e.file, "name", "") or "geometry.json")
                # Normalize name.
                stem = Path(original_name).stem.lower()
                stem = re.sub(r"[^a-z0-9_-]", "_", stem).strip("_") or "geometry"
                dest_name = f"{stem}.json"
                _GEO_DIR.mkdir(parents=True, exist_ok=True)
                dest = _GEO_DIR / dest_name
                # Validate JSON + expected geometry structure.
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError as exc:
                    geo_status.text = f"Invalid JSON: {exc}"
                    geo_status.classes(replace="text-sm text-red-600")
                    return
                validation_error = _validate_geometry_payload(payload)
                if validation_error:
                    geo_status.text = f"Invalid geometry schema: {validation_error}"
                    geo_status.classes(replace="text-sm text-red-600")
                    return
                dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                geo_status.text = f"Uploaded geometry: {dest_name}"
                geo_status.classes(replace="text-sm text-green-600")
                # Refresh geometry dropdown in editor.
                new_key = str(dest)
                _refresh_geo_options()
                if _product_state.get("meta"):
                    _product_state["meta"]["product_geometry_file"] = new_key
                    _render_config_rows(
                        selected_config_rows,
                        selected_row_controls,
                        _product_state["meta"],
                        product_name=str(_product_state.get("name", "") or ""),
                    )
                    _persist_selected_product_meta()
                new_product_meta["product_geometry_file"] = new_key
                _render_new_product_config()
                if geo_upload_widget is not None:
                    try:
                        geo_upload_widget.reset()
                    except Exception:
                        pass
                _refresh_geometry_browser()

            ui.separator().classes("my-2")

            # --- Editable Product Requirements ---
            ui.label("Requirements").classes("text-base font-semibold mb-1")
            ui.label(
                "Select a requirement file to view or edit. The product's linked file is pre-selected."
            ).classes("text-xs text-slate-500 mb-1")

            _REQ_DIR.mkdir(parents=True, exist_ok=True)
            status_label = ui.label("").classes("text-sm")

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

            def _persist_selected_product_meta() -> None:
                path_str = str(_product_state.get("path", "") or "").strip()
                name = str(_product_state.get("name", "") or "").strip()
                if not path_str or not name:
                    return
                data = bridge.load_config(path_str)
                data[name] = _normalize_product_meta(_product_state.get("meta", {}), product_name=name)
                bridge.save_config(path_str, data)
                _refresh_json_viewer()

            def _selected_requirement_path() -> str:
                raw = str(_product_state.get("meta", {}).get("product_specification_file", "") or "").strip()
                return raw

            def _set_selected_requirement_path(path_value: str) -> None:
                raw = str(path_value or "").strip()
                _product_state.setdefault("meta", {})
                _product_state["meta"]["product_specification_file"] = raw
                if _product_state.get("name"):
                    _render_config_rows(
                        selected_config_rows,
                        selected_row_controls,
                        _product_state["meta"],
                        product_name=str(_product_state.get("name", "") or ""),
                    )
                    _persist_selected_product_meta()

            def _refresh_requirement_file_list(select_path: str | None = None):
                """Show all requirement files, pre-selecting the one linked to the product."""
                product_name = str(_product_state.get("name", "") or "").strip()
                current_path = str(select_path or _selected_requirement_path() or "").strip()
                if not product_name:
                    req_select.options = {}
                    req_select.value = None
                    req_select.update()
                    req_editor.value = ""
                    status_label.text = "Select a product first."
                    status_label.classes(replace="text-sm text-slate-600")
                    upload_info_label.text = ""
                    return
                # List all requirement files from the directory
                options: dict[str, str] = {}
                if _REQ_DIR.is_dir():
                    for p in sorted(_REQ_DIR.iterdir()):
                        if p.is_file() and p.suffix == ".txt":
                            options[str(p)] = p.name
                req_select.options = options
                if current_path and current_path in options:
                    req_select.value = current_path
                else:
                    req_select.value = next(iter(options.keys()), None)
                req_select.update()
                _load_req()

            def _load_req():
                if not req_select.value:
                    req_editor.value = ""
                    status_label.text = "No requirement file linked to this product."
                    status_label.classes(replace="text-sm text-slate-600")
                    return
                path = Path(req_select.value)
                if path.exists():
                    req_editor.value = path.read_text(encoding="utf-8")
                    status_label.text = ""
                    status_label.classes(replace="text-sm")
                else:
                    req_editor.value = "[Product Requirements]\n- "
                    status_label.text = f"Linked file is missing: {path.name}"
                    status_label.classes(replace="text-sm text-amber-600")

            def _save_req():
                if not req_select.value:
                    status_label.text = "No file selected"
                    status_label.classes(replace="text-sm text-amber-600")
                    return
                path = Path(req_select.value)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(req_editor.value, encoding="utf-8")
                _set_selected_requirement_path(str(path))
                status_label.text = f"Saved {path.name}"
                status_label.classes(replace="text-sm text-green-600")
                _refresh_requirement_file_list(str(path))

            def _delete_req():
                if not req_select.value:
                    return
                path = Path(req_select.value)
                name = path.name
                if path.exists():
                    path.unlink()
                # Clear the product meta reference so the file is fully unlinked.
                _product_state.setdefault("meta", {})
                _product_state["meta"]["product_specification_file"] = ""
                _persist_selected_product_meta()
                # Refresh the list to show remaining files.
                _refresh_requirement_file_list()
                status_label.text = f"Deleted {name}"
                status_label.classes(replace="text-sm text-red-600")

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
                if not str(_product_state.get("name", "") or "").strip():
                    status_label.text = "Select a product before uploading a requirement file."
                    status_label.classes(replace="text-sm text-amber-600")
                    return
                content = await e.file.read()
                original_name = str(getattr(e.file, "name", "") or "uploaded.txt")
                uploaded_name = _normalized_upload_name(original_name)
                dest = _next_available_upload_path(uploaded_name)
                dest.write_bytes(content)
                _set_selected_requirement_path(str(dest))
                status_label.text = f"Uploaded as new file: {dest.name}"
                status_label.classes(replace="text-sm text-green-600")
                _refresh_requirement_file_list(str(dest))
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
                product_name = str(_product_state.get("name", "") or "").strip()
                if not product_name:
                    status_label.text = "Select a product before creating a requirement file."
                    status_label.classes(replace="text-sm text-amber-600")
                    return
                name_input.value = name_input.value.strip()
                if not name_input.value:
                    name_input.value = product_name
                original_name = name_input.value
                name = _normalized_upload_name(original_name)
                dest = _REQ_DIR / name
                if dest.exists():
                    status_label.text = f"{name} already exists — select it to edit"
                    status_label.classes(replace="text-sm text-amber-600")
                    return
                dest.write_text("[Product Requirements]\n- ", encoding="utf-8")
                _set_selected_requirement_path(str(dest))
                if name != original_name and name != f"{original_name}.txt":
                    status_label.text = f"Created {name} (normalized from \"{original_name}\")"
                else:
                    status_label.text = f"Created {name}"
                status_label.classes(replace="text-sm text-green-600")
                name_input.value = ""
                _refresh_requirement_file_list(str(dest))

            req_select.on_value_change(_load_req)
            create_btn.on_click(_create_new)
            product_select.on_value_change(lambda _: _refresh_requirement_file_list())

            # Action buttons row.
            with ui.row().classes("gap-2 mt-1 items-end flex-wrap"):
                ui.button("Save", on_click=_save_req, icon="save").props("color=primary")
                ui.button("Delete", on_click=_delete_req, icon="delete").props("flat color=red")

            # Initial load.
            _load_product()
            _refresh_requirement_file_list()

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
                "Expand a geometry file to inspect Gazebo/Real slot coordinates and part metadata, or delete it."
            ).classes("text-xs text-slate-500 mb-2")
            geometry_browser = ui.column().classes("w-full gap-2")

            def _delete_geometry_file(geo_file: Path) -> None:
                target = str(geo_file)
                cleared_products = _unlink_geometry_from_products(bridge, target)
                if geo_file.exists():
                    geo_file.unlink()
                _refresh_geo_options()
                if _paths_match(_product_state.get("meta", {}).get("product_geometry_file", ""), target):
                    _product_state["meta"]["product_geometry_file"] = ""
                    _render_config_rows(
                        selected_config_rows,
                        selected_row_controls,
                        _product_state["meta"],
                        product_name=str(_product_state.get("name", "") or ""),
                    )
                    _refresh_json_viewer()
                if _paths_match(new_product_meta.get("product_geometry_file", ""), target):
                    new_product_meta["product_geometry_file"] = ""
                    _render_new_product_config()
                if cleared_products:
                    geo_status.text = (
                        f"Deleted geometry: {geo_file.name}. Cleared references from "
                        + ", ".join(sorted(cleared_products))
                    )
                else:
                    geo_status.text = f"Deleted geometry: {geo_file.name}"
                geo_status.classes(replace="text-sm text-green-600")
                _refresh_geometry_browser()

            def _refresh_geometry_browser() -> None:
                geometry_browser.clear()
                with geometry_browser:
                    if not _GEO_DIR.exists() or not list(_GEO_DIR.glob("*.json")):
                        ui.label("No geometry files found.").classes("text-slate-400 italic")
                        return
                    for geo_file in sorted(_GEO_DIR.glob("*.json")):
                        try:
                            data = json.loads(geo_file.read_text(encoding="utf-8"))
                        except Exception as exc:
                            with ui.expansion(geo_file.name, icon="warning").classes("w-full"):
                                with ui.row().classes("w-full justify-between items-center"):
                                    ui.label("Failed to load geometry.").classes("text-red-700 text-sm")
                                    ui.button(
                                        "Delete Geometry File",
                                        icon="delete",
                                        on_click=lambda _=None, path=geo_file: _delete_geometry_file(path),
                                    ).props("flat color=red")
                                ui.label(f"Failed to load geometry: {exc}").classes("text-red-700 text-sm")
                            continue

                        if not isinstance(data, dict):
                            with ui.expansion(geo_file.name, icon="warning").classes("w-full"):
                                with ui.row().classes("w-full justify-between items-center"):
                                    ui.label("Invalid geometry format.").classes("text-red-700 text-sm")
                                    ui.button(
                                        "Delete Geometry File",
                                        icon="delete",
                                        on_click=lambda _=None, path=geo_file: _delete_geometry_file(path),
                                    ).props("flat color=red")
                                ui.label("Invalid format: expected JSON object.").classes("text-red-700 text-sm")
                            continue

                        with ui.expansion(geo_file.name, icon="category").classes("w-full"):
                            with ui.row().classes("w-full justify-between items-center"):
                                ui.label("Geometry file details").classes("text-sm text-slate-500")
                                ui.button(
                                    "Delete Geometry File",
                                    icon="delete",
                                    on_click=lambda _=None, path=geo_file: _delete_geometry_file(path),
                                ).props("flat color=red")

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

            _refresh_geometry_browser()

      # ── Right column: chat panel ──────────────────────────
      with ui.column().classes("w-96 shrink-0 sticky top-20 self-start"):
        render_chat(bridge, agent_jid="product", title="Product Agent Chat")
