"""Inspect Spec2Primitives configuration and planned paper protocols."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from nicegui import ui

from cais_spade_llm.spec2primitives.adapters.ui_runtime import Spec2PrimitivesUIRuntime
from cais_spade_llm.spec2primitives.project_results import (
    NOT_RECORDED,
    read_artifact,
    read_interactions,
)

_ROOT = Path(__file__).resolve().parent


def render_setup(runtime: Spec2PrimitivesUIRuntime) -> None:
    """Read configuration, saved setup, and protocols without capturing context."""
    ui.label("Current setup").classes("text-lg font-semibold")
    ui.label(
        "Enter a requirement and capture fresh context in run. "
        "Inspecting saved setup here does not select or resume that interaction."
    ).classes("text-sm text-slate-600")
    with ui.expansion("Loaded model settings and grounding limits", icon="settings").classes(
        "w-full"
    ):
        config = asdict(runtime.model_config) if runtime.model_config is not None else NOT_RECORDED
        ui.code(json.dumps(config, indent=2), language="json").classes("w-full")
    for label, path in (
        ("Approved documents and CAD", "references/products/approved_sources.json"),
        ("Configured resources", "config/workcell_profile.json"),
        ("Configured composition and validation budgets", "config/phase5_validation.json"),
    ):
        with ui.expansion(label, icon="description").classes("w-full"):
            ui.label(path).classes("text-xs")
            ui.code(json.dumps(read_artifact(_ROOT, path), indent=2), language="json").classes(
                "w-full"
            )
    with ui.expansion(
        "Saved requirement, selected resource, RGB-D, and primitive catalog", icon="history"
    ).classes("w-full"):
        try:
            records = {
                row["interaction_identifier"]: row
                for row in read_interactions(runtime.contexts_root)
            }
        except OSError as exc:
            records = {}
            ui.label(f"Saved setup is unavailable: {exc}").classes("text-amber-700")
        if not records:
            ui.label("No saved interactions are available.")
        else:
            selection = ui.select(
                list(records), label="Saved interaction", with_input=True
            ).classes("w-full")
            details = ui.column().classes("w-full")

            def show() -> None:
                details.clear()
                if selection.value not in records:
                    return
                row = records[selection.value]
                root = runtime.contexts_root / selection.value
                with details:
                    ui.label(f"product_requirement: {row['product_requirement']}")
                    ui.label(f"selected_resource_jid: {row['selected_resource_jid']}")
                    observations = sorted(
                        str(path.relative_to(root))
                        for path in root.glob("products/observations/*")
                        if path.is_dir()
                    )
                    ui.label("RGB-D records: " + (", ".join(observations) or NOT_RECORDED)).classes(
                        "break-all"
                    )
                    catalogs = sorted(
                        str(path.relative_to(root))
                        for path in root.glob("resources/*/primitive_catalog_snapshot/*.json")
                    )
                    for path in catalogs:
                        with ui.expansion(path, icon="account_tree").classes("w-full"):
                            ui.code(
                                json.dumps(read_artifact(root, path), indent=2), language="json"
                            ).classes("w-full")
                    if not catalogs:
                        ui.label(f"primitive_catalog: {NOT_RECORDED}")

            selection.on_value_change(show)
    ui.label("Planned paper experiments").classes("text-lg font-semibold")
    ui.label(
        "These protocols describe planned trials; saved observations are under results."
    ).classes("text-sm text-slate-600")
    for filename in ("COMPOSITION_EVALUATION.md", "BIAS_VALIDATION.md"):
        path = _ROOT / filename
        with ui.expansion(filename, icon="science").classes("w-full"):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                ui.label(f"Protocol is unavailable: {exc}").classes("text-amber-700")
                continue
            ui.markdown(text)
            ui.button(
                "Download protocol",
                icon="download",
                on_click=lambda path=path: ui.download.file(path),
            )
