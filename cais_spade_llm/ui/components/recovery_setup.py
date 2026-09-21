"""Explicit configuration forms for recovery-framework experiments."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from nicegui import ui

from cais_spade_llm.recovery_framework import ROOT, read_json
from cais_spade_llm.resources.nominal_des import build_nominal_resource_des_models
from cais_spade_llm.resources.environment_models import build_environment_models
from cais_spade_llm.ui import recovery_setup as settings


def _options(paths: list[Path], current: str | None, root: Path) -> dict[str, str]:
    options = {str(path.relative_to(root)): path.name for path in paths}
    if current and current not in options:
        options[current] = current
    return options


def _keep_value(options: dict, value: Any) -> dict:
    if value is not None and value not in options:
        return {**options, value: f"{value} (unavailable for this selection)"}
    return options


def _render_definitions(draft: dict, root: Path, message: Any, changed, refresh_forms) -> None:
    def product_changed(e) -> None:
        draft["selected_product"] = e.value
        try:
            meta = next(iter(read_json(settings.reference_path(e.value, root)).values()))
            draft["product_geometry_file"] = meta["product_geometry_file"]
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
            message.text = f"Product configuration could not be read: {exc}"
            return
        changed()
        refresh_forms()

    def order_changed(e) -> None:
        draft["selected_product_order_file"] = e.value
        changed()
        refresh_forms()

    with ui.card().classes("w-full"):
        ui.label("Product Order and Safety").classes("font-semibold")
        ui.select(
            _options(
                sorted((root / "cais_spade_llm/initialization/products").glob("*.json")),
                draft["selected_product"],
                root,
            ),
            label="Product",
            value=draft["selected_product"],
            on_change=product_changed,
        ).classes("w-full")
        ui.select(
            _options(
                sorted((root / "cais_spade_llm/specification/products/orders").glob("*.json")),
                draft["selected_product_order_file"],
                root,
            ),
            label="Product Order JSON",
            value=draft["selected_product_order_file"],
            on_change=order_changed,
        ).classes("w-full")
        ui.label(
            "Edit components and quantity in Products. This setup supports quantity: 1."
        ).classes("text-sm")
        ui.label(f"Product Geometry: {draft['product_geometry_file']}").classes("text-xs break-all")
        ui.label(f"Plant configuration: {draft['scene_file']}").classes("text-xs break-all")
        safety_options = {
            "__NONE__": "None",
            **_options(
                sorted((root / "cais_spade_llm/specification/safety").glob("*.txt")),
                draft["selected_safety_file"],
                root,
            ),
        }
        ui.select(
            safety_options, label="Safety Requirement (.txt)", on_change=lambda _: changed()
        ).bind_value(draft, "selected_safety_file").classes("w-full")
        ui.select(
            _keep_value(settings.MODE_OPTIONS, draft["execution_mode"]),
            label="Mode",
            on_change=lambda _: changed(),
        ).bind_value(draft, "execution_mode")


def _render_resources(draft: dict, models: dict, changed, refresh_failure) -> None:
    rows = []
    for rid, model in models.items():
        events = [
            event
            for event in model["events"]
            if event["parameter_bindings"]["resource_id"].get("equals") == rid
        ]
        parts = list(
            dict.fromkeys(
                part
                for field in ("nominal_parts", "supported_products", "slots")
                for part in model["assignments"].get(field, [])
            )
        )
        held = model["state_variables"].get("held_part", {}).get("domain", [])
        parts.extend(part for part in held if part is not None and part not in parts)
        rows.append(
            {
                "resource_id": rid,
                "role": ", ".join(dict.fromkeys(event["event_name"] for event in events)),
                "eligible_parts": "part_name (runtime capability check)" if model.get("schema_version") == 2 else ", ".join(parts),
                "execution_support": model["execution_support"],
            }
        )
    with ui.card().classes("w-full"):
        ui.label("Resources permitted for this experiment").classes("font-semibold")
        ui.label(
            "Excluding a resource prevents its use from the beginning. A breakdown occurs later. "
            "Permitted resources constrain runtime environmental exploration."
        ).classes("text-sm text-slate-600")
        table = (
            ui.table(
                columns=[
                    {"name": key, "label": label, "field": key}
                    for key, label in (
                        ("resource_id", "Resource"),
                        ("role", "Configured tasks"),
                        ("eligible_parts", "Part binding"),
                        ("execution_support", "Execution support"),
                    )
                ],
                rows=rows,
                row_key="resource_id",
                selection="multiple",
                pagination=12,
            )
            .classes("w-full")
            .props("wrap-cells")
        )
        table.selected = [row for row in rows if row["resource_id"] in draft["permitted_resources"]]

        def select_resources() -> None:
            draft["permitted_resources"] = [row["resource_id"] for row in table.selected]
            changed()
            refresh_failure()

        table.on_select(select_resources)
        unknown = [rid for rid in draft["permitted_resources"] if rid not in models]
        if unknown:
            ui.label("Unknown configured resources: " + ", ".join(unknown)).classes("text-red-700")


def _render_slippage(
    failure: dict, models: dict, selected_parts: list[str], permitted: list[str], changed, refresh
) -> None:
    rid = failure["resource_id"]
    parts = (
        settings.eligible_parts(models[rid], selected_parts)
        if rid in models and rid in permitted
        else []
    )

    def part_changed(e) -> None:
        failure.update(part_name=e.value, event_id=None, event_name=None, parameter_bindings={})
        changed()
        refresh()

    ui.select(
        _keep_value({part: part for part in parts}, failure.get("part_name")),
        label="NIST part",
        value=failure.get("part_name"),
        on_change=part_changed,
    ).classes("w-full")
    events = (
        settings.part_task_events(models, rid, failure["part_name"])
        if rid in models and failure.get("part_name") in parts
        else []
    )

    def task_changed(e) -> None:
        event = next(event for event in events if event["event_id"] == e.value)
        failure.update(
            event_id=event["event_id"],
            event_name=event["event_name"],
            parameter_bindings=deepcopy(event["parameter_bindings"]),
        )
        changed()

    ui.select(
        _keep_value(
            {event["event_id"]: settings.task_label(event) for event in events},
            failure.get("event_id"),
        ),
        label="Task",
        value=failure.get("event_id"),
        on_change=task_changed,
    ).classes("w-full")
    ui.label(
        "Configured drop pose in world — a future Gazebo injection target, not an observation or a validated reachable pose."
    ).classes("text-sm")
    with ui.row().classes("flex-wrap"):
        for key in ("x", "y", "z"):
            ui.number(f"drop_pose.{key}", on_change=lambda _: changed()).bind_value(
                failure["drop_pose"], key
            ).classes("w-32")
    with ui.expansion("Drop orientation", value=False), ui.row().classes("flex-wrap"):
        for key in ("qx", "qy", "qz", "qw"):
            ui.number(key, on_change=lambda _: changed()).bind_value(
                failure["orientation_quat"], key
            ).classes("w-28")

    def condition_changed(e) -> None:
        failure["additional_condition"] = (
            {"resource_id": None, "part_name": None} if e.value else None
        )
        changed()
        refresh()

    condition = failure.get("additional_condition")
    ui.checkbox(
        "Another resource holds a selected part",
        value=condition is not None,
        on_change=condition_changed,
    )
    if condition is not None:
        others = [
            resource
            for resource in permitted
            if resource != rid
            and resource in models
            and settings.eligible_parts(models[resource], selected_parts)
        ]

        def other_changed(e) -> None:
            condition.update(resource_id=e.value, part_name=None)
            changed()
            refresh()

        ui.select(
            _keep_value({resource: resource for resource in others}, condition.get("resource_id")),
            label="Other resource",
            value=condition.get("resource_id"),
            on_change=other_changed,
        ).classes("w-full")
        other = condition.get("resource_id")
        other_parts = (
            [
                part
                for part in settings.eligible_parts(models[other], selected_parts)
                if part != failure.get("part_name")
            ]
            if other in others
            else []
        )
        ui.select(
            _keep_value({part: part for part in other_parts}, condition.get("part_name")),
            label="Other held part",
            on_change=lambda _: changed(),
        ).bind_value(condition, "part_name").classes("w-full")
        ui.label(
            "A future trigger condition; saving it does not change custody or select a recovery robot."
        ).classes("text-sm")


def _render_failure_form(
    draft: dict, models: dict, selected_parts: list[str], changed, refresh
) -> None:
    failure = draft.get("failure_scenario")

    def choose_scenario(e) -> None:
        scenario = e.value
        if not scenario:
            draft["failure_scenario"] = None
        else:
            rid = {
                "Conveyor breakdown": "Conveyor",
                "ur5e-1 breakdown": "ur5e-1",
                "Machining breakdown during part processing": "M1",
                "Part slippage": "ur5e-3",
            }[scenario]
            draft["failure_scenario"] = {
                "scenario": scenario,
                "resource_id": rid,
                "mode": "once",
                "checkpoint": "before_execute",
            }
            if scenario == "Part slippage":
                draft["failure_scenario"].update(
                    {
                        "part_name": None,
                        "event_id": None,
                        "event_name": None,
                        "parameter_bindings": {},
                        "drop_pose": {axis: None for axis in ("x", "y", "z")},
                        "orientation_quat": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
                        "additional_condition": None,
                    }
                )
        changed()
        refresh()

    def example(rid: str, part: str) -> None:
        draft["failure_scenario"] = settings.slippage_example(models, rid, part)
        changed()
        refresh()

    with ui.card().classes("w-full"):
        ui.label("Failure scenarios").classes("font-semibold")
        ui.select(
            {"": "None", **{name: name for name in settings.FAILURE_SCENARIOS}},
            label="Failure scenario",
            value=failure["scenario"] if failure else "",
            on_change=choose_scenario,
        ).classes("w-full")
        with ui.row().classes("gap-2 flex-wrap"):
            for rid, part in (("ur5e-3", "KET4_Square_4mm"), ("ur5e-4", "gear_large")):
                button = ui.button(
                    f"Example: {rid} / {part}",
                    on_click=lambda rid=rid, part=part: example(rid, part),
                ).props("flat")
                button.set_enabled(
                    rid in draft["permitted_resources"]
                    and rid in models
                    and part in settings.eligible_parts(models[rid], selected_parts)
                )
        ui.label(
            "Failure injection and recovery execution are not integrated. Examples fill a draft only."
        ).classes("text-amber-800 text-sm")
        if not failure:
            return
        ui.label("Occurrence: once per run").classes("text-sm")
        ui.select(
            _keep_value(settings.CHECKPOINTS, failure.get("checkpoint")),
            label="Trigger point",
            on_change=lambda _: changed(),
        ).bind_value(failure, "checkpoint").classes("w-full")
        with ui.expansion("Checkpoint identifiers"):
            for checkpoint, description in settings.CHECKPOINTS.items():
                ui.label(f"{checkpoint}: {description}").classes("text-sm")
        rid = failure["resource_id"]
        choices = [resource for resource in draft["permitted_resources"] if resource in models]
        scenario = failure["scenario"]
        if scenario == "Conveyor breakdown":
            choices = [resource for resource in choices if resource == "Conveyor"]
        elif scenario == "ur5e-1 breakdown":
            choices = [resource for resource in choices if resource == "ur5e-1"]
        elif scenario == "Machining breakdown during part processing":
            choices = [
                resource
                for resource in choices
                if any(
                    event["event_name"] == "machine_part" for event in models[resource]["events"]
                )
            ]
        else:
            choices = [
                resource
                for resource in choices
                if settings.eligible_parts(models[resource], selected_parts)
            ]

        def resource_changed(e) -> None:
            failure["resource_id"] = e.value
            if scenario == "Part slippage":
                failure.update(
                    part_name=None, event_id=None, event_name=None, parameter_bindings={}
                )
            changed()
            refresh()

        ui.select(
            _keep_value({resource: resource for resource in choices}, rid),
            label="Failure resource",
            value=rid,
            on_change=resource_changed,
        ).classes("w-full")
        if scenario != "Part slippage":
            ui.label(
                "Resource failure settings only. Processing progress and scenario execution will be implemented later."
            ).classes("text-sm")
            return
        _render_slippage(
            failure, models, selected_parts, draft["permitted_resources"], changed, refresh
        )


def _render_recovery_options(draft: dict, changed, bridge: Any) -> None:
    with ui.expansion("Recovery experiment settings", icon="settings").classes("w-full"):
        ui.select(
            _keep_value(settings.RECOVERY_MODE_OPTIONS, draft["runtime_recovery_mode"]),
            label="Recovery Handoff Mode",
            on_change=lambda _: changed(),
        ).bind_value(draft, "runtime_recovery_mode")
        ui.select(
            _keep_value(
                settings.RECOVERY_VALIDATION_OPTIONS, draft["runtime_recovery_validation_policy"]
            ),
            label="Recovery Safety Mode",
            on_change=lambda _: changed(),
        ).bind_value(draft, "runtime_recovery_validation_policy")
        archive_options = {"": "Existing automatic archive selection"}
        if hasattr(bridge, "list_runtime_recovery_archives"):
            archive_options.update(
                {entry["path"]: entry["label"] for entry in bridge.list_runtime_recovery_archives()}
            )

        def archive_changed(e) -> None:
            draft["runtime_recovery_archive_label"] = (
                archive_options.get(e.value, "") if e.value else ""
            )
            changed()

        ui.select(
            _keep_value(archive_options, draft["runtime_recovery_archive_path"]),
            label="Archived Recovery Run",
            with_input=True,
            on_change=archive_changed,
        ).bind_value(draft, "runtime_recovery_archive_path").classes("w-full")
        ui.input("Archived Recovery Run label", on_change=lambda _: changed()).bind_value(
            draft, "runtime_recovery_archive_label"
        ).classes("w-full")
        ui.label(
            "An empty Archived Recovery Run retains the existing Pre-ran automatic archive selection. These settings take effect only when Start System is clicked."
        ).classes("text-sm")
        ui.label(
            "Existing selector settings are read from their file and retained with this setup."
        ).classes("text-sm")
        ui.code(json.dumps(draft["recovery_experiment_settings"], indent=2), language="json")


def _read_definitions(draft: dict, root: Path, message: Any) -> tuple[dict, list[str]]:
    models: dict = {}
    parts: list[str] = []
    try:
        inputs = settings.product_inputs(
            draft["selected_product"], draft["selected_product_order_file"], root
        )
        builder = build_nominal_resource_des_models if "completion_conditions" in inputs["product_order"] else build_environment_models
        models = builder(read_json(settings.reference_path(draft["scene_file"], root)))
        parts = inputs["selected_parts"]
        if draft["product_geometry_file"] != inputs["product_geometry_file"]:
            raise ValueError(
                "Geometry reference differs from the product manifest; reload referenced definitions"
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        message.text = f"Configuration references need correction: {exc}"
    return models, parts


def render_setup(bridge: Any, *, root: Path = ROOT) -> None:
    """Edit a local draft and save only when Save setup is explicitly clicked."""
    path = root / settings.SETUP_RELATIVE
    try:
        draft = settings.load_setup(path, root=root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ui.label(f"Experiment setup could not be loaded: {exc}").classes("text-red-700")
        return
    with ui.row().classes("gap-4 items-center"):
        ui.label("Experiment setup").classes("text-lg font-semibold")
        ui.link("Products", "/products")
        ui.link("Resources", "/resources")
        ui.link("Safety", "/safety")
        ui.link("Open run", "/recovery-framework?tab=run")
    ui.label(
        "Choose saved definitions here. Save setup stores experiment settings only; "
        "it does not start planning, move robots, or inject failures."
    ).classes("text-sm text-slate-600")
    message = ui.label(
        "Saved setup" if path.exists() else "Unsaved defaults — quantity: 1"
    ).classes("text-sm")
    models, selected_parts = _read_definitions(draft, root, message)

    def changed() -> None:
        message.text = "Unsaved changes — run continues to show the saved setup."

    @ui.refreshable
    def definitions() -> None:
        _render_definitions(draft, root, message, changed, refresh_forms)

    @ui.refreshable
    def resources() -> None:
        _render_resources(draft, models, changed, failure_form.refresh)

    @ui.refreshable
    def failure_form() -> None:
        _render_failure_form(draft, models, selected_parts, changed, failure_form.refresh)

    @ui.refreshable
    def recovery_options() -> None:
        _render_recovery_options(draft, changed, bridge)

    def save() -> None:
        if any(
            bool(getattr(bridge, field, False))
            for field in ("system_running", "_starting", "_stopping")
        ):
            message.text = "Stop the system before saving experiment settings."
            return
        try:
            settings.save_setup(deepcopy(draft), path, root=root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            message.text = f"Setup was not saved: {exc}"
            return
        reason = settings.startup_block_reason(draft, root=root)
        message.text = "Setup saved. " + (
            reason or "No failure scenario or resource restriction is selected."
        )
        saved_json.content = json.dumps(draft, indent=2, ensure_ascii=False)

    def reload_saved() -> None:
        try:
            loaded = settings.load_setup(path, root=root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            message.text = f"Saved setup could not be loaded: {exc}"
            return
        draft.clear()
        draft.update(loaded)
        message.text = "Saved setup reloaded" if path.exists() else "Unsaved defaults"
        refresh_forms()
        saved_json.content = json.dumps(draft, indent=2, ensure_ascii=False)

    def reload_definitions() -> None:
        try:
            meta = next(
                iter(read_json(settings.reference_path(draft["selected_product"], root)).values())
            )
            draft["product_geometry_file"] = meta["product_geometry_file"]
            draft["recovery_experiment_settings"] = read_json(
                settings.reference_path(draft["recovery_experiment_settings_file"], root)
            )
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
            message.text = f"Referenced definitions could not be reloaded: {exc}"
            return
        changed()
        refresh_forms()

    def refresh_forms() -> None:
        nonlocal models, selected_parts
        models, selected_parts = _read_definitions(draft, root, message)
        definitions.refresh()
        resources.refresh()
        failure_form.refresh()
        recovery_options.refresh()

    definitions()
    resources()
    failure_form()
    recovery_options()
    with ui.row().classes("gap-3"):
        ui.button("Save setup", icon="save", on_click=save).props("color=primary")
        ui.button("Reload saved setup", icon="refresh", on_click=reload_saved).props("flat")
        ui.button("Reload referenced definitions", on_click=reload_definitions).props("flat")
    with ui.expansion("Saved setup JSON", icon="data_object").classes("w-full"):
        saved_json = ui.code(
            json.dumps(draft, indent=2, ensure_ascii=False), language="json"
        ).classes("w-full")
    ui.label(
        "Product planning: automatic planning through Start System and optional plan generation "
        "in setup are the next integration step. Existing saved offline plans remain in Products and Resources."
    ).classes("text-sm text-slate-600")
