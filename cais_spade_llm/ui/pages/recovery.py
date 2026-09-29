"""Recorded prompts, validation findings, and explicit recovery diagnostic tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from nicegui import ui

from cais_spade_llm.recovery_framework.diagnostics import MODES, ROOT, jobs, read_object
from cais_spade_llm.ui.evidence import (
    BackgroundSection,
    TablePager,
    contained_path,
    lazy_file,
    render_file,
)
from cais_spade_llm.ui.recovery_evidence import (
    DEBUG_ROOT,
    STAGES,
    list_examples,
    read_stage_record,
    stage_records,
)
from cais_spade_llm.ui.refresh import PageRefresh


def _cell(value: Any) -> str:
    if value is None:
        return "not recorded"
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= 500 else text[:500] + "… (see original record)"


def _table(label: str, rows: list[dict], fields: tuple[str, ...]) -> None:
    ui.label(label).classes("font-semibold mt-3")
    if not rows:
        ui.label("not recorded").classes("text-sm text-slate-500")
        return
    projected = [
        {"row": index, **{key: _cell(row.get(key)) for key in fields}}
        for index, row in enumerate(rows)
        if isinstance(row, dict)
    ]
    table = (
        ui.table(
            columns=[{"name": key, "field": key, "label": key, "align": "left"} for key in fields],
            rows=[],
            row_key="row",
        )
        .classes("w-full")
        .props("wrap-cells")
    )
    TablePager(table, lambda: projected).update()


def _record(data: dict, path: Path) -> None:
    payload, response = data["payload"], data["response"]
    with ui.card().classes("w-full gap-3"):
        with ui.row().classes("items-center gap-2"):
            for key in ("status", "decision", "selection_status", "recovery_safety_status"):
                if key in payload:
                    ui.badge(f"{key}: {payload[key]}").props("color=blue-grey outline")
        for key in (
            "turn_index",
            "phase",
            "outline_id",
            "resource_jid",
            "primitive_local_turn_index",
            "scenario",
            "outline_source",
        ):
            if key in payload:
                ui.label(f"{key}: {_cell(payload[key])}").classes("text-sm")
        if "recovery_safety_status" in payload:
            ui.label(
                f"Generated rules: {len(payload.get('rules') or [])} · Recorded rule results: {len(data['rules'])}"
            ).classes("font-semibold")
        _table(
            "Accepted trace / selected transition",
            data["trace"],
            ("outline_id", "event_name", "resource_jid", "expected_end_state"),
        )
        if data["candidates"]:
            _table(
                "Candidate decisions",
                data["candidates"],
                (
                    "candidate_id",
                    "valid",
                    "selection_status",
                    "remaining_blocked_issues",
                    "rejection_reason",
                    "constraint_codes",
                ),
            )
        if response.get("primitive_steps"):
            _table("primitive_steps", response["primitive_steps"], ("primitive", "params"))
        if response.get("context_requests"):
            _table(
                "context_requests",
                response["context_requests"],
                ("request_type", "resource_jid", "part_name", "reason"),
            )
        with ui.expansion("prompts", icon="chat").classes("w-full"):
            if not data["prompts"]:
                ui.label("prompts: not recorded")
            for reference, resolved in data["prompts"]:
                ui.label(reference).classes("text-xs break-all text-slate-500")
                if resolved is None:
                    ui.label("Prompt file not recorded in this example.").classes("text-amber-700")
                else:
                    lazy_file("Exact recorded prompt", resolved)
        lazy_file("responses / original record", path)
        _validation(data)


def _validation(data: dict) -> None:
    payload = data["payload"]
    with ui.expansion("validation", icon="fact_check", value=True).classes("w-full"):
        _table(
            "Recorded validation stages",
            data["validations"],
            ("candidate_id", "validation_category", "validator_role", "status", "mocked", "reason"),
        )
        if data["rules"]:
            _table(
                "Recovery safety rule results",
                data["rules"],
                ("rule_id", "status", "failure_reason", "aps", "ltlf"),
            )
        for key in ("transition_validation", "candidate_rejection_feedback", "failure_reason"):
            if payload.get(key):
                ui.label(f"{key}: {_cell(payload[key])}").classes("text-sm break-all")
        if (
            not data["validations"]
            and not data["rules"]
            and not payload.get("transition_validation")
        ):
            ui.label("Validation findings are not recorded for this artifact.").classes(
                "text-amber-700"
            )
        for reference, resolved in data["dfa"]:
            ui.label(reference).classes("text-xs break-all")
            if resolved is not None:
                lazy_file("Recorded DFA", resolved)
            else:
                ui.label("Associated DFA file is not recorded in this example.")


def _test_controls(refresh, *, root: Path, debug_root: Path, is_active) -> None:
    with ui.expansion("Test a failure scenario", icon="science").classes("w-full"):
        ui.label(
            "Generate and validate using saved context. Resource observations come from that context."
        ).classes("text-sm text-slate-600")
        options = BackgroundSection()
        form = ui.column().classes("w-full")

        def read_options() -> dict:
            scenarios = sorted(
                (root / "cais_spade_llm/initialization/failure_scenarios").glob("*.json")
            )
            contexts = sorted((root / "test/fixtures").rglob("runtime_context*.json"))
            contexts += sorted((debug_root / "test_runs").glob("*/inputs/runtime_context.json"))
            checkpoints = sorted(
                (debug_root / "test_runs").glob("*/outline_checkpoint.json"), reverse=True
            )
            return {"scenarios": scenarios, "contexts": contexts, "checkpoints": checkpoints}

        def show(values: dict) -> None:
            form.clear()
            with form:
                scenario = ui.select(
                    {str(p): p.name for p in values["scenarios"]},
                    label="Scenario JSON",
                    with_input=True,
                ).classes("w-full")
                context_options = {str(p): str(p.relative_to(root)) for p in values["contexts"]}
                runtime_context = ui.select(
                    context_options, label="runtime_context JSON", with_input=True
                ).classes("w-full")
                custom = ui.input("Additional saved runtime_context JSON path").classes("w-full")
                mode = ui.select(list(MODES), value="outline", label="Test mode").classes("w-full")
                checkpoint = ui.select(
                    {
                        "": "Generate a new outline",
                        **{str(p): p.parent.name for p in values["checkpoints"]},
                    },
                    value="",
                    label="Accepted outline",
                    with_input=True,
                ).classes("w-full")
                ui.label(
                    "Reusing an outline requires matching saved inputs, settings, and validator code."
                ).classes("text-xs text-slate-500")
                message = ui.label().classes("text-sm")
                progress = ui.column().classes("w-full")
                last_status = None

                async def start() -> None:
                    start_button.disable()
                    message.text = "Checking saved inputs…"
                    try:
                        source = custom.value or runtime_context.value
                        if not source or not scenario.value:
                            raise ValueError(
                                "Select a scenario and its matching runtime_context JSON"
                            )
                        context_path = contained_path(root, source)
                        checkpoint_path = (
                            contained_path(debug_root / "test_runs", checkpoint.value)
                            if checkpoint.value
                            else None
                        )
                        directory = await jobs.start(
                            mode.value, context_path, Path(scenario.value), checkpoint_path
                        )
                        message.text = (
                            "Test started. Prompts and validation are saved as stages complete."
                        )
                        refresh(str(directory.relative_to(debug_root)))
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        message.text = f"Test was not started: {exc}"
                    finally:
                        if not start_button.is_deleted:
                            start_button.set_enabled(jobs.task is None or jobs.task.done())

                async def cancel() -> None:
                    await jobs.cancel()
                    refresh()
                    await poll()

                with ui.row().classes("gap-2"):
                    start_button = ui.button("Run test", icon="play_arrow", on_click=start)
                    cancel_button = ui.button("Cancel test", icon="stop", on_click=cancel).props(
                        "outline"
                    )
                    cancel_button.disable()
                    ui.button(
                        "Refresh saved inputs",
                        icon="refresh",
                        on_click=lambda: options.load(read_options, show),
                    ).props("flat")

                async def poll() -> None:
                    nonlocal last_status
                    active = jobs.task is not None and not jobs.task.done()
                    start_button.set_enabled(not active)
                    cancel_button.set_enabled(active)
                    if jobs.current is None:
                        return
                    try:
                        record = await asyncio.to_thread(read_object, jobs.current / "run.json")
                    except (OSError, ValueError):
                        return
                    if progress.is_deleted:
                        return
                    status = (record["id"], record["status"])
                    if status == last_status:
                        return
                    last_status = status
                    progress.clear()
                    with progress:
                        ui.label(f"{record['id']} · {record['mode']} · {record['status']}").classes(
                            "font-semibold"
                        )
                        lazy_file("Test progress log", jobs.current / "runner.log")
                    if not active:
                        refresh()

                PageRefresh(is_active=lambda: not form.is_deleted and is_active()).timer(1, poll)

        options.load(read_options, show)


def _all_files(example: dict) -> None:
    with ui.expansion("All recorded files", icon="folder_open").classes("w-full") as all_files:
        files_view = BackgroundSection()

    def files_loaded(paths: list[Path]) -> None:
        file = ui.select(
            {str(p): str(p.relative_to(example["path"])) for p in paths},
            label="Recorded file",
            with_input=True,
        ).classes("w-full")
        preview = ui.column().classes("w-full")

        def selected_file() -> None:
            preview.clear()
            if not file.value:
                return
            path = contained_path(example["path"], file.value)
            with preview:
                if path.suffix == ".png":
                    ui.image(str(path)).classes("max-w-3xl")
                else:
                    render_file(path)

        file.on_value_change(selected_file)

    all_files.on_value_change(
        lambda e: files_view.load(
            lambda: sorted(
                p
                for p in example["path"].rglob("*")
                if p.suffix in {".json", ".txt", ".dot", ".png"} and p.is_file()
            ),
            files_loaded,
        )
        if e.value
        else None
    )


def render(*, root: Path = ROOT, debug_root: Path = DEBUG_ROOT, is_active=lambda: True) -> None:
    """Browse exact stage evidence and explicitly launch saved-context tests."""
    ui.label("recovery").classes("text-2xl font-semibold")
    ui.label("Inspect the prompts, decisions, and validation behind recovery.").classes(
        "text-slate-600"
    )
    examples: dict[str, dict] = {}
    with ui.row().classes("w-full items-center"):
        selector = ui.select({}, label="Saved example / test", with_input=True).classes("grow")
        refresh_button = ui.button("Refresh evidence", icon="refresh").props("outline")
    loading = BackgroundSection()
    view = ui.column().classes("w-full")

    def select() -> None:
        view.clear()
        example = examples.get(selector.value)
        if example is None:
            return
        with view:
            with ui.row().classes("gap-3 items-center"):
                ui.badge(example["status"]).props("color=blue-grey outline")
                ui.label(f"scenario: {example['scenario']}").classes("text-sm")
                ui.label(example["id"]).classes("text-xs text-slate-500")
            stage = ui.toggle(
                STAGES, value=next(iter(example["stages"]), "recovery_outline")
            ).props("no-caps")
            records = BackgroundSection()

            def select_stage() -> None:
                directory = example["stages"].get(stage.value)
                current_stage = stage.value
                if directory is None:
                    records.load(lambda: [], show_records)
                else:
                    records.load(lambda: stage_records(directory, current_stage), show_records)

            def show_records(paths: list[Path]) -> None:
                if not paths:
                    ui.label("This stage is not recorded in the selected example.").classes(
                        "text-slate-500"
                    )
                    return
                default = next(
                    (p for p in paths if p.name == "recovery_safety_generation_result.json"),
                    paths[-1],
                )
                artifact = ui.select(
                    {str(p): p.name for p in paths},
                    value=str(default),
                    label="Turn / artifact",
                    with_input=True,
                ).classes("w-full")
                detail = BackgroundSection()

                def show_record() -> None:
                    path = Path(artifact.value)
                    detail.load(
                        lambda: read_stage_record(path.parent, path),
                        lambda data: _record(data, path),
                    )

                artifact.on_value_change(show_record)
                show_record()

            stage.on_value_change(select_stage)
            select_stage()
            _all_files(example)
            if example["id"].startswith("test_runs/"):
                lazy_file("Test inputs and settings", example["path"] / "inputs.json")
                lazy_file("Test progress log", example["path"] / "runner.log")

    def loaded(rows: list[dict], selection: str | None = None) -> None:
        examples.clear()
        examples.update((row["id"], row) for row in rows)
        previous = selector.value
        value = (
            selection
            if selection in examples
            else (previous if previous in examples else next(iter(examples), None))
        )
        selector.set_options({key: row["label"] for key, row in examples.items()}, value=value)
        if value == previous:
            select()
        if not rows:
            ui.label("No saved recovery examples or tests are available.")

    def refresh(selection: str | None = None) -> None:
        loading.load(lambda: list_examples(debug_root), lambda rows: loaded(rows, selection))

    selector.on_value_change(select)
    refresh_button.on_click(lambda: refresh())
    _test_controls(refresh, root=root, debug_root=debug_root, is_active=is_active)
    refresh()
