"""Recorded prompts, validation findings, and explicit recovery diagnostic tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from nicegui import ui

from cais_spade_llm.recovery_framework.diagnostics import MODES, ROOT, inspect_inputs, jobs, read_object
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
            "outline_source", "run_id", "stage", "turn", "tool_round", "attempt",
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
        with ui.expansion("Requests and raw responses", icon="chat").classes("w-full"):
            captures = data.get("captures") or []
            if not captures:
                ui.label("Exact application request: not captured").classes("text-amber-700")
            for capture in captures:
                record = capture["record"]
                title = (
                    f"{record.get('kind', 'record')} · round {record.get('tool_round', '?')} "
                    f"· attempt {record.get('attempt', '?')} · {capture['status']}"
                )
                if capture["status"] == "captured":
                    lazy_file(title, Path(capture["path"]))
                else:
                    ui.label(f"{title}: {capture['path']}").classes("text-amber-700")
            for reference, resolved in data["prompts"]:
                if resolved is not None:
                    lazy_file("Legacy prompt / prepared preview (not proof of a sent request)", resolved)
        if payload.get("prepared_preview"):
            ui.label("Prepared preview: not sent")
        lazy_file("responses / original record", path)
        _validation(data, path)


def _validation(data: dict, path: Path) -> None:
    payload = data["payload"]
    with ui.expansion("validation", icon="fact_check", value=True).classes("w-full"):
        _table(
            "Recorded validation stages",
            data["validations"],
            ("candidate_id", "validation_category", "validator_role", "status", "outcome", "mocked", "reason"),
        )
        for stage in data["validations"]:
            evidence = stage.get("evidence")
            title = f"Candidate {stage.get('candidate_id', '?')} · {stage.get('validator_role', '?')} · {stage.get('validation_category', '?')}"
            with ui.expansion(title).classes("w-full"):
                if not isinstance(evidence, dict):
                    ui.label("Detailed evidence: not recorded")
                    continue
                ui.label(f"Evidence source: {'mocked' if stage.get('mocked') else 'validator record'}")
                if evidence.get("feasibility_status"):
                    ui.badge(evidence["feasibility_status"])
                support = evidence.get("primitive_support") or {}
                if support:
                    _table("Backward derivation: matching successor transitions", support.get("matching_transitions") or [],
                           ("event_name", "parameter_bindings", "valuation_coverage", "composition_prerequisites"))
                    _table("Forward validation: primitive union and witnesses", support.get("primitives") or [],
                           ("primitive", "feasibility_status", "params", "reason", "contributors"))
                    ui.label(f"Valuation fields checked: {support.get('covered_valuation_fields', [])}")
                    ui.label(f"Valuation fields deferred to composition: {support.get('deferred_valuation_fields', [])}")
                safety = evidence.get("safety_context") or evidence.get("safety_ctx") or {}
                if safety:
                    _table("CCA requirements and DFA steps", safety.get("rule_checks") or [],
                           ("rule_id", "from", "label", "to", "accepting", "accepting_reachable", "status", "reason"))
                lazy_file("Complete recorded findings and parameter attempts", path)
        for candidate in data.get("validation_details") or []:
            with ui.expansion(f"Candidate {candidate.get('candidate_id', candidate.get('candidate_index', '?'))}: selection and projected state").classes("w-full"):
                _table("Recorded candidate", [candidate],
                       ("candidate_id", "selection_status", "projected_state", "task", "rejection_reason"))
                lazy_file("Complete candidate record", path)
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
                response_source = ui.select(
                    ["live", "fixture_response_replay"], value="live", label="Model response source",
                ).classes("w-full")
                replay = ui.input("Response replay JSON path (used only for explicit replay)").classes("w-full")
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
                with ui.expansion("Inspect selected inputs (prepared preview: not sent)", icon="preview").classes("w-full") as preview:
                    inputs_view = BackgroundSection()

                def inspect_selected() -> None:
                    if not preview.value:
                        return
                    source = custom.value or runtime_context.value
                    if not source or not scenario.value:
                        return

                    def read() -> dict:
                        return inspect_inputs(contained_path(root, source), Path(scenario.value), root=root)

                    def display(inputs: dict) -> None:
                        saved = inputs["runtime_context"]
                        ui.label(f"Input fingerprint: {inputs['fingerprint']}").classes("text-xs break-all")
                        ui.label(f"Evidence source: {saved.get('evidence_source', 'synthetic / not declared')}")
                        ui.label(f"Goal: {_cell(saved.get('goal_state'))}")
                        ui.label(f"Remaining obligations: {_cell(saved.get('obligation_targets'))}")
                        ui.label(f"Model settings: {_cell(inputs['generation_settings'])}")
                        _table("Resource observations", saved.get("resource_snapshots") or [],
                               ("resource_jid", "resource_state", "held_part", "current_pose", "evidence_source"))
                        lazy_file("Starting state, observations, goals and operator guidance", Path(inputs["context_source"]))
                        lazy_file("Failure scenario and trigger", Path(inputs["scenario_source"]))
                        dependencies = ui.select(list(inputs["files"]), label="Referenced input file", with_input=True).classes("w-full")
                        dependency_view = ui.column().classes("w-full")

                        def selected_dependency() -> None:
                            dependency_view.clear()
                            if dependencies.value and dependencies.value != "runtime_context":
                                with dependency_view:
                                    render_file(root / dependencies.value)

                        dependencies.on_value_change(selected_dependency)

                    inputs_view.load(read, display)

                preview.on_value_change(lambda _: inspect_selected())
                for control in (scenario, runtime_context, custom):
                    control.on_value_change(lambda _: inspect_selected())
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
                        if response_source.value == "fixture_response_replay" and not replay.value:
                            raise ValueError("Explicit replay requires a response replay JSON file")
                        directory = await jobs.start(
                            mode.value, context_path, Path(scenario.value), checkpoint_path,
                            replay_responses=(
                                contained_path(root, replay.value)
                                if response_source.value == "fixture_response_replay" and replay.value else None
                            ),
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


def render(*, root: Path = ROOT, debug_root: Path = DEBUG_ROOT, is_active=lambda: True, bridge=None) -> None:
    """Browse exact stage evidence and explicitly launch saved-context tests."""
    ui.label("recovery").classes("text-2xl font-semibold")
    if bridge is not None:
        from cais_spade_llm.ui.components.recovery_safety_preparation import render_safety_preparation

        render_safety_preparation(bridge)
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
                if "response_source" in example:
                    ui.badge(f"Model response source: {example['response_source']}").props("outline")
                    ui.label(f"Evidence source: {example['evidence_source']}").classes("text-sm")
                    ui.label("Operator guidance: " + ("assisted" if example["assisted"] else "unassisted" if example["assisted"] is False else "not recorded")).classes("text-sm")
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
                lazy_file("Predefined task-level DES audit", example["path"] / "task_des_audit.json")
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
