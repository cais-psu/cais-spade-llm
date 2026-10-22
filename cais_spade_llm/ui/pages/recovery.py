"""Recorded prompts, validation findings, and explicit recovery diagnostic tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nicegui import ui

from cais_spade_llm.recovery_framework.diagnostics import ROOT
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
    """Show the failure, saved observations, generation, and execution workflow."""
    ui.label("recovery").classes("text-2xl font-semibold")
    if bridge is not None:
        from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

        render_part_slippage(bridge, is_active=is_active)


def _saved_evidence(*, root: Path, debug_root: Path, is_active, bridge) -> None:
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
    refresh()
