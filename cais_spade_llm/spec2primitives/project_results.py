"""Read-only inspection of saved Spec2Primitives interactions and programs."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

NOT_RECORDED = "not recorded"
RESULT_FIELDS = (
    "interaction_identifier",
    "product_requirement",
    "grounding_status",
    "selected_resource_jid",
    "refinement_run",
    "composition_status",
    "first_pass_validation",
    "candidate_count",
    "elapsed_sec",
    "execution_run",
    "simulation_status",
    "assembly_success",
    "physical_outcome",
    "record_error",
)


def read_artifact(root: Path, relative_path: str) -> dict[str, Any]:
    """Read a saved object without following references outside the interaction."""
    try:
        path = root / relative_path
        path.resolve().relative_to(root.resolve())
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        return {"record_error": f"{type(exc).__name__}: {exc}"}
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        return payload
    except ValueError as exc:
        return {"record_error": f"{type(exc).__name__}: {exc}", "raw_text": text}


def _value(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    return NOT_RECORDED if value is None or value == "" else value


def _latest(root: Path, pattern: str) -> dict[str, Any]:
    paths = sorted(root.glob(pattern))
    return read_artifact(root, str(paths[-1].relative_to(root))) if paths else {}


def _first_pass_validation(root: Path, run: Path, result: dict[str, Any]) -> Any:
    candidates = result.get("candidate_refs")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return NOT_RECORDED
    first_ref = candidates[0].get("ref")
    if not isinstance(first_ref, str):
        return NOT_RECORDED
    for path in sorted(run.glob("validation_*.json")):
        report = read_artifact(root, str(path.relative_to(root)))
        candidate_ref = report.get("candidate_ref")
        if isinstance(candidate_ref, dict) and candidate_ref.get("ref") == first_ref:
            return _value(report, "status")
    return NOT_RECORDED


def read_interactions(contexts_root: Path) -> list[dict[str, Any]]:
    """List every saved interaction, including incomplete and failed attempts.

    Each row is an interaction, not an independent experimental trial. The
    latest refinement run supplies summary fields; every version remains in
    the artifact browser. No runtime or evaluator authority is invoked.
    """
    rows = []
    for root in sorted(contexts_root.glob("interaction_*")):
        if not root.is_dir():
            continue
        try:
            root.resolve().relative_to(contexts_root.resolve())
        except ValueError:
            continue
        rows.append(_read_interaction(root))
    return rows


def _read_interaction(root: Path) -> dict[str, Any]:
    row = dict.fromkeys(RESULT_FIELDS, NOT_RECORDED)
    row["interaction_identifier"] = root.name
    errors: list[str] = []

    def record(relative: str) -> dict[str, Any]:
        payload = read_artifact(root, relative)
        if "record_error" in payload:
            errors.append(f"{relative}: {payload['record_error']}")
        return payload

    requirement = record("products/user_requirement/product_requirement.json")
    row["product_requirement"] = _value(requirement, "product_requirement")
    completion = _latest(root, "interaction_record/context_completion_*.json")
    turn = _latest(root, "interaction_record/turn_*.json")
    output = turn.get("PA_output")
    row["grounding_status"] = (
        _value(completion, "status")
        if completion
        else (_value(output, "grounding_status") if isinstance(output, dict) else NOT_RECORDED)
    )
    assignment = _latest(root, "composition/selected_ra_assignments/assignment_*.json")
    row["selected_resource_jid"] = _value(assignment, "selected_resource_jid")
    runs = sorted(path for path in root.glob("composition/refinement_runs/run_*") if path.is_dir())
    if runs:
        latest_run = runs[-1]
        row["refinement_run"] = latest_run.name
        result_path = latest_run / "result.json"
        if result_path.exists():
            result = record(str(result_path.relative_to(root)))
            row["composition_status"] = _value(result, "status")
            row["first_pass_validation"] = _first_pass_validation(root, latest_run, result)
        events = sorted(latest_run.glob("event_*.json"))
        if events:
            event = record(str(events[-1].relative_to(root)))
            row["elapsed_sec"] = _value(event, "elapsed_sec")
    candidate_root = root / "composition/primitive_program_candidates"
    if candidate_root.is_dir():
        row["candidate_count"] = sum(
            1 for path in candidate_root.glob("attempt_*") if path.is_dir()
        )
    executions = sorted(path for path in root.glob("execution/run_*") if path.is_dir())
    if executions:
        row["execution_run"] = executions[-1].name
        result_path = executions[-1] / "result.json"
        if result_path.exists():
            result = record(str(result_path.relative_to(root)))
            if result.get("record_type") == "PrimitiveExecutionResult":
                row["simulation_status"] = _value(result, "status")
                row["assembly_success"] = _value(result, "assembly_success")
    for payload in (completion, turn, assignment):
        if "record_error" in payload:
            errors.append(payload["record_error"])
    row["record_error"] = "\n".join(errors) if errors else NOT_RECORDED
    return row


def artifact_paths(root: Path) -> list[str]:
    """List recorded setup, candidates, findings, timings, and execution evidence."""
    return sorted(str(path.relative_to(root)) for path in root.rglob("*.json") if path.is_file())


def results_csv(rows: list[dict[str, Any]]) -> str:
    """Export the same recorded values shown in the results table."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, NOT_RECORDED) for field in RESULT_FIELDS})
    return output.getvalue()


def render_results(contexts_root: Path) -> None:
    """Render saved evidence without restoring it into an active interaction."""
    from nicegui import ui

    ui.label("Saved Spec2Primitives evidence").classes("text-lg font-semibold")
    ui.label(
        "One row per interaction. Summary fields describe its latest refinement and execution runs. "
        "candidate_count includes all saved attempts in the interaction. Revisions are not additional trials. "
        "Recorded validation does not establish assembly success."
    ).classes("text-sm text-slate-600")
    rows: list[dict[str, Any]] = []
    with ui.row().classes("w-full items-center"):
        search = ui.input("Search saved results").classes("grow")
        refresh_button = ui.button("Refresh saved results", icon="refresh")
        export_button = ui.button("Export CSV", icon="download")
    message = ui.label().classes("text-slate-600")
    table = ui.table(
        columns=[
            {"name": field, "field": field, "label": field, "sortable": True}
            for field in RESULT_FIELDS
        ],
        rows=[],
        row_key="interaction_identifier",
        selection="single",
        pagination=10,
    ).classes("w-full")
    details = ui.column().classes("w-full")

    def filtered() -> list[dict[str, Any]]:
        query = search.value or ""
        return [
            row
            for row in rows
            if query.casefold()
            in " ".join(str(row.get(field, "")) for field in RESULT_FIELDS).casefold()
        ]

    def update() -> None:
        table.rows = filtered()
        table.selected = []
        details.clear()

    def refresh() -> None:
        rows.clear()
        try:
            rows.extend(read_interactions(contexts_root))
            message.text = "" if rows else "No saved interactions are available."
        except OSError as exc:
            message.text = f"Saved interactions are unavailable: {exc}"
        update()

    def select() -> None:
        details.clear()
        if not table.selected:
            return
        identifier = table.selected[0]["interaction_identifier"]
        root = contexts_root / identifier
        with details:
            ui.label(identifier).classes("font-semibold")
            paths = artifact_paths(root)
            if not paths:
                ui.label("No saved artifacts are available.")
                return
            ui.label(
                "Inspect request records for the saved setup; candidate.json for primitive_steps; "
                "validation and binding records for findings; event records for timings; "
                "execution records for observed command outcomes."
            ).classes("text-sm text-slate-600")
            artifact = ui.select(
                paths, label="Saved artifact", value=paths[0], with_input=True
            ).classes("w-full")
            content = ui.code(language="json").classes("w-full")

            def show() -> None:
                content.content = json.dumps(
                    read_artifact(root, artifact.value), indent=2, ensure_ascii=False
                )

            artifact.on_value_change(show)
            show()

    search.on_value_change(update)
    refresh_button.on_click(refresh)
    export_button.on_click(
        lambda: ui.download.content(
            results_csv(filtered()), "spec2primitives-results.csv", "text/csv"
        )
    )
    table.on_select(select)
    refresh()
