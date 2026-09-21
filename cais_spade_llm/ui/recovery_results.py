"""Read saved recovery evidence without loading proposals into the runtime."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

NOT_RECORDED = "not recorded"
RECOVERY_RESULTS_ROOT = (
    Path(__file__).resolve().parents[1]
    / "agents/intelligent_product/replanner/llm_recovery/runtime_data"
)
RESULT_FIELDS = (
    "session_id",
    "status",
    "phase",
    "turn_index",
    "recovery_selection_mode",
    "final_output_stage",
    "primitive_program_complete",
    "elapsed_sec",
    "execution_mode",
    "execution_status",
    "physical_outcome",
    "source",
    "record_error",
)


def read_artifact(root: Path, relative_path: str) -> dict[str, Any]:
    """Read one JSON artifact contained by the selected evidence directory."""
    path = root / relative_path
    try:
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


def _sections(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sections = [payload]
    for section in sections:
        for name in (
            "session_state",
            "current_turn",
            "multi_turn_session",
            "multi_turn_session_result",
            "multi_turn_current_turn",
            "recovery_debug",
            "prepared_recovery_request",
            "multi_turn_session_state",
            "recovery_session",
        ):
            child = section.get(name)
            if isinstance(child, dict):
                sections.append(child)
    return sections


def _value(sections: list[dict[str, Any]], *keys: str) -> Any:
    for key in keys:
        for section in sections:
            value = section.get(key)
            if value is not None and value != "":
                return value
    return NOT_RECORDED


def read_sessions(root: Path = RECOVERY_RESULTS_ROOT) -> list[dict[str, Any]]:
    """Group recorded session identifiers, retaining failed and incomplete evidence.

    Artifacts without a recorded session identifier stay separate. They are not
    inferred to be independent trials or assigned to a session by filename.
    """
    paths = sorted(
        {
            path
            for pattern in ("*.json", "*.txt")
            for path in root.rglob(pattern)
            if path.is_file()
            and any(
                token in path.name
                for token in (
                    "_result",
                    "_response",
                    "_audit",
                    "_stack",
                    "_session_",
                    "_checkpoint",
                )
            )
            and "_llm_response_" not in path.name
        },
        key=lambda path: ("_latest." in path.name, str(path)),
    )
    groups: dict[str, dict[str, Any]] = {}
    copies: dict[tuple[str, str, str], str] = {}
    for path in paths:
        relative = str(path.relative_to(root))
        payload = read_artifact(root, relative)
        sections = _sections(payload)
        session_id = _value(sections, "session_id", "recovery_session_id")
        if not isinstance(session_id, str):
            session_id = NOT_RECORDED
        key = session_id if session_id != NOT_RECORDED else relative
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        copy_name = re.sub(r"_(?:latest|\d{8}T\d{6})(?=\.)", "", path.name)
        copy_key = (str(path.parent), copy_name, digest)
        if session_id == NOT_RECORDED and "_latest." in path.name and copy_key in copies:
            key = copies[copy_key]
        copies[copy_key] = key
        row = groups.setdefault(
            key,
            {
                **dict.fromkeys(RESULT_FIELDS, NOT_RECORDED),
                "session_id": session_id,
                "source": relative,
                "artifacts": [],
                "_rank": (-1, -1),
            },
        )
        row["artifacts"].append(relative)
        turn_index = _value(sections, "turn_index")
        turn_rank = turn_index if isinstance(turn_index, int) else -1
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            modified = -1
        rank = (turn_rank, modified)
        if rank >= row["_rank"]:
            row["_rank"] = rank
            row["source"] = relative
            for field in RESULT_FIELDS:
                if field not in {"source", "session_id"}:
                    row[field] = _value(sections, field)
            row["status"] = _value(sections, "status", "decision", "selection_status")
            row["phase"] = _value(sections, "current_phase", "phase")
    return sorted(groups.values(), key=lambda row: row["_rank"], reverse=True)


def results_csv(rows: list[dict[str, Any]]) -> str:
    """Export exactly the displayed measurements and their source references."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, NOT_RECORDED) for field in RESULT_FIELDS})
    return output.getvalue()


def render_results(root: Path = RECOVERY_RESULTS_ROOT) -> None:
    """Render a searchable, read-only recovery evidence browser."""
    from nicegui import ui

    ui.label("Saved recovery evidence").classes("text-lg font-semibold")
    ui.label(
        "Validation, simulation execution, and physical outcomes are separate. "
        "Artifacts without session_id remain separate; rows are not a trial count."
    ).classes("text-sm text-slate-600")
    ui.label(
        "Gazebo execution must be recorded in the evidence. Inspect each artifact for its "
        "recorded configuration; missing settings remain not recorded. The current setup "
        "never fills gaps in historical results."
    ).classes("text-sm text-slate-600")
    records: dict[str, dict[str, Any]] = {}
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
        row_key="source",
        selection="single",
        pagination=10,
    ).classes("w-full")
    details = ui.column().classes("w-full")

    def filtered() -> list[dict[str, Any]]:
        query = search.value or ""
        return [
            row
            for row in records.values()
            if query.casefold()
            in " ".join(str(row.get(field, "")) for field in RESULT_FIELDS).casefold()
        ]

    def update() -> None:
        table.rows = [{field: row[field] for field in RESULT_FIELDS} for row in filtered()]
        table.selected = []
        details.clear()

    def refresh() -> None:
        records.clear()
        try:
            records.update((row["source"], row) for row in read_sessions(root))
            message.text = "" if records else "No saved recovery evidence is available."
        except OSError as exc:
            message.text = f"Saved recovery evidence is unavailable: {exc}"
        update()

    def select() -> None:
        details.clear()
        if not table.selected:
            return
        row = records[table.selected[0]["source"]]
        with details:
            ui.label(f"session_id: {row['session_id']}").classes("font-semibold")
            artifact = ui.select(
                row["artifacts"], label="Saved artifact", value=row["source"]
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
            results_csv(filtered()), "recovery-framework-results.csv", "text/csv"
        )
    )
    table.on_select(select)
    refresh()
