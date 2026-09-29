"""Read recovery evidence and archive only on an explicit operator action."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from cais_spade_llm.recovery_framework.delivery import RUN_DIRECTORY
from cais_spade_llm.recovery_framework.reports import archive_latest
from cais_spade_llm.ui.evidence import BackgroundSection, TablePager, lazy_file, read_json_cached


def read_report(path: Path, *, environment: bool) -> dict:
    """Read report metadata and transitions outside the UI thread."""
    report = read_json_cached(path)
    if not isinstance(report, dict):
        raise ValueError("Expected a JSON object")
    required = {"outcome", "models", "transitions"} if environment else {"outcome", "transitions"}
    if not required <= report.keys():
        raise ValueError("Unsupported report format")
    if not environment and (
        report.get("schema_version") != 1 or report.get("evidence") != "gazebo"
    ):
        raise ValueError("Unsupported Gazebo report format")
    rows = (
        [
            {
                "revision": record.get("revision"),
                "task": (record.get("task") or {}).get("event_name"),
                "resource": (record.get("task") or {}).get("resource_id"),
                "part": ((record.get("acknowledgement") or {}).get("observations") or {}).get(
                    "part_name"
                ),
            }
            for record in report.get("transitions", [])
        ]
        if not environment
        else []
    )
    return {"run_id": report.get("run_id"), "outcome": report["outcome"], "rows": rows}


def render_gazebo_delivery_runs(
    directory: Path = RUN_DIRECTORY, *, environment: bool = False
) -> None:
    """Defer report reads and show bounded recorded observations on demand."""
    title = "Environmental runs" if environment else "Gazebo delivery runs"
    with ui.expansion(title, icon="precision_manufacturing").classes("w-full") as expansion:
        ui.label(
            f"Recorded {'environmental' if environment else 'Gazebo'} evidence. Selecting a report performs reads only."
        ).classes("text-sm")
        selector = ui.select({}, label="Saved Gazebo run").classes("w-full")
        index = BackgroundSection()
        content = BackgroundSection()
        displayed_run = {"run_id": None}

        def render_report(report: dict) -> None:
            displayed_run["run_id"] = report.get("run_id")
            outcome = report.get("outcome") or {}
            ui.label(f"Outcome: {outcome.get('status', 'not recorded')}").classes("font-semibold")
            if outcome.get("reason"):
                ui.label(str(outcome["reason"])).classes("text-sm")
            rows = report["rows"]
            table = ui.table(
                columns=[
                    {"name": key, "label": key, "field": key}
                    for key in ("revision", "task", "resource", "part")
                ],
                rows=[],
                row_key="revision",
            ).classes("w-full")
            TablePager(table, lambda: rows).update()
            lazy_file("Recorded setup and observations", Path(selector.value))

        def display() -> None:
            displayed_run["run_id"] = None
            if selector.value:
                path = Path(selector.value)
                content.load(lambda: read_report(path, environment=environment), render_report)
            else:
                content.cancel()
                content.container.clear()

        def indexed(paths: list[Path]) -> None:
            options = {str(path): path.parent.name for path in paths}
            value = selector.value if selector.value in options else None
            latest = str(directory / "latest/run.json")
            value = value or (latest if latest in options else next(iter(options), None))
            previous = selector.value
            selector.set_options(options, value=value)
            if value == previous:
                display()
            if not paths:
                ui.label("No saved runs are available.").classes("text-sm text-slate-500")

        def refresh() -> None:
            index.load(
                lambda: sorted(directory.glob("*/run.json"))
                + sorted(directory.glob("archive/*/run.json")),
                indexed,
            )

        def archive() -> None:
            try:
                if (
                    selector.value != str(directory / "latest/run.json")
                    or not displayed_run["run_id"]
                ):
                    raise ValueError("Select the latest run before archiving")
                path = archive_latest(directory, expected_run_id=displayed_run["run_id"])
                ui.notify(f"Archived {path.parent.name}")
                refresh()
            except (OSError, ValueError) as exc:
                ui.notify(str(exc), type="warning")

        selector.on_value_change(display)
        ui.button("Refresh Gazebo reports", on_click=refresh, icon="refresh")
        ui.button("Archive this run", on_click=archive, icon="archive")
        opened = False

        def show() -> None:
            nonlocal opened
            if expansion.value and not opened:
                opened = True
                refresh()

        expansion.on_value_change(show)
