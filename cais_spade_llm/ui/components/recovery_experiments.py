"""Experiments tab: run the Study 1 ablation on the retained captures and read its audited results.

The tab talks only to the recovery-framework batch runner and its saved records; it never reaches the live
runtime. Paid LLM calls start only when the operator starts a batch.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path

from nicegui import ui

from cais_spade_llm.recovery_framework import outline_experiments as experiments
from cais_spade_llm.recovery_framework.outline_audit import AUDIT_NAME, OUTCOME_CLASSES
from cais_spade_llm.ui.components.part_slippage_recovery import draft_rows
from cais_spade_llm.ui.evidence import BackgroundSection, TablePager, lazy_file
from cais_spade_llm.ui.refresh import PageRefresh

# "not sent" is only for models without a reasoning setting, such as gpt-4o.
_EFFORTS = {**{name: name for name in ("none", "minimal", "low", "medium", "high", "xhigh")}, "not sent (gpt-4o)": None}
_CHECK_NAMES = {"check_ra_feasibility": "RA", "check_cca_safety": "CCA safety", "check_nonblocking": "CCA nonblocking"}
_ATTEMPT_COLUMNS = (
    "attempt_id", "scenario", "configuration", "repetition", "status", "outcome", "first_failing_event",
    "failing_stage", "outline_length", *(f"rejections_{check}" for check in experiments.REJECTION_CHECKS),
    "revisions", "rewrites", "lookups", "provider_calls", "tokens", "elapsed_seconds", "stop_cause",
)
_OUTCOME_COLUMNS = ("configuration", "scenario", "started", *OUTCOME_CLASSES, "not audited", "median tokens",
                    "median seconds", "median lookups")
_REJECTION_COLUMNS = ("configuration", "scenario", *experiments.REJECTION_CHECKS, "revisions or rewrites")
_EVENT_COLUMNS = ("step", "event_name", "resource", "part_name", "end_state", "rationale")


def _columns(names: tuple[str, ...]) -> list[dict]:
    return [{"name": name, "field": name, "label": name.replace("_", " "), "align": "left"} for name in names]


def _configuration_text(flags: dict) -> str:
    generation = "whole outline in one answer" if flags.get("single_pass_outline") else "one event per turn"
    checks = [text for key, text in _CHECK_NAMES.items() if flags.get(key)]
    rules = {"knowledge_base": "; safety rules as a lookup", "prompt": "; safety rules in the prompt"}.get(
        flags.get("safety_rules_to_llm"), "")
    return f"{generation}; " + (", ".join(checks) + " checks" if checks else "no checks") + rules


def _named(value: object, names: dict[str, str]) -> object:
    if not isinstance(value, str):
        return value
    for jid, name in names.items():
        value = value.replace(jid, name)
    return value


def render_recovery_experiments(*, is_active: Callable[[], bool] | None = None,  # noqa: C901, PLR0915 - one tab's controls
                                runner: experiments.OutlineExperiments | None = None) -> None:
    """Render the Study 1 setup, run controls and results.

    Args:
        is_active: Whether the Experiments tab is visible; polling pauses otherwise.
        runner: The batch runner; this process's shared runner by default.
    """
    from cais_spade_llm.recovery_framework.part_slippage_example import default_observation_paths

    runner = runner or experiments.outline_experiments()
    captures = default_observation_paths()
    defaults = experiments.DEFAULT_SETTINGS
    shown: dict = {"batch": None, "rows": [], "revision": None}

    with ui.card().classes("w-full gap-2"):
        ui.label("Study 1: ablation").classes("text-lg font-semibold")
        ui.label("Each attempt is a fresh outline draft on a retained capture; every attempt is then audited "
                 "with the RA and CCA checks. Paid LLM calls start only when you start a batch, and every record "
                 "is kept, failed attempts included.").classes("text-sm text-slate-600")
        ui.label("Failure scenarios").classes("font-semibold")
        scenarios = {name: ui.checkbox(name, value=name == "Part slippage") for name in captures}
        if not captures:
            ui.label("No retained capture is available.").classes("text-red-700")
        ui.label("Attempts per scenario").classes("font-semibold")
        attempts = {}
        for name, flags in experiments.CONFIGURATIONS.items():
            with ui.row().classes("items-center gap-3"):
                attempts[name] = ui.number(name, value=defaults["attempts"][name], min=0,
                                           max=experiments.MAX_ATTEMPTS, step=1, format="%d").classes("w-56")
                ui.label(_configuration_text(flags)).classes("text-sm text-slate-600")
        with ui.row().classes("items-end gap-3"):
            model = ui.input("Model", value=defaults["model"])
            effort = ui.select(list(_EFFORTS), value=defaults["reasoning_effort"], label="Reasoning effort")
            max_turns = ui.number("Max turns per attempt", value=defaults["max_turns"], min=1,
                                  max=experiments.MAX_TURNS, step=1, format="%d")
        planned = ui.label().classes("text-sm")
        with ui.row().classes("gap-2"):
            start_button = ui.button("Start batch", icon="play_arrow")
            stop_button = ui.button("Stop", icon="stop").props("outline")
        ui.label("Run one batch at a time and avoid Recovery tab drafts meanwhile: they share the laptop's memory "
                 "and the provider's rate limit. Changing code or a capture stops a running batch.").classes(
            "text-xs text-slate-500")

    with ui.card().classes("w-full gap-1"):
        ui.label("Progress").classes("font-semibold")
        progress = ui.label().classes("text-sm")
        progress_detail = ui.label().classes("text-sm text-slate-600")

    with ui.card().classes("w-full gap-2"):
        ui.label("Results").classes("text-lg font-semibold")
        with ui.row().classes("w-full items-end gap-2"):
            batch_select = ui.select({}, label="Batch").classes("grow")
            refresh_button = ui.button("Refresh", icon="refresh").props("flat")
            csv_button = ui.button("Export CSV", icon="download").props("outline")
            record_button = ui.button("Download batch record", icon="download").props("outline")
            audit_button = ui.button("Audit missing", icon="fact_check").props("outline")
        batch_message = ui.label().classes("text-sm text-slate-600")
        ui.label("Audited outcome by configuration (Fig. 11)").classes("font-semibold")
        outcome_table = ui.table(columns=_columns(_OUTCOME_COLUMNS), rows=[], row_key="key").classes(
            "w-full").props("dense wrap-cells")
        ui.label("Rejections during generation (Table 6)").classes("font-semibold")
        rejection_table = ui.table(columns=_columns(_REJECTION_COLUMNS), rows=[], row_key="key").classes(
            "w-full").props("dense wrap-cells")
        ui.label("Attempts").classes("font-semibold")
        attempt_table = ui.table(columns=_columns(_ATTEMPT_COLUMNS), rows=[], row_key="attempt_id",
                                 selection="single").classes("w-full").props("dense wrap-cells")
        pager = TablePager(attempt_table, lambda: [{column: row.get(column) for column in _ATTEMPT_COLUMNS}
                                                   for row in shown["rows"]])
        details = ui.column().classes("w-full")

    def settings() -> dict:
        return {"scenarios": [name for name, box in scenarios.items() if box.value],
                "attempts": {name: int(field.value or 0) for name, field in attempts.items()},
                "model": (model.value or "").strip(), "reasoning_effort": _EFFORTS[effort.value],
                "max_turns": int(max_turns.value or 0)}

    def show_planned() -> None:
        chosen = settings()
        count, per_scenario = len(chosen["scenarios"]), sum(chosen["attempts"].values())
        planned.text = (f"Planned: {count * per_scenario} attempts ({count} scenario{'' if count == 1 else 's'} × "
                        f"{per_scenario} attempts per scenario).")

    def show_progress() -> None:
        state = runner.snapshot()
        start_button.set_enabled(not state["running"] and bool(captures))
        stop_button.set_enabled(state["running"])
        audit_button.set_enabled(not state["running"] and shown["batch"] is not None)
        current = state["current"]
        if not state["running"]:
            progress.text = state["message"] or "No batch is running."
            progress_detail.text = ""
        else:
            progress.text = f"{state['batch_id']}: {state['done']} of {state['planned']} attempts recorded."
            progress_detail.text = "" if not current else (
                f"{current.get('attempt_id')} · {current.get('scenario')} · {current.get('configuration')} · "
                f"repetition {current.get('repetition')} · {current.get('stage')} · "
                f"{current.get('provider_calls') or 0} provider calls · "
                f"{time.time() - current.get('started', time.time()):.0f} s · {current.get('draft_message') or ''}")

    async def show_batch() -> None:
        details.clear()
        batch_id = batch_select.value
        if not batch_id:
            shown.update(batch=None, rows=[])
            batch_message.text = "No batch has run yet."
            outcome_table.rows, rejection_table.rows = [], []
            pager.update()
            return
        batch = await asyncio.to_thread(experiments.read_batch, runner.root / batch_id / experiments.BATCH_NAME,
                                        live=runner.live_batch)
        shown.update(batch=batch, rows=batch.get("rows") or [])
        summary = experiments.summarize(shown["rows"])
        outcome_table.rows = [{"key": f"{row['configuration']}|{row['scenario']}", **row} for row in summary["outcomes"]]
        rejection_table.rows = [{"key": f"{row['configuration']}|{row['scenario']}", **row}
                                for row in summary["rejections"]]
        chosen = batch.get("settings") or {}
        batch_message.text = (f"{batch.get('status')}: {batch.get('message') or ''} Model {chosen.get('model')}, "
                              f"effort {chosen.get('reasoning_effort')}, max turns {chosen.get('max_turns')}, "
                              f"scheduling seed {batch.get('seed')}.")
        pager.update(reset=False)

    async def refresh_batches(*, select_live: bool = False) -> None:
        batches = await asyncio.to_thread(experiments.list_batches, runner.root, live=runner.live_batch)
        options = {row["batch_id"]: f"{row['batch_id']} · {row['status']} · {row['done']}/{row['planned']}"
                   for row in batches}
        live = runner.snapshot()["batch_id"]
        if select_live and live in options:
            value = live
        else:
            value = batch_select.value if batch_select.value in options else next(iter(options), None)
        batch_select.set_options(options, value=value)
        await show_batch()

    async def start() -> None:
        start_button.set_enabled(False)
        try:
            await runner.start(settings())
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            show_progress()
            return
        ui.notify("Batch started.", type="positive")
        await refresh_batches(select_live=True)

    def stop() -> None:
        try:
            runner.stop()
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
        show_progress()

    async def audit_missing() -> None:
        if shown["batch"] is None:
            return
        audit_button.set_enabled(False)
        try:
            await runner.audit_missing(shown["batch"]["batch_id"])
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            show_progress()
            return
        ui.notify("Auditing the attempts that have no audit.", type="info")

    def export_csv() -> None:
        if shown["batch"] is not None:
            ui.download.content(experiments.rows_csv(shown["rows"]), f"{shown['batch']['batch_id']}.csv", "text/csv")

    def download_record() -> None:
        if shown["batch"] is not None:
            ui.download.file(runner.root / shown["batch"]["batch_id"] / experiments.BATCH_NAME)

    def select_attempt() -> None:
        details.clear()
        if not attempt_table.selected or shown["batch"] is None:
            return
        row = next((row for row in shown["rows"] if row["attempt_id"] == attempt_table.selected[0]["attempt_id"]), {})
        path = Path(str(row.get("draft_path")))
        names = (shown["batch"].get("resource_names") or {}).get(row.get("scenario"), {})
        with details:
            ui.label(f"{row.get('attempt_id')}: {row.get('configuration')} on {row.get('scenario')}, "
                     f"repetition {row.get('repetition')}; {row.get('outcome')}").classes("font-semibold")
            ui.label(str(row.get("message") or "")).classes("text-sm text-slate-600")
            if not path.is_file():
                ui.label("No draft was recorded for this attempt.").classes("text-sm")
                return
            audit_path = path.parent / AUDIT_NAME

            def read() -> tuple[dict, dict | None]:
                audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
                return json.loads(path.read_text(encoding="utf-8")), audit

            def render(loaded: tuple[dict, dict | None]) -> None:
                draft, audit = loaded
                if audit is not None:
                    where = (f"event {audit['first_failing_event']}" if audit.get("first_failing_event")
                             else "the outline's end" if audit.get("failing_stage") else "no event")
                    ui.label(f"Audit: {audit.get('outcome', audit.get('status'))}, first failing check "
                             f"{audit.get('failing_stage') or 'none'} at {where}.").classes("text-sm")
                    for finding in (audit.get("findings") or [])[:5]:
                        ui.label(f"• {_named(finding.get('reason'), names)}").classes("text-sm text-slate-600")
                rows = [{key: _named(value, names) for key, value in event.items()} for event in draft_rows(draft)]
                ui.table(columns=_columns(_EVENT_COLUMNS), rows=rows, row_key="step").classes("w-full").props(
                    "dense wrap-cells")
                lazy_file("Draft record (draft.json)", path)
                if audit_path.is_file():
                    lazy_file("Audit record (audit.json)", audit_path)

            BackgroundSection().load(read, render)

    async def poll() -> None:
        show_progress()
        state = runner.snapshot()
        if state["revision"] != shown["revision"]:
            # A batch started, stopped, or recorded an attempt: refresh the tables once, not on every tick.
            shown["revision"] = state["revision"]
            await refresh_batches(select_live=state["running"])

    for field in (*scenarios.values(), *attempts.values()):
        field.on_value_change(show_planned)
    start_button.on_click(start)
    stop_button.on_click(stop)
    refresh_button.on_click(lambda: refresh_batches())
    csv_button.on_click(export_csv)
    record_button.on_click(download_record)
    audit_button.on_click(audit_missing)
    batch_select.on_value_change(show_batch)
    attempt_table.on_select(select_attempt)
    show_planned()
    show_progress()
    PageRefresh(is_active).timer(2, poll)
