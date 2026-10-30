"""Write the RCIM results page (Section 6.3 figures and tables) from the Study 1 batch records.

The page has the layout of the figure mock-ups: Fig. 11, Table 6, Fig. 12 (outline rows) and Table 7 for the
main model, Fig. 13 for both models, and every attempt. With ``--placeholder`` it is filled with layout numbers
instead, so the page can be reviewed before any batch has run.

Usage, from the repository root:
    poetry run python "writing/Journal Paper 2 (recovery framework)/figures/results_page/results_page.py" --placeholder
    poetry run python "writing/Journal Paper 2 (recovery framework)/figures/results_page/results_page.py" [--batch ID ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

from cais_spade_llm.recovery_framework.outline_audit import OUTCOME_CLASSES  # noqa: E402
from cais_spade_llm.recovery_framework.outline_experiments import (  # noqa: E402
    BATCH_NAME,
    CONFIGURATIONS,
    EXPERIMENTS,
    read_batch,
)

logger = logging.getLogger(__name__)

TEMPLATE = HERE / "results_template.html"
OUTPUT = HERE / "results.html"
# The main model gives every Section 6.3.1 result; the second model appears only in the model comparison.
# planned: attempts per setting and scenario.
MODELS = [
    {"id": "opus", "model": "claude-opus-4-7", "name": "Claude Opus 4.7 medium", "role": "main model",
     "vendor": "Claude", "color": "var(--s3)", "planned": 20},
    {"id": "gpt54", "model": "gpt-5.4", "name": "gpt-5.4 medium", "role": "second model",
     "vendor": "OpenAI", "color": "var(--s1)", "planned": 10},
]
EFFORT = "medium"
MAX_TURNS = 40
SETTING_NOTES = ["one answer", "one answer", "one event per turn", "one event per turn", "whole-outline rewrites",
                 "one-event revisions"]
SCENARIOS = ["Conveyor breakdown", "ur5e-1 breakdown", "Machining breakdown during part processing", "Part slippage"]
SCENARIO_SHORT = ["Conveyor", "ur5e-1", "Machining", "Part slippage"]
FW_SINGLE, FW_INCREMENTAL = 4, 5
# The audit's own outcome names, then attempts whose audit is missing or failed.
NOT_AUDITED = "not audited"
CLASSES = [*OUTCOME_CLASSES, NOT_AUDITED]
# Lookup references in the order the outline prompt lists them, with the name or key left out.
REFERENCES = [
    "/resources/<name>/state_variables", "/resources/<name>/snapshot", "/resources/<name>/static_capabilities",
    "/resources/<name>/pose_relations", "/resources/<name>/pending_tasks", "/parts/<name>/state_variables",
    "/parts/<name>/snapshot", "/parts/<name>/pose_relations", "/parts/<name>/target", "/geometry",
    "/geometry/<key>", "/regions", "/regions/<name>", "/safety_rules",
]
STARTED = {"finished", "failed", "interrupted"}


def reference_path(reference: str) -> str:
    """Return a lookup reference as the outline prompt lists it, with the resource, part, region or key left out."""
    segments = str(reference).strip("/").split("/")
    if len(segments) >= 2 and segments[0] in ("resources", "parts", "regions"):
        segments[1] = "<name>"
    elif len(segments) >= 2 and segments[0] == "geometry":
        segments[1] = "<key>"
    return "/" + "/".join(segments)


def setting_index(flags: dict) -> int | None:
    """Return the setting whose OutlineDraft flags these are, or None for a setting no longer in the study."""
    return next((index for index, expected in enumerate(CONFIGURATIONS.values()) if expected == flags), None)


def _median(values: list[float]) -> float | None:
    numbers = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return statistics.median(numbers) if numbers else None


def _round(value: float | None, digits: int = 0) -> float | None:
    if value is None:
        return None
    return round(value, digits) if digits else round(value)


def _number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def empty_cell() -> dict:
    return {"n": 0, "classes": [0] * len(CLASSES), "tokens_k": None, "minutes": None, "lookups": None,
            "completed": 0, "rejections": None, "redo": None, "proposals": None, "references": None}


def _draft_details(row: dict) -> dict:
    """Read what the batch row does not hold: checked candidate events and lookups by reference."""
    path = Path(str(row.get("draft_path") or ""))
    if not path.is_file():
        return {"proposals": None, "references": None}
    record = json.loads(path.read_text(encoding="utf-8"))
    references: dict[str, int] = {}
    for reference in record.get("retrievals") or []:
        references[reference_path(reference)] = references.get(reference_path(reference), 0) + 1
    proposals = None
    if record.get("ra_feasibility_check") and not record.get("single_pass_outline"):
        proposals = sum(len(turn.get("results") or []) for turn in record.get("ra_feasibility_checks") or [])
    return {"proposals": proposals, "references": references}


def attempts_from_batches(batch_ids: list[str] | None) -> tuple[list[dict], list[dict]]:
    """Return the started attempts of the selected batches and one source line per batch.

    Batches are selected by id, or else every batch of a study model at the study effort and turn budget.
    """
    models = {model["model"]: model["id"] for model in MODELS}
    attempts, sources = [], []
    for path in sorted(EXPERIMENTS.glob(f"*/{BATCH_NAME}")):
        batch = read_batch(path)
        settings = batch.get("settings") or {}
        if batch_ids and batch["batch_id"] not in batch_ids:
            continue
        if not batch_ids and (settings.get("model") not in models or settings.get("reasoning_effort") != EFFORT
                              or settings.get("max_turns") != MAX_TURNS):
            logger.info("Left out %s: %s %s, %s turns", batch["batch_id"], settings.get("model"),
                        settings.get("reasoning_effort"), settings.get("max_turns"))
            continue
        if settings.get("model") not in models:
            logger.warning("Left out %s: %s is not a study model", batch["batch_id"], settings.get("model"))
            continue
        started = 0
        for row in batch.get("rows") or []:
            if row.get("status") not in STARTED:
                continue
            setting = setting_index((batch.get("configurations") or {}).get(row.get("configuration"), {}))
            if setting is None:
                logger.info("Left out %s %s: %s is not a current setting", batch["batch_id"], row.get("attempt_id"),
                            row.get("configuration"))
                continue
            started += 1
            attempts.append({"model": models[settings["model"]], "setting": setting, "scenario": row.get("scenario"),
                             "batch": batch["batch_id"], "row": row, **_draft_details(row)})
        git = batch.get("git") or {}
        sources.append({"batch": batch["batch_id"], "model": settings.get("model"), "status": batch.get("status"),
                        "started": started, "planned": len(batch.get("plan") or []),
                        "created": str(batch.get("created_at", ""))[:16].replace("T", " "),
                        "code": f"{str(git.get('head', '?'))[:8]}, {git.get('uncommitted_files', '?')} uncommitted files"})
    return attempts, sources


def cells_from_attempts(attempts: list[dict]) -> dict:
    """Return one cell per model, setting and scenario: counts by class, medians and check totals."""
    cells = {model["id"]: [{scenario: empty_cell() for scenario in SCENARIOS} for _ in CONFIGURATIONS]
             for model in MODELS}
    grouped: dict[tuple, list[dict]] = {}
    for attempt in attempts:
        if attempt["scenario"] in SCENARIOS:
            grouped.setdefault((attempt["model"], attempt["setting"], attempt["scenario"]), []).append(attempt)
    for (model, setting, scenario), group in grouped.items():
        rows = [attempt["row"] for attempt in group]
        cell = cells[model][setting][scenario]
        cell["n"] = len(rows)
        for row in rows:
            outcome = row.get("outcome")
            cell["classes"][CLASSES.index(outcome) if outcome in OUTCOME_CLASSES else CLASSES.index(NOT_AUDITED)] += 1
        cell["completed"] = sum(row.get("draft_status") == "draft" for row in rows)
        tokens = _median([_number(row.get("tokens")) for row in rows])
        seconds = _median([row.get("elapsed_seconds") for row in rows])
        cell["tokens_k"] = _round(tokens / 1000 if tokens is not None else None)
        cell["minutes"] = _round(seconds / 60 if seconds is not None else None, 1)
        cell["lookups"] = _round(_median([_number(row.get("lookups")) for row in rows]))
        if setting in (FW_SINGLE, FW_INCREMENTAL):
            total = lambda key: sum(_number(row.get(key)) or 0 for row in rows)  # noqa: E731
            cell["rejections"] = {"ra": total("rejections_ra_feasibility"),
                                  "physical": total("rejections_physical_evidence"),
                                  "cca": total("rejections_cca_safety"), "eq6": total("rejections_nonblocking")}
            cell["redo"] = _round(_median([_number(row.get("rewrites" if setting == FW_SINGLE else "revisions"))
                                           for row in rows]), 1)
            proposals = [attempt["proposals"] for attempt in group if attempt["proposals"] is not None]
            cell["proposals"] = sum(proposals) if proposals else None
        counted = [attempt["references"] for attempt in group if attempt["references"] is not None]
        if counted:
            paths = {path for references in counted for path in references}
            cell["references"] = {path: round(sum(references.get(path, 0) for references in counted) / len(counted), 1)
                                  for path in paths}
    return cells


def attempt_lines(attempts: list[dict]) -> list[dict]:
    """Return one line per attempt for the page's attempt table."""
    names = list(CONFIGURATIONS)
    lines = []
    for attempt in attempts:
        row = attempt["row"]
        tokens, seconds = _number(row.get("tokens")), row.get("elapsed_seconds")
        lines.append({
            "model": attempt["model"], "setting": f"{attempt['setting'] + 1} · {names[attempt['setting']]}",
            "scenario": attempt["scenario"], "outcome": row.get("outcome"),
            "stage": row.get("failing_stage") or "", "stop": row.get("stop_cause"),
            "events": row.get("outline_length"), "lookups": row.get("lookups"),
            "tokens_k": round(tokens / 1000) if tokens is not None else None,
            "minutes": round(seconds / 60, 1) if isinstance(seconds, (int, float)) else None,
            "attempt": f"{attempt['batch']} · {row.get('attempt_id')}",
        })
    order = {scenario: index for index, scenario in enumerate(SCENARIOS)}
    model_order = {model["id"]: index for index, model in enumerate(MODELS)}
    return sorted(lines, key=lambda line: (model_order[line["model"]], line["setting"], order.get(line["scenario"], 9)))


# ---------------------------------------------------------------- placeholder numbers (layout only)

# Main model, N = 20 per cell: attempts per class in CLASSES order (no "not audited"), per scenario and setting.
_PLACEHOLDER_OUTCOME = {
    "Part slippage": [[6, 6, 5, 2, 1], [9, 6, 1, 3, 1], [9, 5, 4, 2, 0], [12, 5, 1, 2, 0], [18, 0, 0, 0, 2], [17, 0, 0, 0, 3]],
    "Conveyor breakdown": [[8, 7, 2, 2, 1], [9, 7, 1, 2, 1], [10, 6, 1, 2, 1], [11, 6, 0, 2, 1], [18, 0, 0, 0, 2], [18, 0, 0, 0, 2]],
    "Machining breakdown during part processing": [[0, 9, 4, 6, 1], [1, 10, 2, 6, 1], [2, 8, 3, 5, 2], [3, 9, 1, 5, 2], [9, 0, 0, 0, 11], [11, 0, 0, 0, 9]],
    "ur5e-1 breakdown": [[4, 10, 3, 2, 1], [5, 11, 1, 2, 1], [7, 8, 2, 3, 0], [8, 9, 1, 2, 0], [15, 0, 0, 0, 5], [13, 0, 0, 0, 7]],
}
# Main model: median tokens (k), minutes and lookups per attempt.
_PLACEHOLDER_EFFORT = {
    "Part slippage": [[15, 0.5, 6], [17, 0.6, 6], [43, 1.4, 9], [47, 1.6, 9], [34, 1.9, 7], [263, 8.1, 12]],
    "Conveyor breakdown": [[21, 0.8, 8], [23, 0.9, 8], [64, 2.3, 12], [69, 2.5, 12], [58, 3.4, 9], [310, 12, 15]],
    "Machining breakdown during part processing": [[30, 1.1, 7], [33, 1.3, 7], [96, 3.8, 11], [101, 4.1, 11], [118, 7.6, 10], [520, 24, 16]],
    "ur5e-1 breakdown": [[25, 0.9, 7], [27, 1.0, 7], [78, 3.0, 10], [83, 3.2, 10], [86, 5.2, 9], [420, 19, 14]],
}
# Second model, N = 10 per cell: attempts passed per setting.
_PLACEHOLDER_SECOND_PASSED = {
    "Part slippage": [4, 5, 5, 6, 9, 9],
    "Conveyor breakdown": [4, 5, 5, 6, 8, 9],
    "Machining breakdown during part processing": [0, 1, 1, 2, 4, 5],
    "ur5e-1 breakdown": [2, 3, 4, 4, 6, 7],
}
# Framework settings, main model: (median rewrites or revisions, RA, RA physical evidence, CCA safety, CCA
# completion, checked events) for settings 5 and 6.
_PLACEHOLDER_CHECKS = {
    "Part slippage": [(1, 5, 4, 7, 3, None), (3, 25, 16, 18, 6, 300)],
    "Conveyor breakdown": [(1, 7, 5, 2, 2, None), (3, 29, 19, 6, 3, 290)],
    "Machining breakdown during part processing": [(3, 23, 15, 9, 11, None), (9, 94, 62, 31, 14, 560)],
    "ur5e-1 breakdown": [(2, 13, 8, 5, 4, None), (6, 58, 39, 22, 8, 410)],
}
# Main model, setting 6: mean outline lookups per attempt by reference.
_PLACEHOLDER_REFERENCES = {
    "Part slippage": {"/resources/<name>/state_variables": 4.0, "/parts/<name>/state_variables": 2.0,
                      "/resources/<name>/snapshot": 1.0, "/resources/<name>/static_capabilities": 2.0,
                      "/parts/<name>/pose_relations": 2.0, "/parts/<name>/target": 1.0},
    "Conveyor breakdown": {"/resources/<name>/state_variables": 4.0, "/parts/<name>/state_variables": 1.0,
                           "/resources/<name>/snapshot": 1.0, "/resources/<name>/static_capabilities": 2.0,
                           "/regions": 1.0, "/resources/<name>/pose_relations": 1.0,
                           "/resources/<name>/pending_tasks": 4.0},
    "Machining breakdown during part processing": {"/resources/<name>/state_variables": 5.0,
                                                   "/resources/<name>/snapshot": 3.0,
                                                   "/resources/<name>/static_capabilities": 1.0,
                                                   "/parts/<name>/target": 1.0},
    "ur5e-1 breakdown": {"/resources/<name>/state_variables": 2.0, "/resources/<name>/snapshot": 3.0,
                         "/resources/<name>/static_capabilities": 2.0, "/geometry": 1.0, "/regions": 1.0,
                         "/resources/<name>/pose_relations": 1.0, "/parts/<name>/target": 1.0},
}
# Main model, setting 6 through primitive composition.
_PLACEHOLDER_COMPOSITION = {
    "Part slippage": {"comp": 15, "ev": [4, 3], "prims": 21, "lookups": [12, 23], "targeted": 90, "tokens_k": 410, "minutes": 19},
    "Conveyor breakdown": {"comp": 17, "ev": [3, 0], "prims": 14, "lookups": [15, 17], "targeted": 84, "tokens_k": 450, "minutes": 22},
    "Machining breakdown during part processing": {"comp": 7, "ev": [7, 2], "prims": 27, "lookups": [11, 35], "targeted": 89, "tokens_k": 780, "minutes": 41},
    "ur5e-1 breakdown": {"comp": 11, "ev": [5, 1], "prims": 19, "lookups": [11, 29], "targeted": 93, "tokens_k": 620, "minutes": 33},
}


def placeholder_cells() -> dict:
    """Return layout numbers in the shape cells_from_attempts returns."""
    main, second = MODELS[0], MODELS[1]
    cells = {model["id"]: [{scenario: empty_cell() for scenario in SCENARIOS} for _ in CONFIGURATIONS]
             for model in MODELS}
    for scenario in SCENARIOS:
        for setting in range(len(CONFIGURATIONS)):
            counts = _PLACEHOLDER_OUTCOME[scenario][setting]
            tokens_k, minutes, lookups = _PLACEHOLDER_EFFORT[scenario][setting]
            cell = cells[main["id"]][setting][scenario]
            classes = [counts[0], counts[1], 0, counts[2], counts[3], 0, counts[4], 0]
            cell.update(n=main["planned"], classes=classes, tokens_k=tokens_k, minutes=minutes, lookups=lookups,
                        completed=main["planned"] - counts[4] if setting < FW_SINGLE else counts[0])
            passed, rest = _PLACEHOLDER_SECOND_PASSED[scenario][setting], second["planned"] - _PLACEHOLDER_SECOND_PASSED[scenario][setting]
            if setting < FW_SINGLE:
                infeasible, unsafe = round(rest * 0.55), round(rest * 0.2)
                continuation = min(round(rest * 0.2), rest - infeasible - unsafe)
                second_counts = [passed, infeasible, 0, unsafe, continuation, 0,
                                 rest - infeasible - unsafe - continuation, 0]
            else:
                second_counts = [passed, 0, 0, 0, 0, 0, rest, 0]
            cells[second["id"]][setting][scenario].update(
                n=second["planned"], classes=second_counts, tokens_k=round(tokens_k * 1.3),
                minutes=round(minutes * 1.8, 1), lookups=lookups + 2,
                completed=second["planned"] - second_counts[6] if setting < FW_SINGLE else passed)
        for offset, setting in enumerate((FW_SINGLE, FW_INCREMENTAL)):
            redo, ra, physical, cca, eq6, proposals = _PLACEHOLDER_CHECKS[scenario][offset]
            cells[main["id"]][setting][scenario].update(
                redo=redo, proposals=proposals, rejections={"ra": ra, "physical": physical, "cca": cca, "eq6": eq6})
            cells[second["id"]][setting][scenario].update(
                redo=redo, proposals=round(proposals * 0.55) if proposals else None,
                rejections={"ra": round(ra * 0.6), "physical": round(physical * 0.6), "cca": round(cca * 0.65),
                            "eq6": round(eq6 * 0.5)})
        cells[main["id"]][FW_INCREMENTAL][scenario]["references"] = dict(_PLACEHOLDER_REFERENCES[scenario])
    return cells


def reference_rows(cells: dict) -> list[str]:
    """Return every lookup reference the main model's setting 6 attempts used, in the prompt's order."""
    used = {path for cell in cells[MODELS[0]["id"]][FW_INCREMENTAL].values() for path in (cell["references"] or {})}
    return [path for path in REFERENCES if path in used] + sorted(used - set(REFERENCES))


def page_data(placeholder: bool, batch_ids: list[str] | None) -> dict:
    """Return the data block the page template renders."""
    if placeholder:
        attempts, sources, cells = [], [], placeholder_cells()
        composition = _PLACEHOLDER_COMPOSITION
    else:
        attempts, sources = attempts_from_batches(batch_ids)
        cells, composition = cells_from_attempts(attempts), None
    return {
        "mode": "placeholder" if placeholder else "records",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "settings": [{"n": name, "d": note} for name, note in zip(CONFIGURATIONS, SETTING_NOTES)],
        "scenarios": SCENARIOS, "scenario_short": SCENARIO_SHORT, "models": MODELS, "classes": CLASSES,
        "references": reference_rows(cells),
        "cells": cells, "composition": composition, "sources": sources, "attempts": attempt_lines(attempts),
        "max_turns": MAX_TURNS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--placeholder", action="store_true", help="fill the page with layout numbers")
    parser.add_argument("--batch", action="append", help="batch id to include (default: every study batch)")
    parser.add_argument("--out", type=Path, default=OUTPUT, help="page to write")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = page_data(args.placeholder, args.batch)
    marker = "/*RESULTS_DATA*/null"
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(marker) != 1:
        raise SystemExit(f"{TEMPLATE} must hold {marker} once")
    # "</" inside the JSON would end the script element early.
    block = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    args.out.write_text(template.replace(marker, block), encoding="utf-8")
    logger.info("Wrote %s (%s, %d attempts)", args.out, data["mode"], len(data["attempts"]))


if __name__ == "__main__":
    main()
