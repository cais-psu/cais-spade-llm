"""Write the RCIM results page (Section 6.3 figures and tables) from the Study 1 batch records.

The page has the layout of the figure mock-ups: Fig. 11, Table 6 and Table 7, all pooled over the models, Fig. 12
comparing the models, the optional Fig. O1 (outline lookups, pooled), and every attempt. With ``--placeholder`` it is filled with layout numbers
instead, so the page can be reviewed before any batch has run.

Usage, from the repository root:
    poetry run python "writing/Journal Paper 2 (recovery framework)/figures/results_page/results_page.py" --placeholder
    poetry run python "writing/Journal Paper 2 (recovery framework)/figures/results_page/results_page.py" [--batch ID ...]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import statistics
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

from cais_spade_llm.recovery_framework.outline_audit import OUTCOME_CLASSES  # noqa: E402
from cais_spade_llm.recovery_framework.outline_experiments import (  # noqa: E402
    AUDIT_REVIEW_NAME,
    BATCH_NAME,
    CONFIGURATIONS,
    EXPERIMENTS,
    RERUN_NAME,
    read_audit_review,
    read_batch,
    read_rerun,
    rerun_rows,
    reviewed_rows,
)
from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files  # noqa: E402

logger = logging.getLogger(__name__)

TEMPLATE = HERE / "results_template.html"
OUTPUT = HERE / "results.html"
# Fig. 11, Tables 6–7 and Fig. O1 pool these models; Fig. 12 compares them.
# planned: attempts per setting and scenario (user decision 2026-11-06: 10 for every model). mark: Fig. 12 shape.
# color: Fig. 12 color, none of which Fig. 11's outcomes use (user decision, 2026-11-09).
MODELS = [
    {"id": "opus55", "model": "claude-opus-5-5", "name": "Claude Opus 5.5 medium", "role": "Anthropic flagship",
     "vendor": "Claude", "color": "var(--m1)", "mark": "square", "planned": 10},
    {"id": "sol", "model": "gpt-6.1-sol", "name": "GPT-6.1 Sol medium", "role": "OpenAI flagship",
     "vendor": "OpenAI", "color": "var(--m2)", "mark": "circle", "planned": 10},
    {"id": "terra", "model": "gpt-6.1-terra", "name": "GPT-6.1 Terra medium", "role": "OpenAI lower-priced",
     "vendor": "OpenAI", "color": "var(--m3)", "mark": "diamond", "planned": 10},
]
# Fig. 11, Tables 6–7 and Fig. O1 pool the models' attempts (user decision, 2026-11-08); Fig. 12 compares them.
POOLED = "all"
EFFORT = "medium"
MAX_TURNS = 40
SETTING_NOTES = ["one answer", "one answer", "one event per turn", "one event per turn", "whole-outline rewrites",
                 "one-event revisions"]
# The paper's setting names (user decision, 2026-11-03); the Experiments tab and batch records keep their labels.
PAPER_NAMES = {
    "LLM, single-pass": "Baseline, outline-level",
    "LLM + rules, single-pass": "Informed baseline, outline-level",
    "LLM, incremental": "Baseline, event-level",
    "LLM + rules, incremental": "Informed baseline, event-level",
    "Framework, single-pass": "Neuro-symbolic framework, outline-level",
    "Framework, incremental": "Neuro-symbolic framework, event-level",
}
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
# The OutlineDraft check flags, in the order Table 5 shows them.
CHECK_FLAGS = ("check_ra_feasibility", "check_cca_safety", "check_nonblocking")
# OutlineDraft hides the safety rules when a setting leaves safety_rules_to_llm out.
RULES_DEFAULT = "hidden"


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


# Batches left out of every result, with the reason the analysis plan records (Deviations).
EXCLUDED_BATCHES = {
    "batch_20261108T043222Z_79f045": "harness fault in the handoff check (fixed in 4ffcc2c); re-run in full",
    "batch_20261108T052056Z_195a39": "baseline batch started on the faulty commit, stopped; re-run in full on 4ffcc2c",
}


def attempts_from_batches(batch_ids: list[str] | None, live: set[str] = frozenset()) -> tuple[list[dict], list[dict]]:
    """Return the started attempts of the selected batches and one source line per batch.

    Batches are selected by id, or else every batch of a study model at the study effort and turn budget. A batch
    named in ``live`` is still being run, so its running attempts are left out instead of read as interrupted.
    """
    models = {model["model"]: model["id"] for model in MODELS}
    attempts, sources = [], []
    # Outcomes follow the post-collection audit review (user decision, 2026-11-12); the records keep the recorded ones.
    review = read_audit_review()
    # One cell's attempts come from its rerun (user decision, 2026-11-13); the replaced batches keep their records.
    rerun = read_rerun()
    for path in sorted(EXPERIMENTS.glob(f"*/{BATCH_NAME}")):
        if path.parent.name in EXCLUDED_BATCHES:
            logger.info("Left out %s: %s", path.parent.name, EXCLUDED_BATCHES[path.parent.name])
            continue
        batch = read_batch(path, live=path.parent.name if path.parent.name in live else None)
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
        started: dict[str, int] = {}
        rows = rerun_rows(batch["batch_id"], batch.get("rows") or [], rerun)
        for row in reviewed_rows(rows, review):
            if row.get("status") not in STARTED:
                continue
            setting = setting_index((batch.get("configurations") or {}).get(row.get("configuration"), {}))
            if setting is None:
                logger.info("Left out %s %s: %s is not a current setting", batch["batch_id"], row.get("attempt_id"),
                            row.get("configuration"))
                continue
            source = row.get("rerun_batch") or batch["batch_id"]
            started[source] = started.get(source, 0) + 1
            attempts.append({"model": models[settings["model"]], "setting": setting, "scenario": row.get("scenario"),
                             "batch": source, "row": row, **_draft_details(row)})
        sources.append(_source(batch, started.get(batch["batch_id"], 0)))
        replacing = (rerun.get("replaces") or {}).get(batch["batch_id"])
        if replacing is not None:
            rerun_batch = read_batch(EXPERIMENTS.parent / replacing / BATCH_NAME)
            sources.append({**_source(rerun_batch, started.get(rerun_batch["batch_id"], 0)),
                            "replaces": f"{rerun['scenario']}, {rerun['configuration']} of {batch['batch_id']}"})
    return attempts, sources


def _source(batch: dict, started: int) -> dict:
    git = batch.get("git") or {}
    settings = batch.get("settings") or {}
    return {"batch": batch["batch_id"], "model": settings.get("model"), "status": batch.get("status"),
            "started": started, "planned": len(batch.get("plan") or []),
            "created": str(batch.get("created_at", ""))[:16].replace("T", " "),
            "code": f"{str(git.get('head', '?'))[:8]}, {git.get('uncommitted_files', '?')} uncommitted files"}


def rerun_note(attempts: list[dict]) -> dict | None:
    """Return what the page says about the cell rerun: its file and cell, and the attempts counted from it and in place
    of, as the replaced batches record them after the audit review."""
    rerun = read_rerun()
    counted = [attempt for attempt in attempts if attempt["row"].get("rerun_batch")]
    if not counted:
        return None
    cell = (rerun["scenario"], rerun["configuration"])
    used = {attempt["row"]["rerun_batch"] for attempt in counted}
    replaced = [row for batch_id, source in rerun["replaces"].items() if Path(source).name in used
                for row in reviewed_rows(read_batch(EXPERIMENTS / batch_id / BATCH_NAME).get("rows") or [])
                if (row.get("scenario"), row.get("configuration")) == cell and row.get("status") in STARTED]
    return {"name": RERUN_NAME, "scenario": rerun["scenario"],
            "setting": PAPER_NAMES.get(rerun["configuration"], rerun["configuration"]), "attempts": len(counted),
            "passed": sum(attempt["row"].get("outcome") == "passed" for attempt in counted),
            "replaced": len(replaced), "replaced_passed": sum(row.get("outcome") == "passed" for row in replaced)}


def cells_from_attempts(attempts: list[dict]) -> dict:
    """Return one cell per model, setting and scenario, and the same cells pooled over every model under ``POOLED``.

    Each cell holds counts by class, medians and check totals.
    """
    cells = {model_id: [{scenario: empty_cell() for scenario in SCENARIOS} for _ in CONFIGURATIONS]
             for model_id in [*(model["id"] for model in MODELS), POOLED]}
    grouped: dict[tuple, list[dict]] = {}
    for attempt in attempts:
        if attempt["scenario"] in SCENARIOS:
            for model_id in (attempt["model"], POOLED):
                grouped.setdefault((model_id, attempt["setting"], attempt["scenario"]), []).append(attempt)
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
    names = [PAPER_NAMES.get(name, name) for name in CONFIGURATIONS]
    lines = []
    for attempt in attempts:
        row = attempt["row"]
        tokens, seconds = _number(row.get("tokens")), row.get("elapsed_seconds")
        recorded = row.get("recorded_outcome")
        lines.append({
            "model": attempt["model"], "setting": names[attempt["setting"]],
            "scenario": attempt["scenario"],
            "outcome": row.get("outcome") if recorded in (None, row.get("outcome"))
            else f"{row.get('outcome')} (recorded {recorded})",
            "stage": row.get("failing_stage") or "", "stop": row.get("stop_cause"),
            "events": row.get("outline_length"), "lookups": row.get("lookups"),
            "tokens_k": round(tokens / 1000) if tokens is not None else None,
            "minutes": round(seconds / 60, 1) if isinstance(seconds, (int, float)) else None,
            "attempt": f"{attempt['batch']} · {row.get('attempt_id')}",
        })
    order = {scenario: index for index, scenario in enumerate(SCENARIOS)}
    model_order = {model["id"]: index for index, model in enumerate(MODELS)}
    return sorted(lines, key=lambda line: (model_order[line["model"]], names.index(line["setting"]),
                                           order.get(line["scenario"], 9)))


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
# Table 7 layout numbers: per scenario, the composed outlines pooled over the models (3 per model).
_PLACEHOLDER_COMPOSITION = {
    "Part slippage": {"composed": 10, "passed": 9, "share": 0.94, "events": 5, "primitives": 21, "refused": 2, "calls": 8, "tokens_k": 190, "minutes": 14},
    "Conveyor breakdown": {"composed": 9, "passed": 8, "share": 0.91, "events": 9, "primitives": 30, "refused": 4, "calls": 12, "tokens_k": 240, "minutes": 18},
    "Machining breakdown during part processing": {"composed": 9, "passed": 6, "share": 0.78, "events": 18, "primitives": 52, "refused": 7, "calls": 20, "tokens_k": 420, "minutes": 33},
    "ur5e-1 breakdown": {"composed": 9, "passed": 8, "share": 0.9, "events": 7, "primitives": 26, "refused": 3, "calls": 11, "tokens_k": 230, "minutes": 17},
}
# Table 8 layout numbers: the selected Claude Opus 5.5 outline per scenario executed in Gazebo.
_PLACEHOLDER_EXECUTION = {
    "Part slippage": {"model": "Claude Opus 5.5", "events": 4, "primitives": 17, "steps": 17, "completed": 17, "sim_seconds": 95, "result": "completed", "video": ""},
    "Conveyor breakdown": {"model": "Claude Opus 5.5", "events": 7, "primitives": 28, "steps": 28, "completed": 28, "sim_seconds": 240, "result": "completed", "video": ""},
    "Machining breakdown during part processing": {"model": "Claude Opus 5.5", "events": 16, "primitives": 49, "steps": 49, "completed": 49, "sim_seconds": 610, "result": "completed", "video": ""},
    "ur5e-1 breakdown": {"model": "Claude Opus 5.5", "events": 7, "primitives": 27, "steps": 27, "completed": 27, "sim_seconds": 260, "result": "completed", "video": ""},
}


# Study 2's compositions: all 36 composed once on the gear-turn commit (round r9, user decision 2026-11-08); each log
# names its composition directory, and its name is <scenario>_<model>_<rank>. Sol's and Terra's second Part slippage
# attempts are in logs_r9b and stay out of Table 7, which holds one attempt per outline.
COMPOSITION_LOGS = EXPERIMENTS.parent / "study2_compositions" / "logs_r9"
# User decisions 2026-11-08/09: Sol's Part slippage outlines were composed twice on the final commit, and Table 7
# reports the second attempt for each of them, whatever it gave; the caption says so. Terra gets no second attempt.
SECOND_ATTEMPT_LOGS = EXPERIMENTS.parent / "study2_compositions" / "logs_r9b"
SECOND_ATTEMPT = {"PS_sol_1", "PS_sol_2", "PS_sol_3"}
# User decision 2026-11-12 (Option B): Terra's three Machining outlines were composed again after the Conveyor's
# primitives were served with the descriptions and outputs the first compositions lacked (only Terra's outlines have a
# Conveyor event). Table 7 reports the re-run in place of the first composition, whatever it gave; the caption says so.
CONVEYOR_RERUN_LOGS = EXPERIMENTS.parent / "study2_compositions" / "logs_r10_conveyor"
CONVEYOR_RERUN = {"MACH_terra_1", "MACH_terra_2", "MACH_terra_3"}
COMPOSITION_SCENARIOS = {"PS": "Part slippage", "CONV": "Conveyor breakdown",
                         "MACH": "Machining breakdown during part processing", "UR5": "ur5e-1 breakdown"}
COMPOSITION_MODELS = {"opus": "opus55", "sol": "sol", "terra": "terra"}


def _usage(record: Any) -> dict | None:
    """Return the first provider usage block in a response record."""
    if isinstance(record, dict):
        if "input_tokens" in record or "prompt_tokens" in record:
            return record
        values = record.values()
    elif isinstance(record, list):
        values = record
    else:
        return None
    return next((found for value in values if (found := _usage(value)) is not None), None)


# Composition lookup references in the order the composition prompt lists them, with the resource, part, region,
# key or name left out; the model and evidence records the prompt points to follow their resource.
COMPOSITION_REFERENCES = [
    "/outline_event", "/resources/<jid>/snapshot", "/resources/<jid>/current_pose", "/resources/<jid>/held_part",
    "/resources/<jid>/static_capabilities", "/resources/<jid>/recovery_des_model",
    "/resources/<jid>/resource_evidence/controller_config/cartesian_motion", "/resources/<jid>/resource_evidence/<field>",
    "/parts/<name>/snapshot", "/parts/<name>/observed_pose", "/parts/<name>/target", "/primitive_catalog/<jid>",
    "/primitive_contracts/<name>", "/projected_snapshot/<jid>", "/capability_decompositions/<function_name>",
    "/regions", "/regions/<name>", "/geometry", "/geometry/<key>",
]
_NAMED = {"primitive_contracts": "<name>", "primitive_catalog": "<jid>", "projected_snapshot": "<jid>",
          "capability_decompositions": "<function_name>", "regions": "<name>", "geometry": "<key>"}


def composition_reference(ref: str) -> str:
    """Return a requested reference with its resource, part, region, key or name left out."""
    head, *rest = [segment for segment in str(ref).split("/") if segment] or [""]
    if head == "resources" and rest:
        tail = rest[1:]
        if tail[:1] == ["resource_evidence"]:
            return ("/resources/<jid>/resource_evidence/controller_config/cartesian_motion"
                    if tail[1:3] == ["controller_config", "cartesian_motion"]
                    else "/resources/<jid>/resource_evidence/<field>")
        return "/resources/<jid>" + (f"/{tail[0]}" if tail else "")
    if head == "parts" and rest:
        return "/parts/<name>" + (f"/{rest[1]}" if len(rest) > 1 else "")
    if head in _NAMED:
        return f"/{head}" + (f"/{_NAMED[head]}" if rest else "")
    return f"/{head}"


def _read_reply(path: Path) -> str:
    """Read one composing turn's response record, as written (.txt) or compressed (.txt.gz)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as handle:
            return handle.read()
    return path.read_text(errors="replace")


def _reply(text: str) -> dict:
    """Return a composing LLM's turn reply (decision, context_requests, primitive_steps) from its response record."""
    start = text.find("{")
    try:
        record = json.loads(text[start:]) if start >= 0 else {}
    except json.JSONDecodeError:
        return {}
    # The record keeps the LLM's reply under "response"; its schema elsewhere also names "decision".
    reply = record.get("response") if isinstance(record, dict) else None
    if isinstance(reply, dict) and isinstance(reply.get("decision"), str):
        return reply
    raw = ((record.get("validation_evidence") or {}).get("llm_raw_response") if isinstance(record, dict) else None)
    return raw if isinstance(raw, dict) and isinstance(raw.get("decision"), str) else {}


# A composing turn's prompt and reply records share one stamp. Each prompt's feedback block lists what the checks
# and the composing LLM reported for the turns before it, so a turn's refusal is what the next prompt adds.
_COMPOSING_TURN = re.compile(r"multi_turn_(RECOVERY_SEQ\d+)_primitive_generation_turn(\d+)_(prompt|response)_(\d{8}T\d{6})\.txt")
_FEEDBACK_LINE = re.compile(r"^- RECOVERY_SEQ\d+ / \S+ \[(\w+)\]: (.*)$")


def refusal_message(text: str) -> str:
    """Return a refusal message with its numbers, poses and quoted values left out, so equal refusals count together.

    Args:
        text: The message as the feedback block records it.

    Returns:
        The message's first clause, with every step index, number, pose and quoted value replaced by an ellipsis.
    """
    if text.startswith("LLM reported primitive-surface gap"):
        return "LLM reported primitive-surface gap"
    text = re.sub(r"^(\w+) step \d+ \(([^)]*)\): ", r"\1 (\2): ", text).split(";")[0]
    # Innermost braces and brackets first, so nested values collapse to one ellipsis.
    while re.search(r"\{[^{}]*\}|\[[^\[\]]*\]", text):
        text = re.sub(r"\{[^{}]*\}|\[[^\[\]]*\]", "…", text)
    # Numbers standing alone, not digits inside a name such as KET4_Square_4mm or ur5e-3.
    text = re.sub(r"\(-?\d[^)]*\)|'[^']*'|(?<![\w.-])-?\d+(?:\.\d+)?(?:e-?\d+)?%?(?![\w])", "…", text)
    text = re.sub(r"(?:…[\s,]*)+", "… ", text).strip()
    if len(text) <= 130:
        return text
    cut = text[:130].rsplit(" ", 1)[0]
    return cut if cut.endswith("…") else cut + " …"


def _feedback(prompt: Path) -> list[tuple[str, str]]:
    """Return a composing prompt's feedback lines as (code, message)."""
    text = prompt.read_text(errors="replace")
    start = text.find("Previous Validation Feedback")
    lines = []
    for line in text[start:].splitlines()[1:] if start >= 0 else []:
        match = _FEEDBACK_LINE.match(line)
        if not match:
            break
        lines.append((match.group(1), match.group(2)))
    return lines


def composition_refusals(directory: Path, turns: dict[tuple[str, int], dict], stop_message: str) -> list[dict]:
    """Return every refusal one composition received, and whether its event was accepted after it.

    A gap the composing LLM reports is its own reply. A refused program's reasons are what the next prompt's
    feedback block adds; the last refused program of a stopped composition has no next prompt, and its reason is the
    composition's stop message.

    Args:
        directory: The composition's directory.
        turns: Each composing turn's stamp, LLM decision and recorded outcome, per (outline id, turn).
        stop_message: The composition's final message.

    Returns:
        One row per refusal: its record code, its message as ``refusal_message`` gives it, and ``fixed`` when a later
        program for the same event was accepted.
    """
    prompts: dict[tuple[str, int], Path] = {}
    for path in (directory / "recovery_primitves").glob("multi_turn_*_prompt_*.txt"):
        match = _COMPOSING_TURN.match(path.name)
        if match:
            prompts[(match.group(1), int(match.group(2)))] = path
    order = sorted(turns, key=lambda key: turns[key]["stamp"])
    rows = []
    for index, key in enumerate(order):
        turn = turns[key]
        fixed = any(other[0] == key[0] and other[1] > key[1] and turns[other]["outcome"] == "draft_ready"
                    for other in turns)
        if turn["decision"] in ("need_primitive_revision", "primitive_blocked"):
            rows.append({"code": "primitive_revision_requested", "message": "LLM reported primitive-surface gap",
                         "fixed": fixed})
            continue
        if turn["decision"] != "primitive_steps_ready" or turn["outcome"] == "draft_ready":
            continue
        later = order[index + 1] if index + 1 < len(order) else None
        if later is not None and key in prompts and later in prompts:
            added = _feedback(prompts[later])
            for line in _feedback(prompts[key]):
                if line in added:
                    added.remove(line)
            reasons = [(code, message) for code, message in added if code != "primitive_revision_requested"]
        else:
            reason = stop_message.split("paused_after_primitive_blocked: ", 1)[-1] if later is None else ""
            reasons = [("composition_infeasible", reason)] if reason and "stalled" not in reason else []
        rows.extend({"code": code, "message": refusal_message(message), "fixed": fixed} for code, message in reasons
                    or [("composition_infeasible", "(refusal text not recorded)")])
    return rows


def composition_row(log: Path) -> dict | None:
    """Return one finished composition: its outline's events, how many were composed, and its effort."""
    from cais_spade_llm.recovery_framework.outline_draft import _token_usage

    text = log.read_text(errors="replace")
    result = next((json.loads(line) for line in reversed(text.splitlines())
                   if line.startswith("{") and '"primitive_status"' in line), None)
    found = [line.split("composing into ", 1)[1].strip() for line in text.splitlines() if "composing into " in line]
    if result is None or not found:
        return None
    directory = Path(found[-1])
    scenario, model, _rank = log.stem.split("_")
    events = len(json.loads((directory / "outline.json").read_text())["outline"])
    accepted = int(result.get("accepted") or 0)
    passed = result.get("primitive_status") == "completed"
    primitives = None
    if passed:
        program = json.loads((directory / "composition.json").read_text())["accepted_primitive_program"]
        primitives = sum(len(event.get("primitive_steps") or []) for event in program)
    # Study 2's response records were gzip-compressed on 2026-11-10 to save disk space; they read the same.
    replies, turns_seen = [], {}
    for path in sorted((directory / "recovery_primitves").glob("multi_turn_*_response_*.txt*")):
        text = _read_reply(path)
        reply = _reply(text)
        replies.append(reply)
        match = _COMPOSING_TURN.match(path.name.removesuffix(".gz"))
        try:
            outcome = json.loads(text[text.find("{"):]).get("decision") if match else None
        except json.JSONDecodeError:
            outcome = None
        if match:
            turns_seen[(match.group(1), int(match.group(2)))] = {
                "stamp": match.group(4), "decision": reply.get("decision"), "outcome": outcome}
    turns = [reply.get("decision") for reply in replies]
    # A lookup is one reference the composing LLM requested, as the outline stage counts its retrievals.
    requested = [ref for reply in replies if reply.get("decision") == "need_context"
                 for ref in reply.get("context_requests") or []]
    lookups = len(requested)
    references: dict[str, int] = {}
    for ref in requested:
        references[composition_reference(ref)] = references.get(composition_reference(ref), 0) + 1
    calls = tokens = 0
    for path in sorted((directory / "recovery_primitves" / "requests").glob("*/round*_response.json")):
        calls += 1
        usage = _usage(json.loads(path.read_text()))
        tokens += int(_token_usage(usage).get("total_tokens") or 0) if usage else 0
    return {"name": log.stem, "scenario": COMPOSITION_SCENARIOS[scenario], "model": COMPOSITION_MODELS[model],
            "events": events, "accepted": accepted, "share": accepted / events if events else None, "passed": passed,
            "primitives": primitives, "refused": max(turns.count("primitive_steps_ready") - accepted, 0),
            "calls": calls, "lookups": lookups, "references": references, "tokens": tokens,
            "seconds": result.get("seconds"), "refusals": composition_refusals(directory, turns_seen, str(result.get("message") or ""))}


def composition_summary(rows: list[dict]) -> dict | None:
    """Return Table 7's numbers for some compositions: counts, the mean share composed and the medians."""
    if not rows:
        return None
    tokens = _median([row["tokens"] for row in rows])
    minutes = _median([row["seconds"] / 60 for row in rows if row.get("seconds") is not None])
    return {"composed": len(rows), "passed": sum(row["passed"] for row in rows),
            "share": statistics.fmean(row["share"] for row in rows if row["share"] is not None),
            "events": _median([row["events"] for row in rows]),
            "primitives": _median([row["primitives"] for row in rows if row["primitives"] is not None]),
            "refused": _median([row["refused"] for row in rows]), "calls": _median([row["calls"] for row in rows]),
            "lookups": _median([row["lookups"] for row in rows]),
            "tokens_k": _round(tokens / 1000 if tokens is not None else None),
            "minutes": _round(minutes, 1)}


def compositions_from_logs(logs: Path = COMPOSITION_LOGS) -> dict | None:
    """Return Table 7: per scenario, pooled over the models, and per model; finished compositions only."""
    rows = []
    for log in sorted(logs.glob("*.log")):
        second = SECOND_ATTEMPT_LOGS / log.name
        rerun = CONVEYOR_RERUN_LOGS / log.name
        use_second = log.stem in SECOND_ATTEMPT and second.exists() and composition_row(second) is not None
        use_rerun = log.stem in CONVEYOR_RERUN and rerun.exists() and composition_row(rerun) is not None
        row = composition_row(rerun if use_rerun else second if use_second else log)
        if row is not None:
            rows.append({**row, "attempt": 2 if use_second else 1, "conveyor_rerun": use_rerun})
    if not rows:
        return None
    references = {}
    for scenario in SCENARIOS:
        composed = [row for row in rows if row["scenario"] == scenario]
        paths = {path for row in composed for path in row["references"]}
        references[scenario] = ({path: round(sum(row["references"].get(path, 0) for row in composed) / len(composed), 1)
                                 for path in paths} if composed else None)
    used = {path for counts in references.values() if counts for path in counts}
    reference_rows = [path for path in COMPOSITION_REFERENCES if path in used] + sorted(used - set(COMPOSITION_REFERENCES))
    second_attempts = sorted(row["name"] for row in rows if row["attempt"] == 2)
    conveyor_reruns = sorted(row["name"] for row in rows if row["conveyor_rerun"])
    refusals: dict[tuple[str, str], dict] = {}
    for row in rows:
        for refusal in row["refusals"]:
            entry = refusals.setdefault((refusal["code"], refusal["message"]), {
                "code": refusal["code"], "message": refusal["message"], "models": {}, "fixed": 0, "open": 0})
            entry["models"][row["model"]] = entry["models"].get(row["model"], 0) + 1
            entry["fixed" if refusal["fixed"] else "open"] += 1
    return {"second_attempts": second_attempts, "conveyor_reruns": conveyor_reruns, "references": references, "reference_rows": reference_rows,
            "scenarios": {scenario: composition_summary([row for row in rows if row["scenario"] == scenario])
                          for scenario in SCENARIOS},
            "total": composition_summary(rows),
            "models": {model["id"]: composition_summary([row for row in rows if row["model"] == model["id"]])
                       for model in MODELS},
            "refusals": sorted(refusals.values(), key=lambda entry: (entry["code"], -sum(entry["models"].values()))),
            "rows": rows}


def placeholder_cells() -> dict:
    """Return layout numbers in the shape cells_from_attempts returns."""
    main, others = MODELS[0], MODELS[1:]
    cells = {model_id: [{scenario: empty_cell() for scenario in SCENARIOS} for _ in CONFIGURATIONS]
             for model_id in [*(model["id"] for model in MODELS), POOLED]}
    for scenario in SCENARIOS:
        for setting in range(len(CONFIGURATIONS)):
            counts = _PLACEHOLDER_OUTCOME[scenario][setting]
            tokens_k, minutes, lookups = _PLACEHOLDER_EFFORT[scenario][setting]
            cell = cells[main["id"]][setting][scenario]
            classes = [counts[0], counts[1], 0, counts[2], counts[3], 0, counts[4], 0]
            cell.update(n=sum(counts), classes=classes, tokens_k=tokens_k, minutes=minutes, lookups=lookups,
                        completed=sum(counts) - counts[4] if setting < FW_SINGLE else counts[0])
            for second in others:
                passed = _PLACEHOLDER_SECOND_PASSED[scenario][setting]
                rest = second["planned"] - passed
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
            for second in others:
                cells[second["id"]][setting][scenario].update(
                    redo=redo, proposals=round(proposals * 0.55) if proposals else None,
                    rejections={"ra": round(ra * 0.6), "physical": round(physical * 0.6), "cca": round(cca * 0.65),
                                "eq6": round(eq6 * 0.5)})
        cells[main["id"]][FW_INCREMENTAL][scenario]["references"] = dict(_PLACEHOLDER_REFERENCES[scenario])
    cells[POOLED] = deepcopy(cells[main["id"]])
    return cells


def reference_rows(cells: dict) -> list[str]:
    """Return every lookup reference the main model's setting 6 attempts used, in the prompt's order."""
    used = {path for cell in cells[MODELS[0]["id"]][FW_INCREMENTAL].values() for path in (cell["references"] or {})}
    return [path for path in REFERENCES if path in used] + sorted(used - set(REFERENCES))


# ---------------------------------------------------------------- experimental conditions
def setting_rows() -> list[dict]:
    """Return one row per setting, read from the exact OutlineDraft flags it runs with."""
    return [{"name": PAPER_NAMES.get(name, name), "single_pass": bool(flags.get("single_pass_outline")),
             "rules_in_prompt": flags.get("safety_rules_to_llm", RULES_DEFAULT) == "prompt",
             "checks": {flag: bool(flags.get(flag)) for flag in CHECK_FLAGS}}
            for name, flags in CONFIGURATIONS.items()]


def scenario_rows() -> list[dict]:
    """Return each scenario's safety specification count."""
    return [{"scenario": scenario, "safety_specifications": len(default_safety_files(scenario))}
            for scenario in SCENARIOS]


def conditions() -> dict:
    """Return the experimental conditions: the settings, the failure scenarios and what every attempt shares."""
    import inspect

    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        generate_primitive_batch_with_llm_agent,
    )
    from cais_spade_llm.recovery_framework.primitive_composition import _COMPOSITION_TURN_BUDGET

    event_turns = inspect.signature(generate_primitive_batch_with_llm_agent).parameters["max_turns"].default
    return {"settings": setting_rows(), "scenarios": scenario_rows(), "effort": EFFORT,
            "composition_turns": _COMPOSITION_TURN_BUDGET, "composition_event_turns": event_turns}


def page_data(placeholder: bool, batch_ids: list[str] | None, live: set[str] = frozenset()) -> dict:
    """Return the data block the page template renders."""
    if placeholder:
        attempts, sources, cells = [], [], placeholder_cells()
        composition = {"scenarios": _PLACEHOLDER_COMPOSITION, "total": None, "models": None, "rows": []}
        execution = _PLACEHOLDER_EXECUTION
    else:
        attempts, sources = attempts_from_batches(batch_ids, live)
        cells, composition, execution = cells_from_attempts(attempts), compositions_from_logs(), None
    return {
        "mode": "placeholder" if placeholder else "records",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "settings": [{"n": PAPER_NAMES.get(name, name), "d": note} for name, note in zip(CONFIGURATIONS, SETTING_NOTES)],
        "scenarios": SCENARIOS, "scenario_short": SCENARIO_SHORT, "models": MODELS, "classes": CLASSES,
        "references": reference_rows(cells),
        "cells": cells, "composition": composition, "execution": execution, "sources": sources, "attempts": attempt_lines(attempts),
        "max_turns": MAX_TURNS, "conditions": conditions(),
        "audit_review": None if placeholder else {
            "name": AUDIT_REVIEW_NAME,
            "reaudited": sum("recorded_outcome" in attempt["row"] for attempt in attempts),
            "changed": sum(attempt["row"].get("recorded_outcome") not in (None, attempt["row"].get("outcome"))
                           for attempt in attempts)},
        "rerun": None if placeholder else rerun_note(attempts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--placeholder", action="store_true", help="fill the page with layout numbers")
    parser.add_argument("--batch", action="append", help="batch id to include (default: every study batch)")
    parser.add_argument("--out", type=Path, default=OUTPUT, help="page to write")
    parser.add_argument("--live", action="append", default=[],
                        help="batch id still being run: its running attempts are left out, not read as interrupted")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = page_data(args.placeholder, args.batch, set(args.live))
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
