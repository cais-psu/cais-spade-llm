from __future__ import annotations

"""Study 1 batches: the ablation configurations run on the retained failure captures.

A batch runs its planned attempts one at a time, each a fresh OutlineDraft on a saved capture, and audits
every attempt (``outline_audit``). Its manifest freezes the settings, the exact flags of each
configuration, the attempt order and the hashes of the code and inputs; a changed file stops the batch.
Batch and attempt records are experiment results and are never deleted, whether an attempt finished,
failed, was stopped or was interrupted.
"""

import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import random
import secrets
import statistics
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.evaluation_capture import write_record
from cais_spade_llm.recovery_framework.outline_audit import (
    AUDIT_NAME,
    DID_NOT_FINISH,
    OUTCOME_CLASSES,
    ra_stages,
)
from cais_spade_llm.recovery_framework.recovery_observations import RECORDS

logger = logging.getLogger(__name__)

EXPERIMENTS = RECORDS / "outline_experiments"
BATCH_NAME = "batch.json"
# Display labels only: each maps to the exact OutlineDraft flags it runs with.
CONFIGURATIONS = {
    "LLM, single-pass": {"single_pass_outline": True},
    # The prompt lists the safety rules, and nothing checks the LLM's events.
    "LLM + rules, single-pass": {"single_pass_outline": True, "safety_rules_to_llm": "prompt"},
    "LLM, incremental": {},
    "LLM + rules, incremental": {"safety_rules_to_llm": "prompt"},
    # The framework hides the safety rules from the LLM; the CCA checks every rule.
    "Framework, single-pass": {"single_pass_outline": True, "check_ra_feasibility": True,
                               "check_cca_safety": True, "check_nonblocking": True},
    "Framework, incremental": {"check_ra_feasibility": True, "check_cca_safety": True, "check_nonblocking": True},
}
MAX_ATTEMPTS = 20
MAX_TURNS = 200
_ENDED = {"finished", "failed", "interrupted"}
DEFAULT_SETTINGS = {"attempts": dict.fromkeys(CONFIGURATIONS, 3), "model": "gpt-5.4", "reasoning_effort": "medium",
                    "max_turns": 80}
REJECTION_CHECKS = ("ra_feasibility", "physical_evidence", "cca_safety", "nonblocking")
NOT_RECORDED = "not recorded"
ROW_COLUMNS = (
    "attempt_id", "scenario", "configuration", "repetition", "status", "outcome", "failing_stage",
    "first_failing_event", "draft_status", "stop_cause", "outline_length", "handoff_complete",
    *(f"rejections_{check}" for check in REJECTION_CHECKS), "revisions", "rewrites", "lookups", "lookup_turns",
    "provider_calls", "tokens", "prompt_tokens", "completion_tokens", "reasoning_tokens", "elapsed_seconds",
    "audit_seconds", "model", "reasoning_effort", "message", "draft_path",
)


def checked_settings(settings: dict, captures: dict[str, str]) -> dict:
    """Return the batch settings, or raise ValueError naming what is wrong.

    Args:
        settings: ``scenarios``, ``attempts`` per configuration, ``model``, ``reasoning_effort`` and
            ``max_turns``.
        captures: Retained capture paths by exact scenario name.

    Returns:
        The settings the batch records and runs with.
    """
    from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryProduct

    scenarios = list(settings.get("scenarios") or [])
    if not scenarios:
        raise ValueError("Select at least one failure scenario")
    missing = [scenario for scenario in scenarios if scenario not in captures]
    if missing:
        raise ValueError("No retained capture for " + ", ".join(missing))
    attempts = {name: (settings.get("attempts") or {}).get(name, 0) for name in CONFIGURATIONS}
    for name, count in attempts.items():
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= MAX_ATTEMPTS:
            raise ValueError(f"Attempts for {name} must be a whole number from 0 to {MAX_ATTEMPTS}")
    if not any(attempts.values()):
        raise ValueError("Give at least one configuration one attempt")
    model, effort = settings.get("model"), settings.get("reasoning_effort")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Give the model to run")
    if effort is None and model != "gpt-4o":
        # Without an explicit effort the provider would fall back to a capture's saved one or the environment.
        raise ValueError(f"Choose a reasoning effort for {model}; 'not sent' is only for models without one, "
                         "such as gpt-4o")
    # The provider settings refuse a model and reasoning effort they would not send as given.
    SavedRecoveryProduct({}, [], model=model, reasoning_effort=effort)
    max_turns = settings.get("max_turns")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or not 1 <= max_turns <= MAX_TURNS:
        raise ValueError(f"Max turns must be a whole number from 1 to {MAX_TURNS}")
    return {"scenarios": scenarios, "attempts": attempts, "model": model, "reasoning_effort": effort,
            "max_turns": max_turns}


def planned_attempts(scenarios: list[str], attempts: dict[str, int], seed: int) -> list[dict]:
    """Return the attempt order: per scenario and repetition, the configurations still due, shuffled.

    Args:
        scenarios: Selected scenarios, in order.
        attempts: Attempts per configuration.
        seed: Recorded scheduling seed.

    Returns:
        One entry per attempt with its identifier, scenario, configuration and repetition.
    """
    rng = random.Random(seed)
    plan = []
    for scenario in scenarios:
        for repetition in range(max(attempts.values())):
            block = [name for name in CONFIGURATIONS if attempts.get(name, 0) > repetition]
            rng.shuffle(block)
            plan += [{"scenario": scenario, "configuration": name, "repetition": repetition + 1} for name in block]
    return [{"attempt_id": f"attempt_{index:03d}", **attempt} for index, attempt in enumerate(plan, 1)]


def _process_started_at() -> float:
    """Return when this process started, read from /proc (the runtime targets Linux)."""
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    boot = next(float(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                if line.startswith("btime"))
    return boot + int(fields[19]) / os.sysconf("SC_CLK_TCK")


def changed_since_start(files: dict[str, str], *, root: Path = ROOT) -> list[str]:
    """Return the frozen files changed on disk after this process started.

    The UI keeps running the code it imported and keeps some inputs it read in process caches (the scene's
    grounding settings, for one), so after such a change a batch could run with other code or inputs than
    its manifest records.
    """
    started = _process_started_at()
    paths = {name: Path(name) if Path(name).is_absolute() else root / name for name in files}
    return [name for name, path in paths.items() if path.is_file() and path.stat().st_mtime > started]


def _git_state() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True, timeout=30).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                                check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}
    return {"head": head, "uncommitted_files": len(status.splitlines()),
            "status_sha256": hashlib.sha256(status.encode()).hexdigest()}


def rejections(record: dict) -> dict[str, int] | None:
    """Count what validation refused while the outline was generated, by check (Table 6).

    The incremental framework counts every checked candidate a check refused. The single-pass framework
    counts each refused outline once, at the check that refused its first failing step.

    Returns:
        Counts per check, or None for a draft generated without checks.
    """
    if not record.get("ra_feasibility_check"):
        return None
    counts = dict.fromkeys(REJECTION_CHECKS, 0)
    if record.get("single_pass_outline"):
        for outline in record.get("single_pass_checks") or []:
            if not outline.get("accepted"):
                counts[_refusing_check((outline.get("checks") or [{}])[-1])] += 1
        return counts
    for turn in record.get("ra_feasibility_checks") or []:
        for result in turn.get("results") or []:
            if result.get("feasibility_status") != "FEASIBLE":
                counts[_ra_check(result)] += 1
    for turn in record.get("cca_safety_checks") or []:
        counts["cca_safety"] += sum(result.get("safety_status") != "SAFE" for result in turn.get("results") or [])
    counts["nonblocking"] = sum(check.get("nonblocking_status") == "BLOCKING"
                                for check in record.get("cca_nonblocking_checks") or [])
    return counts


def _ra_check(result: dict) -> str:
    return "ra_feasibility" if ra_stages(result)["symbolic"] != "FEASIBLE" else "physical_evidence"


def _refusing_check(step: dict) -> str:
    if "completion" in step:
        return "nonblocking"
    ra = (step.get("ra_feasibility") or [{}])[0]
    return _ra_check(ra) if ra.get("feasibility_status") != "FEASIBLE" else "cca_safety"


def _revisions(order: list) -> int:
    """Count the refused proposals the LLM then revised: a later reply proposed again, not only looked up."""
    return sum(1 for index, decision in enumerate(order)
               if decision == "need_revision" and any(later != "need_context" for later in order[index + 1:]))


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def attempt_row(attempt: dict, directory: Path | None) -> dict:
    """Return one attempt's results row from its draft and audit records.

    Args:
        attempt: The planned attempt with its ``status`` and any ``error``.
        directory: The attempt's draft directory, if a draft was created.

    Returns:
        The row; a measurement the records do not hold reads "not recorded".
    """
    row = dict.fromkeys(ROW_COLUMNS, NOT_RECORDED)
    row.update({key: attempt.get(key) for key in ("attempt_id", "scenario", "configuration", "repetition", "status")})
    record = _read(directory / "draft.json") if directory is not None else None
    audit = _read(directory / AUDIT_NAME) if directory is not None else None
    if record is None:
        row.update(outcome=DID_NOT_FINISH if attempt.get("status") in {"failed", "interrupted"} else NOT_RECORDED,
                   message=attempt.get("error", NOT_RECORDED))
        return row
    usage = [entry.get("usage") or {} for entry in record.get("usage") or []]
    decisions = record.get("turn_decisions") or {}
    counts = rejections(record)
    validated = counts is not None
    order = record.get("turn_decision_order")
    row.update(
        draft_status=record.get("status"), stop_cause=(record.get("stop_cause") or {}).get("cause", NOT_RECORDED),
        outline_length=len(record.get("transition_trace") or []),
        handoff_complete=(record.get("nominal_handoff") or {}).get("complete", NOT_RECORDED),
        **{f"rejections_{check}": counts[check] if validated else "–" for check in REJECTION_CHECKS},
        revisions=(_revisions(order) if order is not None else decisions.get("need_revision", 0))
        if validated and not record.get("single_pass_outline") else "–",
        # Every outline after the first is a rewrite of a refused one.
        rewrites=max(len(record.get("single_pass_checks") or []) - 1, 0)
        if validated and record.get("single_pass_outline") else "–",
        lookups=len(record.get("retrievals") or []), lookup_turns=decisions.get("need_context", 0),
        provider_calls=record.get("provider_calls", NOT_RECORDED),
        tokens=sum(int(entry.get("total_tokens") or 0) for entry in usage) if usage else NOT_RECORDED,
        prompt_tokens=sum(int(entry.get("prompt_tokens") or 0) for entry in usage) if usage else NOT_RECORDED,
        completion_tokens=sum(int(entry.get("completion_tokens") or 0) for entry in usage) if usage else NOT_RECORDED,
        reasoning_tokens=sum(int((entry.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
                             for entry in usage) if usage else NOT_RECORDED,
        elapsed_seconds=record.get("elapsed_seconds", NOT_RECORDED),
        model=(record.get("llm_settings") or {}).get("model", NOT_RECORDED),
        reasoning_effort=(record.get("llm_settings") or {}).get("reasoning_effort"),
        message=record.get("message", NOT_RECORDED), draft_path=str(directory / "draft.json"),
    )
    if audit is None:
        # A draft that ended as an outline waits for its audit (Audit missing); any other did not finish.
        row["outcome"] = "not audited" if record.get("status") == "draft" else DID_NOT_FINISH
    elif audit.get("status") != "audited":
        row["outcome"] = "audit failed"
        row["message"] = audit.get("error", row["message"])
    else:
        row.update(outcome=audit["outcome"], failing_stage=audit.get("failing_stage") or "",
                   first_failing_event=audit.get("first_failing_event") or "",
                   audit_seconds=audit.get("elapsed_seconds", NOT_RECORDED))
    return row


def _median(values: list[Any]) -> float | str:
    numbers = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return round(statistics.median(numbers), 1) if numbers else NOT_RECORDED


def summarize(rows: list[dict]) -> dict[str, list[dict]]:
    """Return the Fig. 11 and Table 6 numbers of a batch.

    Args:
        rows: Attempt rows; attempts that never started are left out of every count.

    Returns:
        ``outcomes``: per configuration and scenario (and pooled), started attempts, the count of each
        outcome class and the median tokens, seconds and lookups. ``rejections``: per validated
        configuration and scenario (and pooled), the refusals by check and the revisions or rewrites.
    """
    started = [row for row in rows if row.get("status") in {"finished", "failed", "interrupted"}]
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in started:
        for scenario in (row["scenario"], "All scenarios"):
            groups.setdefault((row["configuration"], scenario), []).append(row)
    order = {name: index for index, name in enumerate(CONFIGURATIONS)}
    outcomes, refused = [], []
    for (configuration, scenario), group in sorted(groups.items(), key=lambda item: (
            order.get(item[0][0], len(order)), item[0][0], item[0][1] == "All scenarios", item[0][1])):
        counts = {name: sum(row.get("outcome") == name for row in group) for name in OUTCOME_CLASSES}
        outcomes.append({"configuration": configuration, "scenario": scenario, "started": len(group), **counts,
                         "not audited": len(group) - sum(counts.values()),
                         "median tokens": _median([row.get("tokens") for row in group]),
                         "median seconds": _median([row.get("elapsed_seconds") for row in group]),
                         "median lookups": _median([row.get("lookups") for row in group])})
        # Read from the rows, so batches recorded under earlier configuration names keep their rejections.
        if any(isinstance(row.get("rejections_ra_feasibility"), int) for row in group):
            refused.append({"configuration": configuration, "scenario": scenario, **{
                check: sum(row.get(f"rejections_{check}") for row in group
                           if isinstance(row.get(f"rejections_{check}"), int)) for check in REJECTION_CHECKS},
                "revisions or rewrites": sum(row.get(key) for row in group for key in ("revisions", "rewrites")
                                             if isinstance(row.get(key), int))})
    return {"outcomes": outcomes, "rejections": refused}


def rows_csv(rows: list[dict]) -> str:
    """Return attempt rows as CSV text, one column per recorded measurement."""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=ROW_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_batch(path: Path, *, live: str | None = None) -> dict:
    """Read a saved batch; one left running without a live runner reads as interrupted.

    Args:
        path: The batch's ``batch.json``.
        live: The batch this process is running, if any.
    """
    batch = json.loads(path.read_text(encoding="utf-8"))
    if batch.get("status") == "running" and batch.get("batch_id") != live:
        batch["status"] = "interrupted"
        batch["rows"] = [_settled(row, "interrupted") if row.get("status") == "running" else row
                         for row in batch.get("rows") or []]
    return batch


def _settled(row: dict, status: str) -> dict:
    """Rebuild a row left running from its records: a draft that ended as an outline keeps or awaits its audit."""
    path = Path(str(row.get("draft_path")))
    return attempt_row({**row, "status": status}, path.parent if path.is_file() else None)


def list_batches(root: Path = EXPERIMENTS, *, live: str | None = None) -> list[dict]:
    """Return the saved batches, newest first, with their status and attempt counts."""
    batches = []
    for path in sorted(root.glob(f"batch_*/{BATCH_NAME}"), reverse=True):
        try:
            batch = read_batch(path, live=live)
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable experiment batch %s: %s", path, exc)
            continue
        batches.append({"batch_id": batch.get("batch_id"), "created_at": batch.get("created_at"),
                        "status": batch.get("status"), "planned": len(batch.get("plan") or []),
                        "done": sum(row.get("status") in _ENDED for row in batch.get("rows") or []),
                        "path": str(path)})
    return batches


class OutlineExperiments:
    """Run one batch at a time in this process, one attempt after another.

    Each attempt prepares and audits in a worker thread and generates on its own event loop in a worker
    thread, as a fresh Recovery draft does, so the UI stays responsive.
    """

    def __init__(self, root: Path = EXPERIMENTS) -> None:
        """Keep batches under ``root``."""
        self.root = root
        self.message = ""
        self.revision = 0
        self._task: asyncio.Task | None = None
        self._draft = None
        self._stopping = False
        self._starting = False
        self._batch: dict | None = None
        self._current: dict = {}

    @property
    def running(self) -> bool:
        """Whether a batch or an audit run is starting or in progress."""
        return self._starting or (self._task is not None and not self._task.done())

    @property
    def live_batch(self) -> str | None:
        """The batch this process is running, if any."""
        return self._batch["batch_id"] if self.running and self._batch else None

    def snapshot(self) -> dict:
        """Return the progress of the running batch from simple fields only."""
        current = dict(self._current)
        draft = self._draft
        if draft is not None:
            current.update(provider_calls=draft.record.get("provider_calls"), draft_message=draft.record.get("message"))
        batch = self._batch or {}
        return {"running": self.running, "batch_id": batch.get("batch_id"), "status": batch.get("status"),
                "planned": len(batch.get("plan") or []),
                "done": sum(row.get("status") in _ENDED for row in batch.get("rows") or []),
                "message": self.message, "current": current, "revision": self.revision}

    async def start(self, settings: dict, *, captures: dict[str, str] | None = None) -> dict:
        """Freeze a new batch and start its attempts in the background.

        Args:
            settings: The batch settings (see ``checked_settings``).
            captures: Retained capture paths by scenario; the four retained captures by default.

        Returns:
            The runner's snapshot.

        Raises:
            ValueError: A batch is running, the settings are invalid, or code changed after the UI started.
        """
        if self.running:
            raise ValueError("A batch is already running; stop it first")
        # Set before the first await, so a second start cannot pass the check while this one freezes its inputs.
        self._starting, self._stopping = True, False
        try:
            if captures is None:
                from cais_spade_llm.recovery_framework.part_slippage_example import (
                    default_observation_paths,
                )

                captures = default_observation_paths()
            checked = checked_settings(settings, captures)
            self._batch = await asyncio.to_thread(self._new_batch, checked, captures)
            self.message = f"Started {self._batch['batch_id']}: {len(self._batch['plan'])} attempts."
            self._task = asyncio.get_running_loop().create_task(self._run(), name="outline_experiments")
            self._task.add_done_callback(self._run_finished)
        finally:
            self._starting = False
        self.revision += 1
        return self.snapshot()

    def stop(self) -> dict:
        """Stop after the provider call in flight; finished attempts stay recorded."""
        if not self.running:
            raise ValueError("No batch is running")
        self._stopping = True
        draft = self._draft
        if draft is not None:
            draft.stop()
        self.message = "Stopping after the current provider call; finished attempts are kept."
        self.revision += 1
        return self.snapshot()

    async def audit_missing(self, batch_id: str) -> dict:
        """Audit the finished attempts of a saved batch that have no audit yet, in the background.

        Raises:
            ValueError: A batch is running, or the batch's frozen code or inputs changed since it ran.
        """
        from cais_spade_llm.recovery_framework.recovery_evaluation import verify_frozen

        if self.running:
            raise ValueError("A batch is already running; stop it first")
        self._starting, self._stopping = True, False
        try:
            batch = read_batch(self.root / batch_id / BATCH_NAME)
            # An audit must see the code and inputs the attempts were generated with.
            await asyncio.to_thread(verify_frozen, batch["frozen"])
            changed = await asyncio.to_thread(changed_since_start, batch["frozen"]["files"])
            if changed:
                raise ValueError(f"Code or inputs changed on disk after the UI started ({len(changed)} files, e.g. "
                                 f"{changed[0]}); restart the UI before auditing")
            self._batch = batch
            self.message = f"Auditing the unaudited attempts of {batch_id}."
            self._task = asyncio.get_running_loop().create_task(self._audit_missing(),
                                                                name="outline_experiment_audits")
            self._task.add_done_callback(self._audits_finished)
        finally:
            self._starting = False
        self.revision += 1
        return self.snapshot()

    def _new_batch(self, settings: dict, captures: dict[str, str]) -> dict:
        from cais_spade_llm.recovery_framework.recovery_case_safety import (
            default_cca_assumptions,
            default_safety_files,
        )
        from cais_spade_llm.recovery_framework.recovery_evaluation import (
            frozen_files,
            implementation_inventory,
        )

        scenarios = settings["scenarios"]
        files = frozen_files([{"source": captures[scenario]} for scenario in scenarios])
        changed = changed_since_start(files)
        if changed:
            raise ValueError(f"Code or inputs changed on disk after the UI started ({len(changed)} files, e.g. "
                             f"{changed[0]}); restart the UI before starting a batch")
        now = datetime.now(timezone.utc)
        batch_id = f"batch_{now:%Y%m%dT%H%M%SZ}_{secrets.token_hex(3)}"
        seed = secrets.randbits(32)
        batch = {
            "version": 1, "kind": "outline_experiment_batch", "study": "Study 1: ablation",
            "batch_id": batch_id, "created_at": now.isoformat(), "status": "running", "message": "",
            "pid": os.getpid(), "settings": deepcopy(settings),
            "captures": {scenario: captures[scenario] for scenario in scenarios},
            "configurations": {name: deepcopy(CONFIGURATIONS[name]) for name, count in settings["attempts"].items()
                               if count},
            "held_constant": {
                "token_budget": "not set", "lookup_limit": "not set", "candidate_proposal_budget": 5,
                "safety_files": {scenario: default_safety_files(scenario) for scenario in scenarios},
                "cca_assumptions": {scenario: default_cca_assumptions(scenario) for scenario in scenarios}},
            "seed": seed, "plan": planned_attempts(scenarios, settings["attempts"], seed), "rows": [],
            "frozen": {"files": files, "implementation_files": implementation_inventory()}, "git": _git_state(),
        }
        (self.root / batch_id).mkdir(parents=True)
        write_record(self.root / batch_id / BATCH_NAME, batch)
        return batch

    def _save(self) -> None:
        batch = self._batch
        batch["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_record(self.root / batch["batch_id"] / BATCH_NAME, batch)

    def _run_finished(self, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is None:
            return
        logger.error("Outline experiment batch failed", exc_info=task.exception())
        self.message = f"The batch stopped on an error: {task.exception()}"
        if self._batch is not None:
            self._batch["rows"] = [{**_settled(row, "failed"), "message": self.message} if row.get("status") == "running"
                                   else row for row in self._batch["rows"]]
            self._batch.update(status="failed", message=self.message)
            self._save()
        self._current = {}
        self.revision += 1

    def _audits_finished(self, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is None:
            return
        logger.error("Outline experiment audits failed", exc_info=task.exception())
        self.message = f"Auditing stopped on an error: {task.exception()}"
        self._current = {}
        self.revision += 1

    async def _run(self) -> None:
        from cais_spade_llm.recovery_framework.recovery_evaluation import verify_frozen

        batch = self._batch
        for attempt in batch["plan"]:
            if self._stopping:
                break
            self._current = {**attempt, "stage": "checking inputs", "started": time.time()}
            try:
                await asyncio.to_thread(verify_frozen, batch["frozen"])
            except ValueError as exc:
                batch.update(status="inputs_changed", message=f"Stopped before {attempt['attempt_id']}: {exc}")
                break
            if self._stopping or not await self._run_attempt(attempt):
                break
        else:
            batch.update(status="finished", message=f"Finished {len(batch['rows'])} attempts.")
        if self._stopping and batch["status"] in {"running", "finished"}:
            batch.update(status="stopped", message="Stopped by the operator; finished attempts are kept.")
        done = {row["attempt_id"] for row in batch["rows"]}
        batch["rows"] += [attempt_row({**attempt, "status": "not started"}, None)
                          for attempt in batch["plan"] if attempt["attempt_id"] not in done]
        self._current = {}
        self.message = batch["message"]
        self._save()
        self.revision += 1

    async def _run_attempt(self, attempt: dict) -> bool:
        """Run, audit and record one attempt; return whether the batch may go on."""
        from cais_spade_llm.recovery_framework.outline_audit import audit_draft
        from cais_spade_llm.recovery_framework.recovery_evaluation import verify_frozen

        batch = self._batch
        self._current["stage"] = "preparing"
        try:
            observations, draft = await asyncio.to_thread(self._prepare, attempt)
        except (ValueError, OSError, LookupError, RuntimeError) as exc:
            logger.warning("Experiment attempt %s could not start: %s", attempt["attempt_id"], exc)
            batch["rows"].append(attempt_row({**attempt, "status": "failed", "error": str(exc)}, None))
            batch.update(status="failed", message=f"{attempt['attempt_id']} could not start: {exc}")
            return False
        if self._stopping:
            # Stopped while this attempt was prepared: it never started, and its draft has written nothing.
            return False
        directory = draft.directory
        names = {jid: name for name, jid in (observations.get("jids") or {}).items()}
        batch.setdefault("resource_names", {})[attempt["scenario"]] = names
        batch["rows"].append({**attempt_row({**attempt, "status": "running"}, None),
                              "draft_path": str(directory / "draft.json")})
        self._save()
        self._draft = draft
        self._current["stage"] = "generating"
        await asyncio.to_thread(lambda: asyncio.run(draft.generate()))
        self._draft = draft = None
        self._current["stage"] = "auditing"
        scenario = attempt["scenario"]
        await asyncio.to_thread(audit_draft, directory, observations, batch["held_constant"]["safety_files"][scenario],
                                batch["held_constant"]["cca_assumptions"][scenario])
        observations = None
        row = attempt_row({**attempt, "status": "finished"}, directory)
        batch["rows"][-1] = row
        go_on = True
        try:
            await asyncio.to_thread(verify_frozen, batch["frozen"])
        except ValueError as exc:
            row["message"] = f"Inputs changed during this attempt: {exc}"
            batch.update(status="inputs_changed", message=f"Stopped after {attempt['attempt_id']}: {exc}")
            go_on = False
        self._save()
        self.revision += 1
        return go_on

    def _prepare(self, attempt: dict) -> tuple[dict, Any]:
        from cais_spade_llm.recovery_framework.outline_draft import OutlineDraft
        from cais_spade_llm.recovery_framework.recovery_observations import read_observations

        batch, scenario = self._batch, attempt["scenario"]
        observations = read_observations(batch["captures"][scenario])
        if observations.get("scenario") != scenario:
            raise ValueError(f"The capture for {scenario} records {observations.get('scenario')}")
        flags = CONFIGURATIONS[attempt["configuration"]]
        settings = batch["settings"]
        draft = OutlineDraft(observations, batch["held_constant"]["safety_files"][scenario],
                             root=self.root / batch["batch_id"] / "attempts", model=settings["model"],
                             reasoning_effort=settings["reasoning_effort"], max_turns=settings["max_turns"],
                             cca_assumptions=(batch["held_constant"]["cca_assumptions"][scenario]
                                              if flags.get("check_cca_safety") else None), **flags)
        return observations, draft

    async def _audit_missing(self) -> None:
        from cais_spade_llm.recovery_framework.outline_audit import audit_draft
        from cais_spade_llm.recovery_framework.recovery_observations import read_observations

        batch = self._batch
        audited = 0
        for index, row in enumerate(batch["rows"]):
            path = Path(str(row.get("draft_path")))
            if self._stopping or row.get("outcome") != "not audited" or not path.is_file():
                continue
            scenario = row["scenario"]
            self._current = {**row, "stage": "auditing", "started": time.time()}
            observations = await asyncio.to_thread(read_observations, batch["captures"][scenario])
            await asyncio.to_thread(audit_draft, path.parent, observations,
                                    batch["held_constant"]["safety_files"][scenario],
                                    batch["held_constant"]["cca_assumptions"][scenario])
            batch["rows"][index] = attempt_row(row, path.parent)
            audited += 1
            self._save()
            self.revision += 1
        self._current = {}
        self.message = f"Audited {audited} attempts of {batch['batch_id']}."
        self.revision += 1


_RUNNER = OutlineExperiments()


def outline_experiments() -> OutlineExperiments:
    """Return this process's batch runner, shared by every page client."""
    return _RUNNER


def stop_running() -> None:
    """Stop a running batch at shutdown; its finished attempts are already recorded."""
    if _RUNNER.running:
        _RUNNER.stop()
