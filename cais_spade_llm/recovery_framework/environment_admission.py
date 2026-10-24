"""Atomic CCA admission for recovery environmental execution.

Only validated ProductAgent acknowledgement records advance the live monitors.
A start is admitted as Eq. 13 defines safety validation: its labels and its own
projected finish keep an accepting state reachable in every specification
monitor. Checks use detached copies; permission is committed under the context lock.
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.central_controller.local_composition import (
    Action,
    Analysis,
    AnalysisLimit,
    Budget,
    IncompleteModel,
    Scope,
)
from cais_spade_llm.product.environment import fingerprint
from cais_spade_llm.recovery_framework.environment_composition import (
    EnvironmentPlant,
    detached_checker,
    state_labels,
    task_key,
)

logger = logging.getLogger(__name__)


def specification_check(plant: EnvironmentPlant, checker, monitors: dict[str, str], candidate: Action,
                        running: tuple[Action, ...], budget: Budget) -> Analysis:
    """Check one start as Eq. 13 defines safety validation.

    The start extends the trace with its labels and projected state, and so does its own finish,
    which cannot be disabled once the activity starts. Both steps must keep an accepting state
    reachable in every specification monitor. A start that needs a reservation held by running
    work is held. There is no search over the plant's future: Eq. 13 checks the monitors, not that
    every resource can still finish, so other running activities' later finishes are checked when
    they are observed.

    Args:
        plant: Detached nominal plant at the checked revision.
        checker: Detached specification monitors.
        monitors: Each rule's current monitor state.
        candidate: The start to check.
        running: Granted activities still running.
        budget: Time limit for the check.

    Returns:
        An allowed or held analysis with its rule checks as evidence.
    """
    rules = sorted(monitors)
    scope = Scope(rules=set(rules), claims=set(candidate.claims), tasks={action.key for action in running})

    def result(status: str, reason: str, **fields) -> Analysis:
        return Analysis(status, reason, scope, elapsed=budget.clock() - budget.started, **fields)

    conflicts = sorted(candidate.claims & frozenset(claim for action in running for claim in action.claims))
    if conflicts:
        scope.reasons.append({"kind": "running_reservation", "reservations": conflicts})
        return result("held", "running_reservation_conflict")
    try:
        after = plant.project(plant.initial, candidate)
    except ValueError as exc:
        return result("held", "start_not_enabled: " + str(exc))
    running_labels = frozenset(label for action in running for label in action.labels)
    states = [monitors[rule] for rule in rules]
    steps = {
        "start": running_labels | plant.labels(plant.initial) | candidate.labels | plant.labels(after),
        "finish": running_labels | plant.labels(after) | candidate.labels,
    }
    for event, sigma in steps.items():
        budget.check()
        checks = [checker.transition_evidence(rule, state, sigma) for rule, state in zip(rules, states, strict=True)]
        incomplete = [row for row in checks if row["reason"] in {"dfa_missing_transition", "dfa_ambiguous_transition"}]
        if incomplete:
            raise IncompleteModel(incomplete[0]["reason"] + ": " + incomplete[0]["rule_id"])
        scope.reasons.append({"kind": event + "_specification_checks", "rules": len(checks)})
        rejected = [row for row in checks if row["status"] != "passed"]
        if rejected:
            scope.reasons.append({"kind": "specification_violation", "event": event, "transitions": rejected})
            return result("held", "specification_violation",
                          witness=[{"action": candidate.key, "event": event, "rule_checks": rejected}])
    return result("allowed", "")


class EnvironmentAdmission:
    """Own run-scoped grants and local analyses beside authoritative CCA history."""

    def __init__(self, runtime, monitor) -> None:
        """Attach one coordinator to one run and its existing compiled monitor."""
        self.runtime = runtime
        self.context = runtime.context
        self.monitor = monitor
        self.grants: dict[str, Any] = {}
        self.goals: dict[str, dict] = {}
        self.contexts: dict[str, dict] = {}
        self.epoch = 0
        # ack_cursor and acknowledgement_history count completions, the monitor ticks; record_cursor
        # reads every committed record, and starts and faults are kept in activity_history.
        self.ack_cursor = 0
        self.record_cursor = 0
        self.acknowledgement_history: list[dict] = []
        self.activity_history: list[dict] = []
        self.last_results: dict[str, dict] = {}
        self._prepared_admissions: dict[str, dict] = {}
        self.invalid_reason = ""
        runtime.admission = self
        # Context invokes this CCA-owned consumer only after committing validated
        # values, under the same transaction lock as permission commits.
        self.context.acknowledgement_observer = self.synchronize

    def invalidate(self, reason: str) -> None:
        """Discard affected proofs after a run failure without resetting history."""
        with self.context.admission_lock:
            self._prepared_admissions.clear()
            self.invalid_reason = reason
            self.epoch += 1

    def _running_labels(self) -> set[str]:
        return {label for action in self.grants.values() for label in action.labels}

    def _observe(self, record: dict, task: dict) -> None:
        for rid in self.context._task_participants(task):
            self.contexts[rid] = {**task["parameters"], "task_id": task["task_id"]}
        for rid, values in record["after"].items():
            jid = self.runtime.jids[rid]
            self.monitor.resource_state_aps[jid] = set(state_labels(
                self.monitor, {rid: values}, record["product_after"],
                self.runtime.jids, self.contexts))
            self.monitor.resource_states[jid] = {
                "current_state": values.get("resource_state", ""),
                "params": {**self.contexts.get(rid, {}), **values},
            }

    def synchronize(self) -> None:
        """Consume ordered committed acknowledgements exactly once."""
        with self.context.admission_lock:
            while self.record_cursor < len(self.context.transitions):
                record = self.context.transitions[self.record_cursor]
                self.record_cursor += 1
                task = record["acknowledgement"]
                before_states = deepcopy(self.monitor.current_states)
                kind = record.get("kind", "finish")
                if kind != "finish":
                    # A start or a fault keeps the action's grant and steps no specification monitor,
                    # as the local composition steps them only when an action is done.
                    self._observe(record, task)
                    self.activity_history.append({
                        "record": self.record_cursor, "run_id": self.context.run_id, "kind": kind,
                        "task_id": task["task_id"], "task": deepcopy(task),
                        "admitted": task["task_id"] in self.grants, "states": before_states,
                    })
                    self.epoch += 1
                    self._prepared_admissions.clear()
                    continue
                action = self.grants.pop(task["task_id"], None)
                # An observed event commands nothing, so it is acknowledged without a grant.
                model = (getattr(self.context, "models", None) or {}).get(task["resource_id"]) or {}
                observed = any(event.get("event_id") == task.get("event_id") and event.get("controllable") is False
                               for event in model.get("events", []))
                if action is None and not observed:
                    self.invalid_reason = "acknowledgement_without_admission"
                labels = action.labels if action else frozenset(self.monitor._map_task_to_aps(
                    self.runtime.jids[task["resource_id"]], task["event_name"],
                    {**task["parameters"], "task_id": task["task_id"]}))
                for rid in self.context._task_participants(task):
                    self.contexts[rid] = {**task["parameters"], "task_id": task["task_id"]}
                persistent = state_labels(self.monitor, record["after"],
                                          record["product_after"], self.runtime.jids,
                                          self.contexts)
                sigma = frozenset(self._running_labels()) | persistent | labels
                next_states = {}
                rule_checks = []
                for rule, current in self.monitor.current_states.items():
                    evidence = self.monitor.transition_evidence(rule, current, sigma)
                    rule_checks.append(deepcopy(evidence))
                    if evidence["status"] != "passed":
                        self.monitor.history_error = evidence
                        self.invalid_reason = "unexpected_completion"
                    next_states[rule] = evidence["to"]
                self.monitor.current_states = next_states
                self.monitor.running_aps = self._running_labels()
                self._observe(record, task)
                self.ack_cursor += 1
                self.acknowledgement_history.append({
                    "cursor": self.ack_cursor, "run_id": self.context.run_id,
                    "task_id": task["task_id"], "task": deepcopy(task),
                    "admitted": action is not None, "before_states": before_states,
                    "after_states": deepcopy(next_states), "rule_checks": rule_checks,
                })
                self.epoch += 1
                self._prepared_admissions.clear()

    def _revision(self) -> dict:
        return {
            "run_id": self.context.run_id, "revision": self.context.revision,
            "resources": self.context.revisions(), "admission_epoch": self.epoch,
            "unavailable_resources": sorted(self.context.unavailable_resources),
            "stopped": self.runtime.stopped,
            "context_identity": fingerprint([
                self.context.part_tracker, self.context.requirements, self.context.geometry,
                getattr(self.runtime, "operation_goals", {}), self.context.permitted_resources,
                self.monitor.current_states,
            ]),
        }

    def _result(self, task_id: str, result: dict) -> dict:
        self.last_results[task_id] = deepcopy(result)
        self.context.negotiations.append({"kind": "local_composition", "task_id": task_id,
                                         **deepcopy(result)})
        return result

    def _inconclusive(self, task_id: str, reason: str, revision=None) -> dict:
        return self._result(task_id, Analysis("inconclusive", reason, Scope()).evidence(revision))

    def _prepare_composition_start(self, task: dict) -> dict:
        """Retain a start after CCA has validated its winning composition edge.

        This internal entry is only for the composition commit callback, under
        its existing admission lock. The caller owns the completed graph proof;
        this method binds the pending task to the same grant transaction as
        ``check`` without running another search or accepting a caller's Action.
        """
        task = deepcopy(task)
        with self.context.admission_lock:
            self.synchronize()
            task_id = task.get("task_id", "")
            revision = self._revision()
            if self.invalid_reason or self.monitor.history_error or self.runtime.stopped:
                return self._inconclusive(task_id, self.invalid_reason or
                                          "monitor_history_unavailable", revision)
            if (fingerprint(self.context.pending_for(task_id)) != fingerprint(task)
                    or task.get("run_id") != self.context.run_id
                    or not self.context.relevant_revisions_match(task)):
                return self._inconclusive(task_id, "stale_candidate", revision)
            if task_id in self.grants:
                return self._inconclusive(task_id, "prepared_admission_unavailable", revision)
            part = task["parameters"].get("part_name") or task.get("part_name")
            claims = frozenset(self.context._task_reservations(task, part))
            labels = frozenset(self.monitor._map_task_to_aps(
                self.runtime.jids[task["resource_id"]], task["event_name"], task["parameters"]))
            candidate = Action(task_key(task), task, claims, labels)
            result = Analysis("allowed", "", Scope()).evidence(revision)
            self._prepared_admissions[task_id] = {
                "task": deepcopy(task), "candidate": candidate, "goals": [],
                "verdict": deepcopy(result), "fingerprint": fingerprint(result),
            }
            return result

    def commit_prepared(self, task: dict, verdict: dict) -> dict:
        """Commit an unchanged owner-retained preview under the admission lock.

        Args:
            task: Exact pending task used for the successful preview.
            verdict: Unchanged result of ``check(commit=False)`` or the internal
                winning-composition start preparation.

        Returns:
            Committed allowed evidence, or inconclusive evidence without a grant.
            Replaying a consumed preview cannot commit a second time.
        """
        with self.context.admission_lock:
            self.synchronize()
            task_id = task.get("task_id", "")
            revision = self._revision()
            if self.invalid_reason or self.monitor.history_error or self.runtime.stopped:
                return self._inconclusive(task_id, self.invalid_reason or
                                          "monitor_history_unavailable", revision)
            if (fingerprint(self.context.pending_for(task_id)) != fingerprint(task)
                    or task.get("run_id") != self.context.run_id
                    or not self.context.relevant_revisions_match(task)):
                return self._inconclusive(task_id, "stale_candidate", revision)
            prepared = self._prepared_admissions.get(task_id)
            if prepared is None or task_id in self.grants:
                return self._inconclusive(task_id, "prepared_admission_unavailable", revision)
            if fingerprint(prepared["task"]) != fingerprint(task):
                return self._inconclusive(task_id, "stale_candidate", revision)
            if prepared["verdict"]["snapshot_revision"] != revision:
                return self._inconclusive(task_id, "stale_snapshot", revision)
            try:
                unchanged = isinstance(verdict, dict) and fingerprint(verdict) == prepared["fingerprint"]
            except (TypeError, ValueError):
                unchanged = False
            if not unchanged:
                return self._inconclusive(task_id, "prepared_verdict_changed", revision)
            self.grants[task_id] = prepared["candidate"]
            for goal in prepared["goals"]:
                if "part_name" in goal:
                    self.goals[goal["part_name"]] = deepcopy(goal["operation"])
            self.monitor.running_aps = self._running_labels()
            self.epoch += 1
            result = deepcopy(prepared["verdict"])
            result["admitted_epoch"] = self.epoch
            self._prepared_admissions.clear()
            return self._result(task_id, result)

    async def check(self, task: dict, *, commit: bool, budget: Budget | None = None) -> dict:
        """Check outside the lock, then compare revision and atomically grant.

        Args:
            task: Exact prepared task, including run and task identity.
            commit: Resource permission commits a grant; plan checks only warm caches.
            budget: Explicit time limit, otherwise the run's configured budget or the default.

        Returns:
            Structured allowed, held or inconclusive evidence.
        """
        task = deepcopy(task)
        budget = budget or Budget(**getattr(self.runtime, "composition_budget", {}))
        task_id = task.get("task_id", "")
        with self.context.admission_lock:
            self.synchronize()
            revision = self._revision()
            if self.invalid_reason or self.monitor.history_error or self.runtime.stopped:
                return self._inconclusive(task_id, self.invalid_reason or
                                          "monitor_history_unavailable", revision)
            if (fingerprint(self.context.pending_for(task_id)) != fingerprint(task)
                    or task.get("run_id") != self.context.run_id
                    or not self.context.relevant_revisions_match(task)):
                return self._inconclusive(task_id, "stale_candidate", revision)
            if task_id in self.grants:
                result = deepcopy(self.last_results[task_id])
                result.update(status="allowed", reason="already_admitted",
                              snapshot_revision=revision, cache_hit=True)
                return result
            snapshot = self.context.calculation_snapshot()
            checker = detached_checker(self.monitor)
            monitors = dict(self.monitor.current_states)
            running = tuple(self.grants.values())
            goals = deepcopy(self.goals)
            goals.update(deepcopy(getattr(self.runtime, "operation_goals", {})))
            contexts = deepcopy(self.contexts)

        def check_start():
            plant = EnvironmentPlant(snapshot, checker, self.runtime.jids, goals,
                                     [action.task for action in running])
            plant.initial["contexts"] = contexts
            candidate = plant.action(task)
            return candidate, specification_check(plant, checker, monitors, candidate, running, budget)

        try:
            candidate, analysis = await asyncio.to_thread(check_start)
        except (IncompleteModel, KeyError, StopIteration) as exc:
            with self.context.admission_lock:
                return self._inconclusive(task_id, "missing_model_or_binding: " + str(exc), revision)
        except AnalysisLimit as exc:
            with self.context.admission_lock:
                return self._inconclusive(task_id, str(exc), revision)
        with self.context.admission_lock:
            self.synchronize()
            result = analysis.evidence(revision)
            result["analysis_time_sec"] = budget.clock() - budget.started
            if result["analysis_time_sec"] >= budget.seconds:
                result.update(status="inconclusive", reason="time_limit")
                return self._result(task_id, result)
            if (self._revision() != revision
                    or fingerprint(self.context.pending_for(task_id)) != fingerprint(task)
                    or self.invalid_reason):
                result.update(status="inconclusive", reason="stale_snapshot")
                return self._result(task_id, result)
            if analysis.status == "allowed":
                self._prepared_admissions[task_id] = {
                    "task": deepcopy(task), "candidate": deepcopy(candidate),
                    "goals": deepcopy(analysis.scope.goals), "verdict": deepcopy(result),
                    "fingerprint": fingerprint(result),
                }
                if commit:
                    return self.commit_prepared(task, result)
            return self._result(task_id, result)
