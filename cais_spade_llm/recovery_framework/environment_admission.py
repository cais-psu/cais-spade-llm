"""Atomic CCA admission for recovery environmental execution.

Only validated ProductAgent acknowledgement records advance the live monitors.
Searches use detached copies; permission is committed under the context lock.
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from cais_spade_llm.agents.central_controller.local_composition import (
    Analysis, AnalysisLimit, Budget, IncompleteModel, Node, Scope, analyze, state_key,
)
from cais_spade_llm.product.environment import fingerprint
from cais_spade_llm.recovery_framework.environment_composition import (
    EnvironmentPlant, detached_checker, state_labels,
)

logger = logging.getLogger(__name__)


@dataclass
class CachedComposition:
    """A complete graph, its objective and exact nominal model identity."""

    plant: EnvironmentPlant
    analysis: Analysis
    model_identity: str


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
        self.components: list[CachedComposition] = []
        self.epoch = 0
        self.ack_cursor = 0
        self.last_results: dict[str, dict] = {}
        self.invalid_reason = ""
        runtime.admission = self
        # Context invokes this CCA-owned consumer only after committing validated
        # values, under the same transaction lock as permission commits.
        self.context.acknowledgement_observer = self.synchronize

    def invalidate(self, reason: str) -> None:
        """Discard affected proofs after a run failure without resetting history."""
        with self.context.admission_lock:
            self.components.clear()
            self.invalid_reason = reason
            self.epoch += 1

    def _running_labels(self) -> set[str]:
        return {label for action in self.grants.values() for label in action.labels}

    def synchronize(self) -> None:
        """Consume ordered committed acknowledgements exactly once."""
        with self.context.admission_lock:
            while self.ack_cursor < len(self.context.transitions):
                record = self.context.transitions[self.ack_cursor]
                task = record["acknowledgement"]
                action = self.grants.pop(task["task_id"], None)
                if action is None:
                    self.invalid_reason = "acknowledgement_without_admission"
                    self.components.clear()
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
                for rule, current in self.monitor.current_states.items():
                    evidence = self.monitor.transition_evidence(rule, current, sigma)
                    if evidence["status"] != "passed":
                        self.monitor.history_error = evidence
                        self.invalid_reason = "unexpected_completion"
                        self.components.clear()
                    next_states[rule] = evidence["to"]
                self.monitor.current_states = next_states
                self.monitor.running_aps = self._running_labels()
                for rid, values in record["after"].items():
                    jid = self.runtime.jids[rid]
                    self.monitor.resource_state_aps[jid] = set(state_labels(
                        self.monitor, {rid: values}, record["product_after"],
                        self.runtime.jids, self.contexts))
                    self.monitor.resource_states[jid] = {
                        "current_state": values.get("resource_state", ""),
                        "params": {**self.contexts.get(rid, {}), **values},
                    }
                self.ack_cursor += 1
                self.epoch += 1
            # Retire a finished operation only once its scoped release condition holds.
            keep = []
            for component in self.components:
                plant, analysis = component.plant, component.analysis
                state = {"resources": self.context.snapshot(),
                         "products": self.context.part_tracker, "contexts": self.contexts}
                relevant = [action for action in self.grants.values()
                            if self._interacts(action, analysis.scope)]
                if not relevant and plant.complete(state, analysis.scope):
                    for part in analysis.scope.products:
                        self.goals.pop(part, None)
                else:
                    keep.append(component)
            self.components = keep

    @staticmethod
    def _interacts(action, scope) -> bool:
        part = action.task["parameters"].get("part_name") or action.task.get("part_name")
        return part in scope.products or bool(action.claims & scope.claims) or bool(
            action.claims & {f"resource:{rid}" for rid in scope.resources})

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

    async def check(self, task: dict, *, commit: bool, full: bool = False,
                    budget: Budget | None = None) -> dict:
        """Analyze outside the lock, then compare revision and atomically grant.

        Args:
            task: Exact prepared task, including run and task identity.
            commit: Resource permission commits a grant; plan checks only warm caches.
            full: Use the reference selection for controlled small-model experiments.
            budget: Optional deterministic budget for tests.

        Returns:
            Structured allowed, held or inconclusive evidence.
        """
        budget = budget or Budget()
        task_id = task.get("task_id", "")
        with self.context.admission_lock:
            self.synchronize()
            revision = self._revision()
            if self.invalid_reason or self.monitor.history_error or self.runtime.stopped:
                return self._inconclusive(task_id, self.invalid_reason or
                                          "monitor_history_unavailable", revision)
            if (self.context.pending_for(task_id) != task or task.get("run_id") != self.context.run_id
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
            components = list(self.components)

        def search():
            plant = EnvironmentPlant(snapshot, checker, self.runtime.jids, goals,
                                     [action.task for action in running] +
                                     [component.plant.proposed.task for component in components])
            plant.initial["contexts"] = contexts
            candidate = plant.action(task)
            # Force the proposal's exact task identity into generated event bindings.
            plant.proposed = candidate
            model_identity = fingerprint([
                snapshot.models, {rid: sorted(actor.executors)
                                  for rid, actor in snapshot.resources.items()},
                checker.safety_rules, getattr(checker, "dfa_dots", checker.dfas),
                snapshot.requirements, snapshot.geometry, snapshot.permitted_resources,
            ])
            for component in components:
                if full or component.model_identity != model_identity:
                    continue
                old = component.analysis
                # Re-select so a new operation cannot borrow a proof with fewer obligations.
                scope = plant.select(candidate, running, checker, budget, full=False)
                if (scope.resources != old.scope.resources or scope.products != old.scope.products
                        or scope.rules != old.scope.rules or scope.goals != old.scope.goals
                        or scope.claims != old.scope.claims):
                    continue
                relevant = tuple(action for action in running if self._interacts(action, scope))
                node = Node(plant.initial, relevant, tuple(monitors[r] for r in sorted(scope.rules)))
                root = state_key(plant, scope, node)
                if root in old.graph and any(edge.kind == "start" and edge.action == candidate.key
                                            for edge in old.graph[root]):
                    # A graph edge is reusable only with identical AP and claim bindings.
                    target = next(edge.target for edge in old.graph[root]
                                  if edge.kind == "start" and edge.action == candidate.key)
                    successor = old.nodes.get(target)
                    previous = next((item for item in successor.running
                                     if item.key == candidate.key), None) if successor else None
                    if previous is None or (previous.labels, previous.claims) != (
                            candidate.labels, candidate.claims):
                        continue
                    result = Analysis(old.decision(root, candidate), "", scope, root,
                                      old.graph, old.nodes, old.winning,
                                      budget.clock() - budget.started)
                    result.reason = "" if result.status == "allowed" else "no_joint_completion"
                    budget.check()
                    return plant, candidate, result, model_identity, True
            result = analyze(plant, checker, monitors, candidate, running,
                             budget=budget, full=full)
            if result.status == "held" and plant.missing_capabilities:
                result.status = "inconclusive"
                result.reason = "missing_required_capability"
                result.scope.reasons.append({"kind": "unavailable_behavior",
                                             "capabilities": sorted(plant.missing_capabilities)})
            return plant, candidate, result, model_identity, False

        try:
            plant, candidate, analysis, identity, cache_hit = await asyncio.to_thread(search)
        except (IncompleteModel, KeyError, StopIteration) as exc:
            with self.context.admission_lock:
                return self._inconclusive(task_id, "missing_model_or_binding: " + str(exc), revision)
        # A cache-selection deadline is subject to the same fail-closed limit as search.
        except AnalysisLimit as exc:
            with self.context.admission_lock:
                return self._inconclusive(task_id, str(exc), revision)
        with self.context.admission_lock:
            self.synchronize()
            result = analysis.evidence(revision, cache_hit=cache_hit)
            result["analysis_time_sec"] = budget.clock() - budget.started
            if result["analysis_time_sec"] >= budget.seconds:
                result.update(status="inconclusive", reason="time_limit")
                return self._result(task_id, result)
            if (self._revision() != revision or self.context.pending_for(task_id) != task
                    or self.invalid_reason):
                result.update(status="inconclusive", reason="stale_snapshot")
                return self._result(task_id, result)
            if analysis.status == "allowed":
                # Publish only successful replacements, preserving other components.
                scope = analysis.scope
                self.components = [component for component in self.components
                                   if not (scope.resources & component.analysis.scope.resources
                                           or scope.products & component.analysis.scope.products
                                           or scope.rules & component.analysis.scope.rules
                                           or scope.claims & component.analysis.scope.claims)]
                self.components.append(CachedComposition(plant, analysis, identity))
                if commit:
                    self.grants[task_id] = candidate
                    for goal in scope.goals:
                        if "part_name" in goal:
                            self.goals[goal["part_name"]] = deepcopy(goal["operation"])
                    self.monitor.running_aps = self._running_labels()
                    self.epoch += 1
                    result["admitted_epoch"] = self.epoch
            return self._result(task_id, result)
