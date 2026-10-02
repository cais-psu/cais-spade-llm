"""Bounded nonblocking analysis of local tasks and persistent specification DFAs."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker

logger = logging.getLogger(__name__)


class IncompleteModel(ValueError):
    """Required behavior is not established by the supplied nominal model."""


class AnalysisLimit(RuntimeError):
    """Dependency selection, exploration or synthesis exhausted its budget."""


@dataclass
class Budget:
    """One deadline shared by dependency selection, exploration and synthesis."""

    max_states: int = 20_000
    seconds: float = 2.0
    clock: Callable[[], float] = time.monotonic
    started: float = field(init=False)

    def __post_init__(self) -> None:
        self.started = self.clock()

    def check(self, states: int = 0) -> None:
        """Stop before publishing any incomplete search result."""
        if states >= self.max_states:
            raise AnalysisLimit("state_limit")
        if self.clock() - self.started >= self.seconds:
            raise AnalysisLimit("time_limit")


@dataclass(frozen=True)
class Action:
    """An exact bound task and its concurrent claims and event propositions."""

    key: str
    task: dict[str, Any]
    claims: frozenset[str]
    labels: frozenset[str]
    controllable: bool = True


@dataclass
class Scope:
    """Closed dependency set and evidence explaining its membership."""

    resources: set[str] = field(default_factory=set)
    products: set[str] = field(default_factory=set)
    rules: set[str] = field(default_factory=set)
    tasks: set[str] = field(default_factory=set)
    reasons: list[dict[str, Any]] = field(default_factory=list)
    goals: list[dict[str, Any]] = field(default_factory=list)
    terminal_rules: set[str] | None = None
    claims: set[str] = field(default_factory=set)
    task_bindings: dict[str, dict[str, Any]] = field(default_factory=dict)

    def evidence(self) -> dict[str, Any]:
        """Return JSON-ready dependency evidence."""
        return {
            "included_resources": sorted(self.resources),
            "included_products": sorted(self.products),
            "included_specifications": sorted(self.rules),
            "included_tasks": sorted(self.tasks),
            "included_reservations": sorted(self.claims),
            "task_bindings": self.task_bindings,
            "dependency_reasons": self.reasons,
            "completion_conditions": self.goals,
        }


class Plant(Protocol):
    """Nominal semantics shared by local and full reference analyses."""

    initial: dict[str, Any]

    def select(self, candidate: Action, running: tuple[Action, ...],
               checker: BaseSafetyChecker, budget: Budget, *, full: bool) -> Scope:
        """Close dependencies before constructing the parallel composition."""
        ...

    def actions(self, state: dict[str, Any], scope: Scope,
                budget: Budget) -> Iterable[Action]:
        """Enumerate bound starts from resource-owned models."""
        ...

    def project(self, state: dict[str, Any], action: Action) -> dict[str, Any]:
        """Validate guards and compute acknowledged nominal effects."""
        ...

    def labels(self, state: dict[str, Any]) -> frozenset[str]:
        """Evaluate persistent propositions with the live monitor semantics."""
        ...

    def key(self, state: dict[str, Any], scope: Scope) -> str:
        """Identify all values that can affect this closed composition."""
        ...

    def complete(self, state: dict[str, Any], scope: Scope) -> bool:
        """Check operation and release conditions without erasing occupancy."""
        ...


StateKey = tuple[str, tuple[str, ...], tuple[str, ...]]


@dataclass
class Node:
    """Copied plant values, running tasks and specification state vector."""

    state: dict[str, Any]
    running: tuple[Action, ...]
    monitors: tuple[str, ...]


@dataclass(frozen=True)
class Edge:
    """A controllable start or an uncontrollable completion."""

    target: StateKey | None
    action: str
    kind: str
    controllable: bool


@dataclass
class Analysis:
    """A complete reusable graph or a fail-closed incomplete result."""

    status: str
    reason: str
    scope: Scope
    root: StateKey | None = None
    graph: dict[StateKey, list[Edge]] = field(default_factory=dict)
    nodes: dict[StateKey, Node] = field(default_factory=dict)
    winning: set[StateKey] = field(default_factory=set)
    elapsed: float = 0.0
    witness: list[dict[str, Any]] = field(default_factory=list)

    def evidence(self, revision: Any, *, cache_hit: bool = False) -> dict[str, Any]:
        """Expose the checked revision and bounded-analysis diagnostics."""
        return {
            "status": self.status, "reason": self.reason, "snapshot_revision": revision,
            **self.scope.evidence(), "explored_states": len(self.nodes),
            "analysis_time_sec": self.elapsed, "cache_hit": cache_hit,
            "counterexample": self.witness,
        }

    def decision(self, root: StateKey, candidate: Action) -> str:
        """Require the candidate's successor to remain nonblocking."""
        if self.status == "inconclusive":
            return "inconclusive"
        for edge in self.graph.get(root, []):
            if edge.kind == "start" and edge.action == candidate.key:
                return "allowed" if edge.target in self.winning else "held"
        return "held"


def state_key(plant: Plant, scope: Scope, node: Node) -> StateKey:
    """Retain all running identities and all selected DFA states in the key."""
    return (plant.key(node.state, scope),
            tuple(sorted(action.key for action in node.running)), node.monitors)


def nonblocking_region(graph: dict[StateKey, list[Edge]], marked: set[StateKey],
                       budget: Budget) -> set[StateKey]:
    """Alternate coaccessibility and uncontrollable-predecessor removal.

    Unsafe completion edges must remain present: a supervisor cannot disable
    an already running action's completion.
    """
    region = set(graph)
    reverse: dict[StateKey, set[StateKey]] = {}
    for source, edges in graph.items():
        budget.check()
        for edge in edges:
            if edge.target is not None:
                reverse.setdefault(edge.target, set()).add(source)
    while region:
        budget.check()
        reachable = marked & region
        queue = deque(reachable)
        while queue:
            budget.check()
            for parent in reverse.get(queue.popleft(), ()):
                if parent in region and parent not in reachable:
                    reachable.add(parent)
                    queue.append(parent)
        retained = set()
        for node in reachable:
            budget.check()
            if all(edge.controllable or edge.target in reachable for edge in graph[node]):
                retained.add(node)
        if retained == region:
            return region
        region = retained
    return set()


def _step(checker: BaseSafetyChecker, rules: list[str], monitors: tuple[str, ...],
          sigma: frozenset[str]) -> tuple[str, ...] | None:
    next_states = []
    for rule, current in zip(rules, monitors):
        evidence = checker.transition_evidence(rule, current, sigma)
        if evidence["reason"] in {"dfa_missing_transition", "dfa_ambiguous_transition"}:
            raise IncompleteModel(evidence["reason"] + ": " + rule)
        if evidence["status"] != "passed":
            return None
        next_states.append(evidence["to"])
    return tuple(next_states)


def _marked(plant: Plant, checker: BaseSafetyChecker, scope: Scope,
            rules: list[str], node: Node) -> bool:
    if node.running or not plant.complete(node.state, scope):
        return False
    selected = [(rule, state) for rule, state in zip(rules, node.monitors)
                if scope.terminal_rules is None or rule in scope.terminal_rules]
    terminal_rules = [rule for rule, _ in selected]
    terminal = _step(checker, terminal_rules, tuple(state for _, state in selected), frozenset())
    return terminal is not None and all(
        state in checker.dfas[rule]["accepting_states"]
        for rule, state in zip(terminal_rules, terminal)
    )


def _edges(plant: Plant, checker: BaseSafetyChecker, scope: Scope,
           rules: list[str], node: Node, budget: Budget) -> Iterable[tuple[Edge, Node | None]]:
    running_labels = frozenset(label for action in node.running for label in action.labels)
    persistent = plant.labels(node.state)
    for action in node.running:
        budget.check()
        remaining = tuple(item for item in node.running if item.key != action.key)
        try:
            after = plant.project(node.state, action)
        except IncompleteModel:
            raise
        except ValueError:
            yield Edge(None, action.key, "done", False), None
            continue
        sigma = plant.labels(after) | action.labels | frozenset(
            label for item in remaining for label in item.labels)
        monitors = _step(checker, rules, node.monitors, sigma)
        successor = Node(after, remaining, monitors) if monitors is not None else None
        yield Edge(state_key(plant, scope, successor) if successor else None,
                   action.key, "done", False), successor
    claims = frozenset(claim for item in node.running for claim in item.claims)
    for action in plant.actions(node.state, scope, budget):
        budget.check()
        if claims & action.claims or any(item.key == action.key for item in node.running):
            continue
        try:
            after = plant.project(node.state, action)
        except IncompleteModel:
            raise
        except ValueError:
            continue
        # Starts check the projected label, but commit neither effects nor DFA history.
        sigma = running_labels | persistent | action.labels | plant.labels(after)
        if _step(checker, rules, node.monitors, sigma) is None:
            if not action.controllable:
                yield Edge(None, action.key, "start", False), None
            continue
        successor = Node(node.state, node.running + (action,), node.monitors)
        yield Edge(state_key(plant, scope, successor), action.key, "start",
                   action.controllable), successor


def analyze(plant: Plant, checker: BaseSafetyChecker, current_states: dict[str, str],
            candidate: Action, running: tuple[Action, ...] = (), *,
            budget: Budget | None = None, full: bool = False) -> Analysis:
    """Check a proposed start against a closed joint composition.

    Args:
        plant: Detached nominal model and acknowledged values.
        checker: Parsed specification transitions, never live mutable history.
        current_states: Authoritative history copied under the admission lock.
        candidate: Exact proposed start.
        running: Granted actions, including grants awaiting observation.
        budget: Shared dependency, exploration and synthesis limits.
        full: Include the full available model for reference comparisons.

    Returns:
        Analysis usable for admission only after an atomic revision check.
    """
    budget = budget or Budget()
    result = Analysis("inconclusive", "analysis_incomplete", Scope())
    try:
        budget.check()
        result.scope = plant.select(candidate, running, checker, budget, full=full)
        rules = sorted(result.scope.rules)
        if any(rule not in current_states for rule in rules):
            raise IncompleteModel("monitor_history_unavailable")
        relevant_running = tuple(action for action in running
                                 if action.key in result.scope.tasks)
        root = Node(plant.initial, relevant_running,
                    tuple(current_states[rule] for rule in rules))
        result.root = state_key(plant, result.scope, root)
        result.nodes[result.root] = root
        queue = deque([result.root])
        marked = set()
        while queue:
            budget.check(len(result.nodes))
            key = queue.popleft()
            node = result.nodes[key]
            result.graph[key] = []
            if _marked(plant, checker, result.scope, rules, node):
                marked.add(key)
            for edge, successor in _edges(plant, checker, result.scope, rules, node, budget):
                result.graph[key].append(edge)
                result.scope.tasks.add(edge.action)
                if successor is None or edge.target in result.nodes:
                    continue
                budget.check(len(result.nodes) + 1)
                result.nodes[edge.target] = successor
                queue.append(edge.target)
        result.winning = nonblocking_region(result.graph, marked, budget)
        result.status = "complete"
        result.status = result.decision(result.root, candidate)
        result.reason = "" if result.status == "allowed" else "no_joint_completion"
        for node in result.nodes.values():
            for action in node.running:
                result.scope.task_bindings[action.key] = action.task
        if result.status == "held":
            start = next((edge for edge in result.graph[result.root]
                          if edge.kind == "start" and edge.action == candidate.key), None)
            result.witness = [{"action": candidate.key, "event": "start"}]
            if start is not None and start.target is not None:
                queue = deque([(start.target, [])])
                seen = {start.target}
                while queue:
                    budget.check()
                    target, path = queue.popleft()
                    unsafe = next((edge for edge in result.graph[target]
                                   if not edge.controllable and edge.target is None), None)
                    if unsafe is not None:
                        result.witness += path + [{"action": unsafe.action, "event": unsafe.kind,
                                                  "reason": "unsafe_uncontrollable_successor"}]
                        break
                    edges = result.graph[target]
                    if not edges:
                        result.witness += path + [{"reason": "completion_unreachable"}]
                        break
                    for edge in edges:
                        if edge.target is not None and edge.target not in seen and edge.target not in result.winning:
                            seen.add(edge.target)
                            queue.append((edge.target, path + [{"action": edge.action, "event": edge.kind}]))
    except (AnalysisLimit, IncompleteModel) as exc:
        result.status, result.reason = "inconclusive", str(exc)
        result.scope = getattr(plant, "scope", result.scope)
        result.winning.clear()
    result.elapsed = budget.clock() - budget.started
    return result
