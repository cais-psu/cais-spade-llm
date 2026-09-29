"""Separate reachability evidence for the predefined task-level DES baseline."""

from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)


def audit_task_des(
    *,
    model: dict[str, Any] | None,
    failure_valuation: dict[str, Any] | None,
    goal_conditions: dict[str, Any] | None,
    complete: bool = False,
    max_states: int = 200_000,
    deadline_s: float = 10.0,
) -> dict[str, Any]:
    """Search an explicit task model with an exact observed start valuation.

    Args:
        model: State valuations, grounded events, and transitions in the existing
            environment-model representation. Bid fragments are not complete models.
        failure_valuation: Complete post-fault valuation in that model's vocabulary.
        goal_conditions: Exact required terminal fields, including unfinished tasks.
        complete: Whether the supplied model enumerates the entire relevant
            predefined task behavior. This is a model-construction assertion.
        max_states: Exploration limit; exhaustion is inconclusive.
        deadline_s: Wall-clock limit; exhaustion is inconclusive.

    Returns:
        A modeled path, exhaustive no-path evidence, or an inconclusive reason.
        This is evaluator evidence, not physical feasibility or CCA authorization.
    """
    started = time.monotonic()
    visited: set[str] = set()

    def result(
        status: str, reason: str, path: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        return {
            "status": status,
            "reason": reason,
            "path": path or [],
            "explored_states": len(visited),
            "elapsed_s": time.monotonic() - started,
            "model_fingerprint": recovery_validation_fingerprint(model),
            "failure_fingerprint": recovery_validation_fingerprint(failure_valuation),
            "goal_conditions": deepcopy(goal_conditions),
            "model_declared_complete": complete,
            "max_states": max_states,
            "deadline_s": deadline_s,
            "claim_scope": "predefined_task_level_DES",
        }

    if not model or not failure_valuation or not goal_conditions:
        return result(
            "inconclusive", "Missing task model, exact failure valuation, or remaining goals"
        )
    states, transitions, events = (model.get(key) for key in ("states", "transitions", "events"))
    if not all(isinstance(value, dict) for value in (states, transitions, events)):
        return result("inconclusive", "Unsupported or incomplete task-model representation")
    starts = [
        key
        for key, value in states.items()
        if recovery_validation_fingerprint(value)
        == recovery_validation_fingerprint(failure_valuation)
    ]
    if len(starts) != 1:
        return result("inconclusive", "Failure valuation has no unique exact task-model state")
    if any(
        source not in states
        or not isinstance(edges, dict)
        or any(event not in events or target not in states for event, target in edges.items())
        for source, edges in transitions.items()
    ):
        return result("inconclusive", "Task-model transitions reference missing states or events")
    start = starts[0]
    queue = deque([start])
    parents: dict[str, tuple[str, str] | None] = {start: None}
    while queue:
        if len(visited) >= max_states or time.monotonic() - started >= deadline_s:
            return result("inconclusive", "Task-model exploration budget exhausted")
        state = queue.popleft()
        visited.add(state)
        valuation = states[state]
        if all(
            key in valuation
            and recovery_validation_fingerprint(valuation[key])
            == recovery_validation_fingerprint(value)
            for key, value in goal_conditions.items()
        ):
            path = []
            cursor = state
            while parents[cursor] is not None:
                previous, event = parents[cursor]
                path.append(
                    {
                        "event_id": event,
                        "event": deepcopy(events[event]),
                        "from": previous,
                        "to": cursor,
                    }
                )
                cursor = previous
            return result(
                "path_found", "A predefined task-level continuation exists", list(reversed(path))
            )
        for event, successor in transitions.get(state, {}).items():
            if successor not in parents:
                parents[successor] = (state, event)
                queue.append(successor)
    if complete:
        return result(
            "no_path",
            "Exhausted the supplied complete task-level model without reaching the remaining goals",
        )
    return result(
        "inconclusive",
        "No path in a partial model does not establish absence of a task-level recovery",
    )


def failure_snapshot_fingerprint(runtime_context: dict[str, Any]) -> str:
    """Bind baseline evidence to the saved fault, custody, and task obligations."""
    return recovery_validation_fingerprint(
        {
            key: deepcopy(runtime_context.get(key))
            for key in (
                "failure_scenario_id",
                "failure_event",
                "resource_snapshots",
                "part_tracker",
                "task_statuses",
                "goal_state",
                "obligation_targets",
                "recovery_safety_context",
            )
        }
    )


def audit_saved_failure(runtime_context: dict[str, Any]) -> dict[str, Any]:
    """Audit only a task-model start explicitly bound to this failure snapshot."""
    supplied = runtime_context.get("predefined_task_des_audit") or {}
    fingerprint = failure_snapshot_fingerprint(runtime_context)
    bound = supplied.get("snapshot_fingerprint") == fingerprint
    result = audit_task_des(
        model=supplied.get("model") if bound else None,
        failure_valuation=supplied.get("failure_valuation"),
        goal_conditions=supplied.get("goal_conditions"),
        complete=supplied.get("complete") is True,
    )
    if supplied and not bound:
        result["reason"] = "Task-model start is not bound to the current failure snapshot"
    result["snapshot_fingerprint"] = fingerprint
    result["supplied_snapshot_fingerprint"] = supplied.get("snapshot_fingerprint")
    return result
