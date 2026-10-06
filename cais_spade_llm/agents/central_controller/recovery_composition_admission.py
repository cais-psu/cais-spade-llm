from __future__ import annotations

"""CCA-owned admission of completely evidenced generated recovery programs.

Providers are configured by the runtime owner, never selected by agent messages.
The context provider is synchronous and runs under the admission lock. It supplies
``revision``, ``time_exact``, ``current_snapshot``, immutable ``composition_inputs``,
``task_monitor_context``, and an append-only ``observations`` ledger. Physical
monitor history requires ``physical_checkpoint`` or an explicit owner declaration
``physical_rule_activation='prospective'``. No default live evidence is invented.

Resource preparation is asynchronous and binds every event's exact primitives and
all supplied schedule evidence. Only the owner may enable the mock execution mode;
each resource preparation must additionally attest that its executor is a mock.
"""

import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from fractions import Fraction
from threading import RLock
from typing import Any

from cais_spade_llm.agents.central_controller.local_composition import Budget
from cais_spade_llm.agents.central_controller.offline_recovery_composition import (
    analyze_grounded_recovery_composition,
)
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _digest
from cais_spade_llm.resources.resource_safety_preparation import primitive_model_descriptors

logger = logging.getLogger(__name__)


class _ObservationPending(ValueError):
    """Validated nominal history is waiting for matching physical observations."""


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} requires an object")
    return value


def _symbol(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} requires an exact nonempty identifier")
    return value


def _state(value: dict) -> dict:
    result = {name: deepcopy(_object(value.get(name), name)) for name in ("resources", "parts")}
    for part in result["parts"].values():
        part.pop("stationary_until", None)
    return result


def _monitor_identity(context: dict) -> dict:
    native = _object(context.get("task_monitor_context"), "task_monitor_context")
    records = native.get("monitors", [{"scope_id": None, "monitor": native.get("monitor"),
                                       "current_states": native.get("current_states")}])
    monitors = []
    for record in records:
        monitor = record.get("monitor")
        if monitor is None:
            raise ValueError("native_monitor_history_unavailable")
        monitors.append({
            "scope_id": record["scope_id"],
            "event_ids": deepcopy(record.get("event_ids")),
            "rules": deepcopy(getattr(monitor, "safety_rules", None)),
            "dfas": deepcopy(getattr(monitor, "dfa_dots", None) or getattr(monitor, "dfas", {})),
            "current_states": deepcopy(record.get("current_states")),
            "history_error": deepcopy(getattr(monitor, "history_error", None)),
            "tools_catalog": deepcopy(getattr(monitor, "tools_catalog", [])),
            "resource_bindings": deepcopy(getattr(monitor, "resource_bindings", {})),
        })
    return {"monitors": monitors,
        "resources": deepcopy(native.get("resources")),
        "products": deepcopy(native.get("products")),
        "contexts": deepcopy(native.get("contexts")),
        "jids": deepcopy(native.get("jids")),
        "tasks": deepcopy(native.get("tasks")),
    }


def _native_contract(identity: dict) -> dict:
    return {"monitors": [{key: row[key] for key in (
        "scope_id", "event_ids", "rules", "dfas", "tools_catalog", "resource_bindings",
    )} for row in identity["monitors"]], "jids": identity["jids"], "tasks": identity["tasks"]}


def _synthetic(value: Any) -> bool:
    if isinstance(value, dict):
        return (value.get("synthetic") is True or value.get("source_kind") == "synthetic"
                or any(_synthetic(child) for child in value.values()))
    return isinstance(value, list) and any(_synthetic(child) for child in value)


class RecoveryCompositionAdmission:
    """Retain a proved graph and advance it only through authorized observations.

    Args:
        context_provider: Owner callback ``(product_jid, recovery_id) -> context``.
        resource_evidence_provider: Owner callback accepting one preparation request.
        lock: Shared runtime admission RLock, or a coordinator-owned RLock.
        allow_mock_execution: Explicit test-harness permission, never a payload flag.
        allow_nominal_tasks: Owner-enabled use of the same proof for exact nominal tasks.
        budget_factory: Factory for each complete composition analysis budget.
        native_history_commit: Owner callback ``(task_monitor_state, record)`` run
            under the lock after a start grant or a matched trusted completion.
        primitive_models_provider: Trusted owner callback returning pure models.
            Executable callbacks remain outside serialized context and messages.
    """

    def __init__(self, *, context_provider: Callable | None = None,
                 resource_evidence_provider: Callable | None = None, lock=None,
                 allow_mock_execution: bool = False,
                 allow_nominal_tasks: bool = False,
                 budget_factory: Callable[[], Budget] = Budget,
                 native_history_commit: Callable | None = None,
                 primitive_models_provider: Callable | None = None) -> None:
        self.context_provider = context_provider
        self.resource_evidence_provider = resource_evidence_provider
        self.lock = lock or RLock()
        self.allow_mock_execution = allow_mock_execution is True
        self.allow_nominal_tasks = allow_nominal_tasks is True
        self.budget_factory = budget_factory
        self.native_history_commit = native_history_commit
        self.primitive_models_provider = primitive_models_provider
        self.start_validator: Callable | None = None
        self.sessions: dict[str, dict] = {}
        self.last_results: dict[str, dict] = {}
        self.epoch = 0

    def _result(self, status: str, reason: str = "", **extra) -> dict:
        return {"status": status, "reason": reason, "admission_epoch": self.epoch, **deepcopy(extra)}

    def _context(self, product_jid: str, recovery_id: str) -> dict:
        if self.context_provider is None:
            raise ValueError("live_recovery_evidence_unavailable")
        value = self.context_provider(product_jid, recovery_id)
        if inspect.isawaitable(value):
            raise ValueError("context_provider_must_be_synchronous")
        context = dict(_object(value, "authoritative context"))
        if "primitive_models" in context or "primitive_models" in context.get("composition_inputs", {}):
            raise ValueError("executable_primitive_models_require_owner_registration")
        context["primitive_model_descriptors"] = primitive_model_descriptors(self._primitive_models())
        if "revision" not in context or context["revision"] is None:
            raise ValueError("authoritative_revision_unavailable")
        _state(_object(context.get("current_snapshot"), "current_snapshot"))
        Fraction(str(context["time_exact"]))
        _object(context.get("composition_inputs"), "composition_inputs")
        if not isinstance(context.get("observations"), list):
            raise ValueError("authoritative_observation_history_unavailable")
        native = _monitor_identity(context)
        if any(row["history_error"] for row in native["monitors"]):
            raise ValueError("native_monitor_history_invalid")
        if set(context.get("active_recovery_scopes", [])) - {row["scope_id"] for row in native["monitors"]}:
            raise ValueError("additional_recovery_scope_history_unavailable")
        if (context.get("physical_checkpoint") is None
                and context.get("physical_rule_activation") != "prospective"):
            raise ValueError("physical_monitor_history_unavailable")
        mode = context.get("execution_mode")
        if mode not in {"live", "mock"}:
            raise ValueError("execution_evidence_mode_unavailable")
        if mode == "mock" and not self.allow_mock_execution:
            raise ValueError("synthetic_evidence_cannot_authorize_live_execution")
        if mode == "live" and _synthetic(context["composition_inputs"]):
            raise ValueError("synthetic_evidence_cannot_authorize_live_execution")
        return context

    def _primitive_models(self) -> dict:
        if self.primitive_models_provider is None:
            return {}
        models = self.primitive_models_provider()
        primitive_model_descriptors(models)
        return dict(models)

    @staticmethod
    def _context_identity(context: dict) -> str:
        return _digest({
            "revision": context["revision"], "time_exact": context["time_exact"],
            "snapshot": context["current_snapshot"], "observations": context["observations"],
            "inputs": context["composition_inputs"], "monitor": _monitor_identity(context),
            "execution_mode": context["execution_mode"],
            "physical_checkpoint": context.get("physical_checkpoint"),
            "physical_rule_activation": context.get("physical_rule_activation"),
            "nominal_acknowledgements": context.get("nominal_acknowledgements", []),
            "owner_incarnations": context.get("owner_incarnations"),
            "primitive_model_descriptors": context.get("primitive_model_descriptors", {}),
        })

    def _tasks(self, request: dict, inputs: dict, product_jid: str) -> tuple[dict, dict]:
        tasks, preparations = {}, {}
        events = {row["outline_id"]: row for row in inputs["recovery_events"]}
        if not isinstance(request.get("tasks"), list):
            raise ValueError("complete_recovery_task_mapping_unavailable")
        seen = set()
        for raw in request["tasks"]:
            task = deepcopy(_object(raw, "recovery task"))
            for field in ("task_id", "outline_id", "resource_jid", "function_name"):
                _symbol(task.get(field), field)
            identity = task["outline_id"]
            if task["task_id"] in tasks or identity in seen or identity not in events:
                raise ValueError("recovery_task_identity_mismatch")
            seen.add(identity)
            event = events[identity]
            task.setdefault("resource_id", event["resource_id"])
            params = _object(task.get("params"), "dispatch params")
            nominal = self.allow_nominal_tasks and task["function_name"] != "execute_recovery_macro"
            if nominal:
                if task["resource_id"] != event["resource_id"] or task["function_name"] != event["event_name"]:
                    raise ValueError("nominal_dispatch_identity_mismatch")
                primitive_steps = task.get("primitive_steps")
            else:
                primitive_steps = params.get("primitive_steps")
            if not nominal and (task["function_name"] != "execute_recovery_macro"
                    or task["resource_id"] != event["resource_id"]
                    or params.get("outline_id", params.get("recovery_outline_id")) != identity
                    or params.get("event_name") != event["event_name"]
                    or params.get("product_jid") != product_jid
                    or params.get("task_id") != task["task_id"]
                    or params.get("recovery_safety_scope_id") != request.get("recovery_safety_scope_id")
                    or params.get("start_safety_mode") != "cca_check"):
                raise ValueError("recovery_dispatch_identity_mismatch")
            snapshot = inputs["grounding_inputs"]["snapshot"]["resources"][task["resource_id"]]
            if snapshot.get("resource_jid") != task["resource_jid"]:
                raise ValueError("resource_jid_mismatch")
            programs = []
            for choice in inputs["event_start_choices"]:
                program = next(row for row in choice["programs"] if row["resource_id"] == task["resource_id"])
                indices = event["primitive_step_indices"]
                steps = [program["primitive_steps"][index] for index in indices]
                if primitive_steps != steps:
                    raise ValueError("recovery_primitive_program_mismatch")
                programs.append({"schedule_id": choice["id"], "start_time": choice["starts"][identity],
                                 "primitive_steps": deepcopy(steps),
                                 "step_results": [deepcopy(program["step_results"][index]) for index in indices]})
                if "des_event_id" in params and any(
                        trace["source"].get("des_event_id") != params["des_event_id"]
                        for trace in programs[-1]["step_results"]):
                    raise ValueError("recovery_des_event_id_mismatch")
            preparation = {key: task[key] for key in ("task_id", "outline_id", "resource_id", "resource_jid")}
            preparation.update(recovery_id=request["recovery_id"],
                               primitive_steps=deepcopy(primitive_steps), event_programs=programs)
            if nominal:
                preparation.update(function_name=task["function_name"], params=deepcopy(params))
            preparation["program_hash"] = _digest(preparation)
            tasks[task["task_id"]] = task
            preparations[task["task_id"]] = preparation
        if seen != set(events):
            raise ValueError("complete_recovery_task_mapping_unavailable")
        return tasks, preparations

    async def _prepare_resources(self, preparations: dict, mode: str) -> dict:
        if self.resource_evidence_provider is None:
            raise ValueError("resource_preparation_unavailable")
        results = {}
        for task_id, preparation in preparations.items():
            value = self.resource_evidence_provider(deepcopy(preparation))
            result = await value if inspect.isawaitable(value) else value
            result = _object(result, "resource preparation")
            if (result.get("status") != "prepared"
                    or result.get("resource_jid") != preparation["resource_jid"]
                    or result.get("program_hash") != preparation["program_hash"]
                    or result.get("execution_mode") != mode
                    or not isinstance(result.get("preparation_id"), str)
                    or not result["preparation_id"]):
                raise ValueError("resource_preparation_unavailable_or_mismatched")
            if mode == "mock" and (not self.allow_mock_execution or result.get("mock_executor") is not True):
                raise ValueError("synthetic_evidence_requires_mock_executor")
            results[task_id] = deepcopy(result)
        return results

    def _registration_native(self, context: dict, inputs: dict, tasks: dict) -> dict:
        if "nominal_running_tasks" in context and {
                row["task_id"] for row in context["nominal_running_tasks"]} != {
                row["task_id"] for row in inputs["running_work"]}:
            raise ValueError("already_running_nominal_tasks_are_not_completely_modeled")
        if self.native_history_commit is None:
            raise ValueError("native_history_commit_unavailable")
        native = deepcopy(context["task_monitor_context"])
        for task in tasks.values():
            binding = native["tasks"][task["outline_id"]]["task"]
            if (binding.get("task_id") != task["task_id"]
                    or binding.get("function_name") != task["function_name"]
                    or binding.get("resource_id") != task["resource_id"]
                    or binding.get("parameters") != task["params"]):
                raise ValueError("native_task_binding_disagrees_with_dispatch")
        return native

    async def register(self, request: dict, *, product_jid: str) -> dict:
        """Analyze complete owner-evidenced recovery before publishing task references."""
        recovery_id = ""
        try:
            request = deepcopy(_object(request, "recovery composition request"))
            recovery_id = _symbol(request.get("recovery_id"), "recovery_id")
            _symbol(request.get("recovery_safety_scope_id"), "recovery_safety_scope_id")
            _symbol(product_jid, "product_jid")
            with self.lock:
                existing = self.sessions.get(recovery_id)
                if existing is not None:
                    if existing["request"] != request or existing["product_jid"] != product_jid:
                        self._invalidate(existing, "recovery_registration_changed")
                        raise ValueError("recovery_registration_changed")
                    self._synchronize(existing, self._context(product_jid, recovery_id))
                    return self._registration_result(existing)
                if any(not row["complete"] for row in self.sessions.values()):
                    raise ValueError("another_recovery_composition_is_active")
                context = self._context(product_jid, recovery_id)
                if self.sessions and context.get("physical_checkpoint") is None:
                    raise ValueError("existing_physical_history_requires_checkpoint")
                identity, epoch = self._context_identity(context), self.epoch
                inputs = deepcopy(context["composition_inputs"])
                if "composition_inputs" in request and request["composition_inputs"] != inputs:
                    raise ValueError("caller_evidence_disagrees_with_authoritative_context")
                tasks, preparations = self._tasks(request, inputs, product_jid)
                native = self._registration_native(context, inputs, tasks)
                checkpoint = deepcopy(context.get("physical_checkpoint"))
                mode = context["execution_mode"]
                primitive_models = self._primitive_models()
                if primitive_model_descriptors(primitive_models) != context["primitive_model_descriptors"]:
                    raise ValueError("resource_primitive_model_changed")
            prepared = await self._prepare_resources(preparations, mode)
            analysis = await asyncio.to_thread(
                analyze_grounded_recovery_composition, **inputs,
                task_monitor_context=native, physical_checkpoint=checkpoint,
                budget=self.budget_factory(),
                primitive_models=primitive_models or None,
            )
            with self.lock:
                current = self._context(product_jid, recovery_id)
                if self.epoch != epoch or identity != self._context_identity(current):
                    raise ValueError("stale_registration_snapshot")
                if analysis["status"] != "allowed":
                    return self._result(analysis["status"], analysis.get("reason", ""),
                                        recovery_id=recovery_id, analysis=analysis)
                graph = analysis["graph"]
                scope_id = _symbol(request.get("recovery_safety_scope_id"), "recovery_safety_scope_id")
                if scope_id not in {row["scope_id"] for row in _monitor_identity(current)["monitors"]}:
                    raise ValueError("recovery_safety_scope_history_unavailable")
                pending = request.get("pending_nominal_task_ids")
                statuses = inputs["grounding_inputs"]["snapshot"].get("task_statuses", {})
                if (not isinstance(pending, list) or len(set(pending)) != len(pending)
                        or set(pending) != {key for key, value in statuses.items() if value == "pending"}):
                    raise ValueError("pending_nominal_tasks_disagree_with_authoritative_context")
                nodes = {row["id"]: row for row in graph["nodes"]}
                root = analysis.get("root_node", graph.get("root_node"))
                if root not in nodes:
                    raise ValueError("composition_root_unavailable")
                edges = defaultdict(list)
                for edge in graph["edges"]:
                    edges[edge["source"]].append(edge)
                refs = {task_id: {"recovery_id": recovery_id, "task_id": task_id,
                                  "outline_id": row["outline_id"], "program_hash": row["program_hash"],
                                  "problem_id": analysis["problem_id"]}
                        for task_id, row in preparations.items()}
                session = {"request": request, "product_jid": product_jid, "inputs": inputs,
                           "primitive_model_descriptors": deepcopy(current["primitive_model_descriptors"]),
                           "tasks": tasks, "preparations": preparations, "prepared": prepared,
                           "refs": refs, "analysis": analysis, "nodes": nodes, "edges": dict(edges),
                           "node": root, "path": [], "grants": {}, "completed_tasks": set(),
                           "ledger": [], "record_ids": {}, "invalid_reason": "", "complete": False,
                           "execution_mode": mode, "native_identity": _monitor_identity(current),
                           "revision": deepcopy(current["revision"]), "time_exact": current["time_exact"],
                           "physical_history": _digest([current.get("physical_checkpoint"),
                                                        current.get("physical_rule_activation")]),
                           "nominal_acknowledgements": deepcopy(current.get("nominal_acknowledgements", [])),
                           "nominal_run_id": current.get("nominal_run_id"),
                           "owner_incarnations": deepcopy(current.get("owner_incarnations"))}
                self._synchronize(session, current)
                self.sessions[recovery_id] = session
                self.epoch += 1
                result = self._registration_result(session)
                self.last_results[recovery_id] = result
                return result
        except (ValueError, KeyError, TypeError, IndexError, StopIteration, OSError) as exc:
            result = self._result("inconclusive", str(exc), recovery_id=recovery_id)
            self.last_results[recovery_id] = result
            return result

    def _registration_result(self, session: dict) -> dict:
        return self._result("inconclusive" if session["invalid_reason"] else "allowed",
                            session["invalid_reason"], recovery_id=session["request"]["recovery_id"],
                            problem_id=session["analysis"]["problem_id"], task_refs=session["refs"],
                            scope=session["analysis"]["scope"], complete=session["complete"])

    def _invalidate(self, session: dict, reason: str) -> None:
        session["invalid_reason"] = reason
        self.epoch += 1

    def invalidate(self, reason: str) -> None:
        """Invalidate all active proofs without clearing grants or monitor history."""
        with self.lock:
            for session in self.sessions.values():
                if not session["complete"]:
                    self._invalidate(session, reason)

    def _take(self, session: dict, edge: dict) -> None:
        target = edge.get("target")
        if target not in session["nodes"] or not session["nodes"][target]["winning"]:
            raise ValueError("observed_transition_outside_winning_composition")
        if edge["kind"] in {"start", "task_completion"}:
            if self.native_history_commit is None:
                raise ValueError("native_history_commit_unavailable")
            if "_staged_native" in session:
                session["_staged_native"].append(deepcopy(edge))
            else:
                self.native_history_commit(deepcopy(edge["task_monitor_state"]), {
                    **deepcopy(edge), "recovery_id": session["request"]["recovery_id"],
                })
        session["path"].append(edge["edge_id"])
        session["node"] = target
        if "_staged_steps" in session:
            session["_staged_steps"] += 1
        else:
            self.epoch += 1

    def _automatic(self, session: dict, now: Fraction) -> None:
        while True:
            edges = session["edges"].get(session["node"], [])
            if len(edges) != 1:
                return
            edge = edges[0]
            if edge["kind"] not in {"decision", "wait"} or Fraction(edge["time_exact"]) > now:
                return
            self._take(session, edge)

    @staticmethod
    def _observation_matches(edge: dict, record: dict, acknowledged_task_ids: set[str] | None = None) -> bool:
        if edge["kind"] != record.get("kind"):
            return False
        if str(record.get("time_exact")) != edge.get("time_exact"):
            return False
        if edge["kind"] == "observation":
            expected = edge.get("observation")
            actual = record.get("observation")
            if not isinstance(expected, dict) or not isinstance(actual, dict):
                return False
            from cais_spade_llm.agents.central_controller._recovery_monitor_history import (
                acknowledged_product_effects_for_comparison,
            )

            actual = acknowledged_product_effects_for_comparison(expected, actual, acknowledged_task_ids or set())
            fields = ("region_occupancy", "part_region_occupancy", "carried_parts")
            return _state(expected) == _state(actual) and all(actual.get(key) == expected.get(key) for key in fields)
        if edge["kind"] == "task_completion":
            return (record.get("task_id") == edge.get("task_id")
                    and record.get("status") == "completed")
        return False

    @staticmethod
    def _validate_observed_context(session: dict, context: dict) -> tuple[list, Fraction]:
        if context["composition_inputs"] != session["inputs"] or context["execution_mode"] != session["execution_mode"]:
            raise ValueError("recovery_evidence_changed")
        if context.get("primitive_model_descriptors", {}) != session.get("primitive_model_descriptors", {}):
            raise ValueError("resource_primitive_model_changed")
        current_native = _monitor_identity(context)
        if context.get("owner_incarnations") != session.get("owner_incarnations"):
            raise ValueError("resource_owner_incarnation_changed")
        if _native_contract(current_native) != _native_contract(session["native_identity"]):
            raise ValueError("active_native_specifications_changed")
        if _digest([context.get("physical_checkpoint"), context.get("physical_rule_activation")]) != session["physical_history"]:
            raise ValueError("accepted_physical_history_changed")
        ledger = context["observations"]
        if ledger[:len(session["ledger"])] != session["ledger"]:
            raise ValueError("acknowledged_history_changed")
        now = Fraction(str(context["time_exact"]))
        if now < Fraction(str(session["time_exact"])):
            raise ValueError("authoritative_clock_moved_backwards")
        if (context["revision"] != session["revision"] and len(ledger) == len(session["ledger"])
                and now == Fraction(str(session["time_exact"]))
                and len(context.get("nominal_acknowledgements", [])) == len(session.get("nominal_acknowledgements", []))):
            raise ValueError("unacknowledged_authority_revision_change")
        return ledger, now

    def _replay_observations(self, session: dict, ledger: list, now: Fraction,
                             acknowledged_task_ids: set[str]) -> None:
        for record in ledger[len(session["ledger"]):]:
            record = _object(record, "acknowledged observation")
            identity = record.get("record_id", _digest(record))
            if identity in session["record_ids"]:
                if session["record_ids"][identity] != record:
                    raise ValueError("duplicate_observation_changed")
                session["ledger"].append(deepcopy(record))
                continue
            self._automatic(session, now)
            candidates = [edge for edge in session["edges"].get(session["node"], [])
                          if not edge["controllable"] and self._observation_matches(edge, record, acknowledged_task_ids)]
            if len(candidates) != 1:
                raise ValueError("unexpected_or_unacknowledged_recovery_observation")
            edge = candidates[0]
            if Fraction(edge["time_exact"]) > now:
                raise ValueError("observation_is_in_the_future")
            self._take(session, edge)
            if edge["kind"] == "task_completion":
                identity = edge["task_id"]
                task = next((task for task in session["tasks"].values()
                             if task["outline_id"] == identity or task["task_id"] == identity), None)
                if task is not None:
                    if task["task_id"] not in session["grants"]:
                        raise ValueError("completion_without_recovery_admission")
                    session["completed_tasks"].add(task["task_id"])
            session["record_ids"][record.get("record_id", _digest(record))] = deepcopy(record)
            session["ledger"].append(deepcopy(record))

    def _synchronize(self, session: dict, context: dict) -> None:
        if session["invalid_reason"]:
            raise ValueError(session["invalid_reason"])
        original = session
        session = {**original, "path": list(original["path"]), "completed_tasks": set(original["completed_tasks"]),
                   "ledger": deepcopy(original["ledger"]), "record_ids": deepcopy(original["record_ids"]),
                   "_staged_native": [], "_staged_steps": 0}
        before_native = original["nodes"][original["node"]].get("task_monitor_state")
        try:
            ledger, now = self._validate_observed_context(session, context)
            acknowledged_task_ids = {
                row["task_id"] for row in context.get("nominal_acknowledgements", [])
                if row.get("admitted") is True
            }
            # Only the owner ledger reports actual macro completions. Nominal
            # completions additionally require EnvironmentAdmission's audit below.
            acknowledged_task_ids.update(
                row["task_id"] for row in ledger if row.get("kind") == "task_completion"
                and row.get("status") == "completed" and Fraction(str(row["time_exact"])) <= now
                and (context.get("nominal_run_id") is None or
                     session["tasks"].get(row["task_id"], {}).get("function_name") == "execute_recovery_macro"))
            self._replay_observations(session, ledger, now, acknowledged_task_ids)
            self._automatic(session, now)
            reused = self._nominal_history(session, context)
            node = session["nodes"][session["node"]]
            expected = node.get("physical_state", node.get("state"))
            from cais_spade_llm.agents.central_controller._recovery_monitor_history import (
                acknowledged_product_effects_for_comparison,
            )

            actual_snapshot = acknowledged_product_effects_for_comparison(
                expected, context["current_snapshot"], acknowledged_task_ids)
            if not isinstance(expected, dict) or _state(expected) != _state(actual_snapshot):
                raise ValueError("authoritative_physical_snapshot_mismatch")
            refreshed = self._context(session["product_jid"], session["request"]["recovery_id"])
            native = _monitor_identity(refreshed)
            native_states = node.get("task_monitor_state")
            source_states = before_native if session["_staged_native"] else native_states
            expected_states = {row["scope_id"]: row["state"]["states"]
                               for row in (source_states or {}).get("monitors", [])}
            actual_states = {row["scope_id"]: row["current_states"] for row in native["monitors"]}
            if reused:
                expected_states[None] = reused[-1]["after_states"]
            if native_states is None or expected_states != actual_states:
                raise ValueError("authoritative_native_monitor_mismatch")
            if any(row["state"][key] != native[key] for row in native_states["monitors"]
                   for key in ("resources", "products", "contexts")):
                raise ValueError("authoritative_native_state_mismatch")
            if session["_staged_native"]:
                record = {**deepcopy(session["_staged_native"][-1]),
                          "recovery_id": session["request"]["recovery_id"],
                          "acknowledged_transitions": deepcopy(session["_staged_native"]),
                          "reused_monitor_scopes": [None] if reused else []}
                self.native_history_commit(deepcopy(native_states), record)
                refreshed = self._context(session["product_jid"], session["request"]["recovery_id"])
            session["complete"] = bool(node["marked"] and set(session["tasks"]) <= session["completed_tasks"])
            session["revision"] = deepcopy(refreshed["revision"])
            session["time_exact"] = refreshed["time_exact"]
            session["nominal_acknowledgements"] = deepcopy(refreshed.get("nominal_acknowledgements", []))
            self.epoch += session.pop("_staged_steps")
            session.pop("_staged_native")
            original.update(session)
        except _ObservationPending:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            self._invalidate(original, str(exc))
            raise

    @staticmethod
    def _nominal_history(session: dict, context: dict) -> list[dict]:
        """Match owner-committed nominal ticks to the same observed graph edges."""
        history = context.get("nominal_acknowledgements", [])
        previous = session.get("nominal_acknowledgements", [])
        if (not isinstance(history, list) or history[:len(previous)] != previous
                or context.get("nominal_run_id") != session.get("nominal_run_id")):
            raise ValueError("nominal_acknowledgement_history_changed")
        appended = history[len(previous):]
        running = {row["task_id"] for row in session["inputs"]["running_work"]}
        running.update(task["task_id"] for task in session["tasks"].values()
                       if task["function_name"] != "execute_recovery_macro")
        expected = [edge for edge in session["_staged_native"]
                    if edge["kind"] == "task_completion" and edge["task_id"] in running]
        if context.get("nominal_run_id") is None:
            if appended:
                raise ValueError("nominal_acknowledgement_owner_unavailable")
            return []
        seen_tasks = set()
        for index, audit in enumerate(history, 1):
            if (type(audit.get("cursor")) is not int or audit["cursor"] != index or audit.get("task_id") in seen_tasks
                    or audit.get("run_id") != context["nominal_run_id"] or audit.get("admitted") is not True):
                raise ValueError("nominal_acknowledgement_does_not_match_recovery_graph")
            seen_tasks.add(audit.get("task_id"))
        if len(appended) != len(expected):
            if any(row.get("task_id") not in running for row in appended):
                raise ValueError("nominal_acknowledgement_does_not_match_recovery_graph")
            raise _ObservationPending("nominal_acknowledgement_requires_matching_physical_history")
        for index, (audit, edge) in enumerate(zip(appended, expected, strict=True), len(previous) + 1):
            task = session["native_identity"]["tasks"][edge["event_id"]]["task"]
            source = session["nodes"][edge["source"]]["task_monitor_state"]
            before = next(row["state"]["states"] for row in source["monitors"] if row["scope_id"] is None)
            after = next(row["state"]["states"] for row in edge["task_monitor_state"]["monitors"]
                         if row["scope_id"] is None)
            checks = {row["rule_id"]: {key: value for key, value in row.items() if key != "scope_id"}
                      for row in edge["task_rule_checks"] if row["scope_id"] is None}
            if (audit.get("cursor") != index or audit.get("run_id") != context["nominal_run_id"]
                    or audit.get("admitted") is not True or audit.get("task_id") != edge["task_id"]
                    or any(audit.get("task", {}).get(key) != task.get(key)
                           for key in ("task_id", "resource_id", "event_name", "parameters"))
                    or audit.get("before_states") != before or audit.get("after_states") != after
                    or len(audit.get("rule_checks", [])) != len(checks)
                    or {row["rule_id"]: row for row in audit.get("rule_checks", [])} != checks):
                raise ValueError("nominal_acknowledgement_does_not_match_recovery_graph")
        return appended

    def _existing_grant(self, session: dict, task_id: str, reference: dict) -> dict | None:
        if task_id not in session["grants"]:
            return None
        if task_id in session["completed_tasks"]:
            return self._result("held", "recovery_task_already_completed", task_id=task_id)
        return self._result("allowed", "already_admitted", task_id=task_id,
                            recovery_composition_ref=reference,
                            recovery_composition_grant=session["grants"][task_id]["grant"])

    async def check(self, event: dict, *, sender: str, commit: bool = True) -> dict:
        """Authorize the exact next event, never a whole problem's existence result."""
        try:
            event = deepcopy(_object(event, "resource event"))
            params = _object(event.get("params"), "dispatch params")
            reference = _object(params.pop("recovery_composition_ref", None), "recovery_composition_ref")
            task_id, recovery_id = event.get("task_id"), reference.get("recovery_id")
            with self.lock:
                session = self.sessions.get(recovery_id)
                if session is None:
                    raise ValueError("recovery_composition_not_registered")
                task = session["tasks"].get(task_id)
                if (task is None or sender != task["resource_jid"] or reference != session["refs"][task_id]
                        or event.get("resource_jid") != task["resource_jid"]
                        or event.get("function_name") != task["function_name"] or params != task["params"]):
                    raise ValueError("recovery_dispatch_reference_mismatch")
                context = self._context(session["product_jid"], recovery_id)
                self._synchronize(session, context)
                context = self._context(session["product_jid"], recovery_id)
                existing = self._existing_grant(session, task_id, reference)
                if existing is not None:
                    return existing
                identity, epoch = self._context_identity(context), self.epoch
                preparation = deepcopy(session["preparations"][task_id])
                mode = session["execution_mode"]
            prepared = await self._prepare_resources({task_id: preparation}, mode)
            with self.lock:
                current = self._context(session["product_jid"], recovery_id)
                if self.epoch != epoch or self._context_identity(current) != identity:
                    raise ValueError("stale_admission_snapshot")
                self._synchronize(session, current)
                if self.start_validator is not None and not self.start_validator(deepcopy(task)):
                    return self._result("held", "plan_task_not_enabled", task_id=task_id)
                now = Fraction(str(current["time_exact"]))
                edges = session["edges"].get(session["node"], [])
                candidates = [edge for edge in edges if edge["kind"] == "start"
                              and task["outline_id"] in edge["event_ids"]]
                if not candidates:
                    return self._result("held", "event_not_enabled_at_acknowledged_prefix", task_id=task_id)
                if len(candidates) != 1 or candidates[0]["event_ids"] != [task["outline_id"]]:
                    raise ValueError("joint_start_requires_atomic_execution")
                edge = candidates[0]
                if Fraction(edge["time_exact"]) != now:
                    return self._result("held", "event_start_time_not_observed", task_id=task_id)
                if not session["nodes"].get(edge["target"], {}).get("winning", False):
                    waits = [row for row in edges if row["kind"] == "wait"
                             and session["nodes"].get(row["target"], {}).get("winning", False)]
                    if commit and len(waits) == 1:
                        self._take(session, waits[0])
                    return self._result("held", "specific_start_has_no_joint_completion", task_id=task_id)
                if commit:
                    program = next(row for row in preparation["event_programs"]
                                   if Fraction(str(row["start_time"])) == now)
                    grant = {"recovery_composition_ref": reference,
                             "preparation_id": prepared[task_id]["preparation_id"],
                             "schedule_id": program["schedule_id"],
                             "primitive_steps": deepcopy(preparation["primitive_steps"]),
                             "resolved_primitive_steps": [{"primitive": row["primitive"],
                                                           "params": deepcopy(row["resolved_params"])}
                                                          for row in program["step_results"]],
                             "step_results": deepcopy(program["step_results"])}
                    self._take(session, edge)
                    session["grants"][task_id] = {"reference": reference, "preparation": prepared[task_id],
                                                  "edge_id": edge["edge_id"], "grant": grant}
                    session["revision"] = deepcopy(self._context(session["product_jid"], recovery_id)["revision"])
                result = self._result("allowed", task_id=task_id, recovery_composition_ref=reference,
                                      snapshot_revision=current["revision"], committed=commit)
                if commit:
                    result["recovery_composition_grant"] = deepcopy(grant)
                self.last_results[task_id] = result
                return result
        except _ObservationPending as exc:
            return self._result("held", str(exc), task_id=event.get("task_id", ""))
        except (ValueError, KeyError, TypeError, IndexError, StopIteration, OSError) as exc:
            return self._result("inconclusive", str(exc),
                                task_id=event.get("task_id", "") if isinstance(event, dict) else "")

    def observe(self, record: dict, *, sender: str) -> dict:
        """Use an authenticated notification only to refresh owner-acknowledged history."""
        with self.lock:
            affected = []
            for session in self.sessions.values():
                if session["complete"]:
                    continue
                senders = {session["product_jid"], *(row["resource_jid"] for row in session["tasks"].values())}
                resources = session["inputs"]["grounding_inputs"]["snapshot"]["resources"]
                senders.update(row.get("resource_jid") for row in resources.values())
                senders.update(session["native_identity"]["jids"].values())
                if sender not in senders:
                    continue
                task_id = record.get("task_id")
                task = session["tasks"].get(task_id)
                params = record.get("params") or {}
                exact_task = (task is not None and isinstance(params, dict)
                              and params.get("recovery_composition_ref") == session["refs"][task_id]
                              and sender in {session["product_jid"], task["resource_jid"]})
                exact_running = any(row["task_id"] == task_id and sender ==
                                    session["native_identity"]["jids"].get(row["resource_id"])
                                    and (session.get("nominal_run_id") is None
                                         or record.get("run_id") == session["nominal_run_id"])
                                    for row in session["inputs"]["running_work"])
                if not (exact_task or exact_running or sender == session["product_jid"]
                        and record.get("status") == "owner_observation"):
                    continue
                try:
                    status = str(record.get("status", ""))
                    observed_status = str((record.get("observations") or {}).get("status", ""))
                    if (status.startswith("failed") or status in {"error", "cancelled"}
                            or observed_status.startswith("failed")):
                        raise ValueError("recovery_execution_failed")
                    context = self._context(session["product_jid"], session["request"]["recovery_id"])
                    self._synchronize(session, context)
                    if status == "recovery_acknowledgement" and observed_status == "completed":
                        task_id = record.get("task_id")
                        if task_id in session["tasks"] and task_id not in session["completed_tasks"]:
                            raise ValueError("completion_not_in_authoritative_history")
                        running = {row["task_id"] for row in session["inputs"]["running_work"]}
                        completed = session["nodes"][session["node"]]["task_monitor_state"]["completed"]
                        if task_id in running and task_id not in completed:
                            raise ValueError("completion_not_in_authoritative_history")
                    affected.append(self._registration_result(session))
                except _ObservationPending as exc:
                    affected.append(self._result("held", str(exc)))
                except (ValueError, KeyError, TypeError) as exc:
                    self._invalidate(session, str(exc))
                    affected.append(self._result("inconclusive", str(exc)))
            return self._result("observed", results=affected)

    def holds(self, event: dict | None = None) -> bool:
        """Hold new interacting nominal starts until the acknowledged recovery finishes."""
        with self.lock:
            for session in self.sessions.values():
                if session["complete"]:
                    continue
                if event is None:
                    return True
                resource = event.get("resource_id")
                jid = event.get("resource_jid")
                modeled = session["inputs"]["grounding_inputs"]["snapshot"]["resources"]
                if resource is None and jid is None or resource in modeled or any(
                        jid == row.get("resource_jid") for row in modeled.values()):
                    return True
            return False
