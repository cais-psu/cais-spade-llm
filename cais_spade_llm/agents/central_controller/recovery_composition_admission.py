from __future__ import annotations

"""CCA-owned physical admission for nominal and generated recovery execution.

The owner configures complete-graph analysis or prepared-motion checking. Both
calculations use this coordinator's lifecycle and admission lock. Prepared motion
retains physical AP history in the CCA monitor and exact commands in its ledger;
it does not manufacture schedules or weaken the complete-graph input contract.

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

from cais_spade_llm.agents.central_controller.local_composition import AnalysisLimit, Budget
from cais_spade_llm.agents.central_controller.offline_recovery_composition import (
    analyze_grounded_recovery_composition,
)
from cais_spade_llm.agents.central_controller.offline_safety_grounding import (
    _prepare_grounded_primitive_trace,
)
from cais_spade_llm.agents.central_controller.region_admission import (
    RegionReservationLedger,
    reserve_prepared_program,
)
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _digest
from cais_spade_llm.recovery_framework import fingerprint
from cais_spade_llm.resources.resource_safety_preparation import (
    LiveCommandLedger,
    PreparedRobotEvidence,
    primitive_model_descriptors,
)

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


def _overlap(bounds: list, region: list) -> bool:
    return all(bounds[i][0] <= region[i][1] and bounds[i][1] >= region[i][0] for i in range(3))


def checkpoint_region_observation(checkpoint: dict, revision: int) -> dict:
    """Label APs from one complete physics snapshot, including attached geometry."""
    resources = checkpoint["observations"]
    occupancy = {}
    for region, shape in checkpoint["geometry"]["regions"].items():
        occupancy[region] = {}
        for rid, row in resources.items():
            physical = row["physical"]
            boxes = [component["bounds"] for component in physical["component_bounds"]]
            held = physical["observation_state"].get("held_part")
            if held is not None:
                part = checkpoint["parts"][held]
                footprint = checkpoint["geometry"]["parts"][held]["footprint"]
                boxes.append([[a + part["pose"][i], b + part["pose"][i]]
                              for i, (a, b) in enumerate(footprint)])
            if not boxes:
                raise ValueError("Physical region labeling requires complete resource geometry")
            occupancy[region][rid] = any(_overlap(box, shape["bounds"]) for box in boxes)
    return {"revision": revision, "time_exact": str(max(
                row["physical"]["simulation_time"] for row in resources.values())),
            "region_occupancy": occupancy,
            "stationary_resources": sorted(rid for rid, row in resources.items()
                                            if row["physical"].get("idle") is True)}



class RecoveryCompositionAdmission:
    """Retain physical proofs and advance them through authorized observations.

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
        region_reservations: Shared physical exclusion ledger for all runtime owners.
        preparation_authorizer: Trusted synchronous owner callback binding exact prepared
            commands to the complete-program reservation before granting a start.
        runtime: Registered runtime for owner-prepared physical motion.
        cca: CCA owning nominal admission and both observation histories.
        preparation: Configured physical observation and execution coverage provider.
    """

    def __init__(self, *, context_provider: Callable | None = None,
                 resource_evidence_provider: Callable | None = None, lock=None,
                 allow_mock_execution: bool = False,
                 allow_nominal_tasks: bool = False,
                 budget_factory: Callable[[], Budget] = Budget,
                 native_history_commit: Callable | None = None,
                 primitive_models_provider: Callable | None = None,
                 region_reservations: RegionReservationLedger | None = None,
                 preparation_authorizer: Callable | None = None,
                 runtime=None, cca=None, preparation=None) -> None:
        self.context_provider = context_provider
        self.resource_evidence_provider = resource_evidence_provider
        self.lock = lock if lock is not None else (runtime.context.admission_lock if runtime is not None else RLock())
        self.allow_mock_execution = allow_mock_execution is True
        self.allow_nominal_tasks = allow_nominal_tasks is True
        self.budget_factory = budget_factory
        self.native_history_commit = native_history_commit
        self.primitive_models_provider = primitive_models_provider
        self.start_validator: Callable | None = None
        self.sessions: dict[str, dict] = {}
        self.last_results: dict[str, dict] = {}
        self.epoch = 0
        self.region_reservations = region_reservations or RegionReservationLedger(lock=self.lock)
        self.preparation_authorizer = preparation_authorizer
        self._mock_region_identity = None
        self._mock_region_revision = 0
        self.runtime = self.cca = self.preparation = None
        self.regions = self.region_reservations
        self._registration_lock = asyncio.Lock()
        if any(value is not None for value in (runtime, cca, preparation)):
            self.configure_live(runtime=runtime, cca=cca, preparation=preparation)

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
        context["region_reservations"] = self.region_reservations.snapshot()
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
            "region_observation": context.get("region_observation"),
            "region_reservations": context.get("region_reservations"),
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
        if self.preparation is not None:
            return await self._register_prepared_motion(request, product_jid=product_jid)
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

    def _region_observation(self, session: dict, context: dict) -> dict:
        observation = context.get("region_observation")
        if observation is None:
            if session["execution_mode"] != "mock" or not self.allow_mock_execution:
                raise ValueError("authoritative_region_observation_unavailable")
            node = session["nodes"][session["node"]]
            state = node["state"]
            active = {**{row["outline_id"]: row for row in session["inputs"]["recovery_events"]},
                      **{row["task_id"]: row for row in session["inputs"]["running_work"]}}
            moving = {active[identity]["resource_id"] for identity in node["running"]}
            observation = {
                "time_exact": context["time_exact"],
                "region_occupancy": deepcopy(state.get("region_occupancy", {})),
                "stationary_resources": sorted(set(state["resources"]) - moving),
            }
            identity = _digest(observation)
            if identity != self._mock_region_identity:
                self._mock_region_identity = identity
                self._mock_region_revision += 1
            observation["revision"] = self._mock_region_revision
        observation = deepcopy(_object(observation, "region_observation"))
        if Fraction(str(observation["time_exact"])) != Fraction(str(context["time_exact"])):
            raise ValueError("stale_region_observation")
        population = set(session["analysis"]["scope"]["included_resources"])
        if not population <= set(context["current_snapshot"]["resources"]):
            raise ValueError("region_observation_scene_incomplete")
        if any(set(values) != population for values in observation["region_occupancy"].values()):
            raise ValueError("region_observation_scene_incomplete")
        return observation

    def _refresh_region_claims(self, session: dict, context: dict) -> None:
        for task_id in session["completed_tasks"]:
            grant = session["grants"].get(task_id, {}).get("grant", {})
            if token := grant.get("region_reservation_token"):
                self.region_reservations.finish(token, success=True)
        if session["analysis"].get("region_mutexes"):
            self.region_reservations.observe(self._region_observation(session, context))

    def _reserve_regions(self, session: dict, task: dict, preparation: dict,
                         program: dict, context: dict, *, commit: bool) -> dict | None:
        mutexes = session["analysis"].get("region_mutexes", [])
        if not mutexes and session["execution_mode"] != "live":
            return None
        relevance = (session["analysis"]["region_relevance"][program["schedule_id"]][task["outline_id"]]
                     if mutexes else {"affected_regions": []})
        peers = {}
        for mutex in mutexes:
            if (task["resource_id"] in mutex["resources"]
                    and mutex["region"] in relevance["affected_regions"]):
                peers.setdefault(mutex["region"], set()).update(mutex["resources"])
        if not peers and session["execution_mode"] != "live":
            return None
        self._refresh_region_claims(session, context)
        if not mutexes:
            self.region_reservations.observe(self._region_observation(session, context))
        token = _digest({"problem_id": session["analysis"]["problem_id"],
                         "product_jid": session["product_jid"],
                         "reference": session["refs"][task["task_id"]]})
        snapshot = self.region_reservations.snapshot()
        if not commit:
            # Preview on a detached ledger; an observation may refresh existing
            # claims, but a preview never acquires a new execution claim.
            ledger = RegionReservationLedger()
            ledger.revision = self.region_reservations.revision
            ledger.claims = deepcopy(self.region_reservations.claims)
            ledger.observation_revision = self.region_reservations.observation_revision
            ledger.observation = deepcopy(self.region_reservations.observation)
        else:
            ledger = self.region_reservations
        claim = ledger.reserve(
            token=token, task_id=task["task_id"], resource_id=task["resource_id"],
            regions=sorted(peers), expected_revision=snapshot["revision"],
            observation_revision=snapshot["observation_revision"],
            conflicting_resources={region: sorted(resources) for region, resources in peers.items()},
        )
        if commit and session["execution_mode"] == "live":
            if self.preparation_authorizer is None:
                ledger.finish(token, success=False)
                raise ValueError("exact_command_reservation_authorizer_unavailable")
            try:
                result = self.preparation_authorizer(deepcopy(preparation), deepcopy(claim))
                if inspect.isawaitable(result) or result is not True:
                    raise ValueError("exact_command_reservation_not_authorized")
            except (ValueError, KeyError, TypeError, OSError, RuntimeError):
                ledger.finish(token, success=False)
                self._invalidate(session, "exact_command_reservation_not_authorized")
                raise
        return claim

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
            if self.preparation is not None:
                self.monitor.invalid_reason = reason
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
            self._refresh_region_claims(original, refreshed)
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
        """Authorize the exact next event with the configured owner proof."""
        if self.preparation is not None:
            return await self._check_prepared_motion(event, sender=sender, commit=commit)
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
                    if not commit:
                        existing.pop("recovery_composition_grant", None)
                        existing["committed"] = False
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
                program = next(row for row in preparation["event_programs"]
                               if Fraction(str(row["start_time"])) == now)
                nominal_task = None
                if task["function_name"] != "execute_recovery_macro":
                    nominal_task = deepcopy(current["nominal_pending_tasks"][task_id])
                    if (nominal_task.get("run_id") != session["nominal_run_id"]
                            or nominal_task["resource_id"] != task["resource_id"]
                            or nominal_task["event_name"] != task["function_name"]
                            or nominal_task["parameters"] != task["params"]):
                        raise ValueError("nominal_task_identity_changed")
                try:
                    claim = self._reserve_regions(session, task, prepared[task_id], program,
                                                  current, commit=commit)
                except ValueError as exc:
                    if str(exc) in {"region_occupied_by_another_resource",
                                    "region_reserved_by_another_resource"}:
                        return self._result("held", str(exc), task_id=task_id)
                    raise
                if commit:
                    grant = {"recovery_composition_ref": reference,
                             "preparation_id": prepared[task_id]["preparation_id"],
                             "schedule_id": program["schedule_id"],
                             "primitive_steps": deepcopy(preparation["primitive_steps"]),
                             "resolved_primitive_steps": [{"primitive": row["primitive"],
                                                           "params": deepcopy(row["resolved_params"])}
                                                          for row in program["step_results"]],
                             "step_results": deepcopy(program["step_results"])}
                    if task["function_name"] != "execute_recovery_macro":
                        grant["run_id"] = session["nominal_run_id"]
                        grant["nominal_task"] = nominal_task
                    if claim is not None:
                        grant["region_reservation_token"] = claim["token"]
                        grant["region_reservation"] = deepcopy(claim)
                    try:
                        self._take(session, edge)
                    except (ValueError, KeyError, TypeError, OSError, RuntimeError):
                        if claim is not None:
                            self.region_reservations.finish(claim["token"], success=False)
                        self._invalidate(session, "native_history_commit_failed")
                        raise
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
        except (ValueError, KeyError, TypeError, IndexError, StopIteration, OSError, RuntimeError) as exc:
            return self._result("inconclusive", str(exc),
                                task_id=event.get("task_id", "") if isinstance(event, dict) else "")

    def observe(self, record: dict, *, sender: str) -> dict:
        """Use authenticated notifications to refresh owner-acknowledged history."""
        if self.preparation is not None:
            return self._observe_prepared_motion(record, sender=sender)
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
                        grant = session["grants"].get(task_id, {}).get("grant", {})
                        if token := grant.get("region_reservation_token"):
                            self.region_reservations.finish(token, success=False)
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
        """Hold new interacting starts until acknowledged execution completes."""
        if self.preparation is not None:
            with self.lock:
                return bool(self.monitor.invalid_reason) or any(
                    session["invalid_reason"] or any(not work["complete"] for work in session["grants"].values())
                    for session in self.sessions.values())
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


    def configure_live(self, *, runtime, cca, preparation) -> None:
        """Bind physical preparation once through the CCA-owned factory.

        A restarted runtime requires a fresh coordinator; retained grants and
        physical history are never migrated or reset during installation.
        """
        if runtime is None or cca is None or preparation is None:
            raise ValueError("Complete owner physical preparation configuration required")
        with self.lock:
            if self.preparation is not None:
                if self.runtime is runtime and self.cca is cca and self.preparation is preparation:
                    return
                raise ValueError("An active physical safety owner belongs to another run")
            if self.sessions or self.epoch or self.lock is not runtime.context.admission_lock:
                raise ValueError("Physical admission installation requires a fresh runtime")
            monitor = cca.safety_monitor.physical_monitor
            if monitor.revision or monitor.history or monitor.invalid_reason:
                raise ValueError("Physical admission installation requires a fresh runtime")
            commands = getattr(runtime, "physical_commands", None)
            admission = getattr(runtime, "admission", None)
            if (self.regions.claims or getattr(admission, "grants", {})
                    or commands is not None and (commands.lock is not self.lock
                                                 or commands.commands or commands.execution_epoch)):
                raise ValueError("Physical admission installation requires a fresh runtime")
            self.runtime, self.cca, self.preparation = runtime, cca, preparation
            self.commands = commands if commands is not None else LiveCommandLedger(self.lock)
            runtime.physical_commands = self.commands
            self.monitor = monitor
            self.observation_revision = 0
        from cais_spade_llm.recovery_framework.kmr_live_safety import KMRPreparedEvidence

        for owner in runtime.resource_agents:
            provider_type = KMRPreparedEvidence if owner.agent_name == "KMR" else PreparedRobotEvidence
            if owner.agent_name == "KMR" or getattr(owner, "_controller", None) is not None:
                provider = provider_type(owner, self.commands)
                provider.execution_observer = self._record_execution
                owner.recovery_composition_evidence_provider = provider
                if owner.agent_name == "KMR":
                    owner.worker.physical_request_guard = provider.request_guard
                controller = getattr(owner, "_controller", None)
                if controller is not None:
                    from cais_spade_llm.recovery_framework.nominal_safety_adapter import (
                        install_nominal_robot_adapter,
                    )

                    install_nominal_robot_adapter(owner, runtime.context.inputs["scene"])
                    controller._physical_dispatch_guard = provider.dispatch_guard
                    controller._physical_dispatch_accepted = provider.dispatch_accepted
            owner.recovery_composition_start_guard = self.holds


    def _capture(self):
        from cais_spade_llm.recovery_framework.gazebo_safety_preparation import capture_checkpoint

        self.preparation.initialize(model_execution=True)
        checkpoint = capture_checkpoint(self.runtime, self.cca, model_execution=True)
        if checkpoint["unresolved"]:
            raise ValueError("Live physical checkpoint unavailable: " + str(checkpoint["unresolved"]))
        with self.lock:
            self.observation_revision += 1
            observation = checkpoint_region_observation(checkpoint, self.observation_revision)
            self.regions.observe(observation)
        return checkpoint, observation


    def _identity(self):
        return fingerprint({"run_id": self.runtime.context.run_id,
            "stopped": getattr(self.runtime, "stopped", False), "admission_epoch": self.epoch,
            "revision": self.runtime.context.revision,
            "resources": self.runtime.context.snapshot(), "products": self.runtime.context.part_tracker,
            "monitor_states": self.cca.safety_monitor.current_states,
            "physical_history": self.monitor.revision, "execution_epoch": self.commands.execution_epoch,
            "specifications": self.cca.predefined_safety_fingerprint})


    async def _register_prepared_motion(self, request: dict, *, product_jid: str) -> dict:
        """Bind event names to exact PCs; each start still requires a fresh proof."""
        async with self._registration_lock:
            try:
                if product_jid != self.runtime.product_jid or not request.get("recovery_id"):
                    raise ValueError("Live registration product or recovery identity changed")
                if not request.get("tasks") or not request.get("recovery_safety_scope_id"):
                    raise ValueError("Live registration requires exact tasks and a safety scope")
                identity = request["recovery_id"]
                with self.lock:
                    previous = self.sessions.get(identity)
                    if previous is not None:
                        if previous["request"] != request or previous["product_jid"] != product_jid:
                            raise ValueError("Live recovery registration changed")
                        return deepcopy(previous["registration"])
                    tasks, refs = {}, {}
                    for raw in request["tasks"]:
                        task = deepcopy(raw)
                        task_id = task["task_id"]
                        rid = task["resource_id"]
                        if any(task_id in session["tasks"] for session in self.sessions.values()):
                            raise ValueError("Live task identity already belongs to a retained registration")
                        if task_id in tasks or self.runtime.jids.get(rid) != task["resource_jid"]:
                            raise ValueError("Live task resource or identity changed")
                        steps = (task.get("primitive_steps") if task["function_name"] != "execute_recovery_macro"
                                 else task["params"].get("primitive_steps"))
                        if not isinstance(steps, list) or not steps:
                            raise ValueError("A recovery event needs its complete primitive composition")
                        task["primitive_steps"] = deepcopy(steps)
                        tasks[task_id] = task
                        refs[task_id] = {"recovery_id": identity, "task_id": task_id,
                            "outline_id": task["outline_id"], "program_hash": fingerprint(steps),
                            "problem_id": fingerprint([self.runtime.context.run_id, request])}
                    registration = {"status": "allowed", "registration_only": True,
                        "requires_fresh_physical_admission": True, "task_refs": refs,
                        "recovery_id": identity, "scope": {"resources": sorted(self.runtime.jids)}}
                    self.sessions[identity] = {"request": deepcopy(request), "product_jid": product_jid,
                        "tasks": tasks, "refs": refs, "registration": registration, "grants": {},
                        "complete": False, "completed_tasks": set(), "invalid_reason": ""}
                    return deepcopy(registration)
            except (ValueError, KeyError, TypeError) as exc:
                return {"status": "inconclusive", "reason": str(exc)}


    def _prepared_stationary_assumptions(self, checkpoint: dict) -> list[dict]:
        assumptions = []
        model_execution = checkpoint.get("model_execution") is True
        if model_execution:
            goals = checkpoint.get("controller_goals")
            if not isinstance(goals, dict) or not goals or any(
                    row.get("holding") is not True or row.get("has_active_goal") is not False
                    or row.get("has_pending_goal") is not False for row in goals.values()):
                raise ValueError("Model stationary assumption requires current idle controller owners")
        for resource, row in checkpoint["observations"].items():
            physical = row["physical"]
            if not model_execution:
                if physical.get("idle") is not True:
                    raise ValueError("Unobserved stationary participant: " + resource)
                continue
            contract = physical.get("stationary_contract")
            if (not isinstance(contract, dict)
                    or contract.get("kind") not in ("idle_commanded_hold", "static_body_and_idle_containment")
                    or contract.get("requires_no_running_tasks") is not True
                    or contract.get("requires_no_active_goals") is not True
                    or contract.get("future_execution_tracking") != "not_established"):
                raise ValueError("Configured stationary model contract unavailable: " + resource)
            if contract["kind"] == "static_body_and_idle_containment" and physical.get("model_static") is not True:
                raise ValueError("Static model assumption requires observed static equipment: " + resource)
            assumptions.append({"resource_id": resource, "stationary_contract": deepcopy(contract),
                "checkpoint_id": checkpoint["checkpoint_id"], "physical_execution_verified": False})
        return assumptions

    def _prepared_composition_context(self, task: dict, checkpoint: dict) -> dict | None:
        # A new zero-based prediction cannot continue observed temporal history.
        # Its first event may use this graph only while the physical history is empty.
        if self.monitor.history or self.monitor.states:
            raise ValueError("existing_physical_history_requires_compatible_continuous_checkpoint")
        if (checkpoint["runtime"].get("admission") or {}).get("running"):
            raise ValueError("ongoing_nominal_programs_require_complete_common_composition")
        session = next((value for value in self.sessions.values()
                        if value.get("tasks", {}).get(task["task_id"]) == task), None)
        if session is not None and any(identity != task["task_id"]
                and identity not in session.get("completed_tasks", set()) for identity in session["tasks"]):
            raise ValueError("remaining_recovery_programs_require_complete_common_composition")
        monitors = checkpoint["runtime"].get("monitors", [])
        if not any(row.get("rules") for row in monitors):
            return None
        if session is None or not callable(self.context_provider):
            raise ValueError("native_common_continuation_context_unavailable")
        context = self.context_provider(session["product_jid"], session["request"]["recovery_id"])
        if not isinstance(context, dict):
            raise ValueError("native_common_continuation_context_unavailable")
        if context.get("composition_inputs", {}).get("running_work"):
            raise ValueError("ongoing_nominal_programs_require_complete_common_composition")
        from cais_spade_llm.recovery_framework.gazebo_safety_preparation import _detach_monitors

        return _detach_monitors(self.runtime, self.cca, context, checkpoint)


    def _ground(self, task: dict, prepared: dict, checkpoint: dict) -> dict:
        rid = task["resource_id"]
        with self.lock:
            if any(row["status"] == "active" or (row["status"] == "pending" and row.get("reservation_token"))
                   for row in self.commands.commands.values()):
                raise ValueError("A pending or active command prevents stationary scene coverage")
        assumptions = self._prepared_stationary_assumptions(checkpoint)
        native = self._prepared_composition_context(task, checkpoint)
        if not prepared.get("steps") or len(prepared["steps"]) != len(task["primitive_steps"]):
            raise ValueError("Prepared program does not cover every registered primitive")
        owner_jid = self.runtime.jids[rid]
        if task.get("resource_jid") != owner_jid or prepared.get("resource_id") != rid:
            raise ValueError("Prepared program differs from its registered resource owner")
        cursor, results = Fraction(0), []
        commands = []
        for index, row in enumerate(prepared["steps"]):
            command = deepcopy(task["primitive_steps"][index])
            source = deepcopy(_object(command.get("source"), "prepared primitive source"))
            for key in ("outline_id", "des_event_id", "event_name"):
                _symbol(source.get(key), "source." + key)
            if source["outline_id"] != task["outline_id"] or source.get("step_index") != index:
                raise ValueError("Prepared primitive source differs from its registered event")
            if "resource_jid" in source and source["resource_jid"] != owner_jid:
                raise ValueError("Prepared primitive source differs from its resource owner")
            binding = _object(row.get("binding"), "prepared step binding")
            if binding.get("resource_id") != rid or binding.get("resource_jid") != owner_jid:
                raise ValueError("Prepared step binding differs from its registered resource owner")
            command["source"] = source
            commands.append(command)
            end = cursor + Fraction(row["joint_trajectory"]["duration_ns"], 1_000_000_000)
            results.append({"primitive": row["primitive"], "resolved_params": deepcopy(row["params"]),
                "success": True, "start_time": float(cursor), "end_time": float(end), "source": source,
                "model_evidence": {"frame": "world", "preparation_id": row["preparation_id"],
                    "joint_trajectory": deepcopy(row["joint_trajectory"]),
                    "continuous_motion": deepcopy(row["continuous_motion"])}})
            if "resolved_motion" in row:
                results[-1]["model_evidence"]["resolved_motion"] = deepcopy(row["resolved_motion"])
            cursor = end
        if cursor <= 0:
            raise ValueError("Live primitive program has no modeled duration")
        resources = {key: {**deepcopy(row["physical"]["observation_state"]),
                          "resource_jid": self.runtime.jids[key]}
                     for key, row in checkpoint["observations"].items()}
        parts = {}
        for name, ledger in checkpoint["runtime"]["part_tracker"].items():
            parts[name] = {**deepcopy(ledger), "current_pose": deepcopy(checkpoint["parts"][name]["pose"]),
                "contained_by": ledger.get("location") if ledger.get("location") in resources else None,
                "stationary_until": float(cursor), "processCompleted_complete": True,
                "processCompleted_evidence": {"source_kind": "validated_runtime_acknowledgements",
                    "complete": True, "checkpoint": checkpoint["checkpoint_id"],
                    "run_id": self.runtime.context.run_id,
                    "acknowledgements_fingerprint": fingerprint(checkpoint["runtime"]["acknowledgements"])}}
        context = {"composition_inputs": {"grounding_inputs": {}}}
        from cais_spade_llm.agents.central_controller.predefined_safety_runtime import (
            bind_predefined_context,
        )

        bind_predefined_context(self.cca, context)
        selected = context["composition_inputs"]["grounding_inputs"]
        program = {"resource_id": rid, "primitive_steps": commands, "step_results": results}
        models = {owner.agent_name: model for owner in self.runtime.resource_agents
                  if (model := owner.get_recovery_safety_primitive_model()) is not None}
        proof = _prepare_grounded_primitive_trace(scene=checkpoint["scene"],
            catalog=selected["catalog"], requirement_scopes=selected["requirement_scopes"],
            programs=[program], snapshot={"resources": resources, "parts": parts},
            geometry=checkpoint["geometry"], horizon=[0., float(cursor)],
            stationary={key: [] if key == rid else [[0., float(cursor)]] for key in resources},
            primitive_models=models, motion_budget=Budget(seconds=30))
        proof["owner_program"] = program
        proof["model_execution_assumptions"] = assumptions
        source = commands[0]["source"]
        if any(key in task.get("params", {}) and task["params"][key] != source[key]
               for key in ("outline_id", "des_event_id", "event_name")):
            raise ValueError("Prepared primitive source differs from its registered task parameters")
        if any(any(command["source"][key] != source[key]
                   for key in ("outline_id", "des_event_id", "event_name")) for command in commands):
            raise ValueError("Prepared task requires one exact event source")
        inputs = {key: deepcopy(proof["frozen"][key]) for key in (
            "scene", "catalog", "requirement_scopes", "programs", "snapshot", "geometry",
            "horizon", "stationary", "task_evidence", "state_evidence",
        ) if key in proof["frozen"]}
        proof["common_composition"] = analyze_grounded_recovery_composition(
            grounding_inputs=inputs,
            recovery_events=[{"outline_id": source["outline_id"], "des_event_id": source["des_event_id"],
                "event_name": source["event_name"], "resource_id": rid, "predecessors": [],
                "primitive_step_indices": list(range(len(commands)))}],
            running_work=[],
            event_start_choices=[{"id": task["task_id"], "starts": {source["outline_id"]: 0.},
                "programs": deepcopy(inputs["programs"]), "stationary": deepcopy(inputs["stationary"])}],
            completion={"resources": {rid: {"current_pose": deepcopy(
                proof["projected_snapshot"]["resources"][rid]["current_pose"])}}, "parts": {}},
            task_monitor_context=native, primitive_models=models, budget=Budget(seconds=30))
        return proof


    async def _check_prepared_motion(self, event: dict, *, sender: str, commit: bool = True) -> dict:
        """Prove every selected physical requirement before granting one full PC."""
        coverage = None
        prepared = None
        composition = None
        try:
            if getattr(self.runtime, "stopped", False):
                raise ValueError("nominal_runtime_history_unavailable")
            reference = event["params"]["recovery_composition_ref"]
            session = self.sessions[reference["recovery_id"]]
            task = session["tasks"][event["task_id"]]
            params = {key: value for key, value in event["params"].items() if key != "recovery_composition_ref"}
            if (sender != task["resource_jid"] or event["resource_jid"] != sender
                    or reference != session["refs"][task["task_id"]] or params != task["params"]
                    or event["function_name"] != task["function_name"]):
                raise ValueError("Live dispatch identity differs from its registered task")
            if session["invalid_reason"]:
                raise ValueError(session["invalid_reason"])
            if task["task_id"] in session["grants"]:
                if task["task_id"] in session["completed_tasks"]:
                    return {"status": "held", "reason": "task_already_completed"}
                if not commit:
                    return {"status": "allowed", "committed": False}
                return {"status": "allowed", "committed": True,
                        "recovery_composition_grant": deepcopy(session["grants"][task["task_id"]]["grant"])}
            if self.holds():
                return {"status": "held", "reason": "active_physical_program_requires_completion"}
            nominal = task["function_name"] != "execute_recovery_macro"
            admission = self.cca._environment_admission(self.runtime)
            native = None
            if nominal:
                native = await admission.check(task["nominal_task"], commit=False)
                if native["status"] != "allowed":
                    return native
            elif self.cca.safety_monitor.safety_rules:
                raise ValueError("recovery_native_commit_contract_unavailable")
            if not nominal:
                monitor = getattr(self.cca, "plan_fsa_monitor", None)
                if monitor is None or task["task_id"] not in monitor._next_task_ids_from_state(monitor.current_state):
                    return {"status": "held", "reason": "plan_task_not_enabled"}
            identity = self._identity()
            checkpoint, observation = await asyncio.to_thread(self._capture)
            owner = next(row for row in self.runtime.resource_agents if str(row.jid) == sender)
            program = {"resource_id": task["resource_id"], "primitive_steps": deepcopy(task["primitive_steps"])}
            prepare_async = getattr(owner, "prepare_recovery_safety_program_async", None)
            prepared = (await prepare_async(program, checkpoint) if callable(prepare_async) else
                        await asyncio.to_thread(owner.prepare_recovery_safety_program, program, checkpoint))
            if prepared.get("status") != "prepared":
                raise ValueError(prepared.get("reason", "Owner physical preparation unavailable"))
            try:
                await asyncio.to_thread(
                    self.preparation.require_execution_coverage,
                    checkpoint, task["resource_id"], prepared=prepared,
                )
            except ValueError:
                provider = getattr(owner, "recovery_composition_evidence_provider", None)
                records = getattr(provider, "execution_coverage_records", {})
                recorded = records.get(fingerprint(prepared)) if isinstance(records, dict) else None
                if (isinstance(recorded, dict)
                        and recorded.get("checkpoint_id") == checkpoint.get("checkpoint_id")
                        and recorded.get("resource_id") == task["resource_id"]):
                    coverage = deepcopy(recorded)
                raise
            coverage = deepcopy(getattr(self.preparation, "last_execution_coverage", None))
            proof = await asyncio.to_thread(self._ground, task, prepared, checkpoint)
            composition = proof.get("common_composition")
            if not isinstance(composition, dict):
                raise ValueError("common_composition_unavailable")
            if composition.get("status") != "allowed":
                return {"status": composition.get("status", "inconclusive"),
                        "reason": composition.get("reason", "common_continuation_unavailable"),
                        "common_composition": deepcopy(composition), "execution_coverage": coverage}
            if not any(choice.get("kind") == "start" and choice.get("status") == "allowed"
                       and choice.get("event_ids") == [task["outline_id"]]
                       for choice in composition.get("choices", [])):
                return {"status": "held", "reason": "common_composition_start_not_enabled",
                        "common_composition": deepcopy(composition), "execution_coverage": coverage}
            verdict = self.monitor.check(proof, budget=Budget(seconds=30))
            if verdict["status"] != "allowed":
                return {**verdict, "common_composition": deepcopy(composition)}
            fresh, observation = await asyncio.to_thread(self._capture)
            from cais_spade_llm.recovery_framework.gazebo_safety_preparation import controller_goal_identity
            from cais_spade_llm.resources.resource_safety_preparation import (
                observation_identity,
                validate_prepared_start,
            )

            for field in ("launch_id", "model_execution", "runtime", "scene"):
                if checkpoint.get(field) != fresh.get(field):
                    raise ValueError("Physical checkpoint identity changed during preparation: " + field)
            if (controller_goal_identity(checkpoint.get("controller_goals"),
                                         model_execution=checkpoint.get("model_execution") is True)
                    != controller_goal_identity(fresh.get("controller_goals"),
                                                model_execution=checkpoint.get("model_execution") is True)):
                raise ValueError("Controller identity or command changed during admission preparation")
            model_execution = checkpoint.get("model_execution") is True
            if model_execution and not checkpoint.get("controller_goals"):
                raise ValueError("Modeled admission requires captured controller identities")
            if (set(checkpoint["observations"]) != set(fresh["observations"])
                    or set(checkpoint.get("parts", {})) != set(fresh.get("parts", {}))):
                raise ValueError("Physical checkpoint population changed during preparation")
            geometry, fresh_geometry = deepcopy(checkpoint.get("geometry")), deepcopy(fresh.get("geometry"))
            spatial_changed = geometry != fresh_geometry
            for rid, old in checkpoint["observations"].items():
                current = fresh["observations"][rid]
                if ({key: value for key, value in old.items() if key != "physical"}
                        != {key: value for key, value in current.items() if key != "physical"}):
                    raise ValueError("Resource ownership or configuration changed during preparation: " + rid)
                old_physical, current_physical = observation_identity(old["physical"]), observation_identity(current["physical"])
                if model_execution and rid == task["resource_id"]:
                    first = prepared["steps"][0]
                    if first["start"].get("model_execution") is not True:
                        raise ValueError("Prepared start lacks its modeled execution binding")
                    for physical in (old["physical"], current["physical"]):
                        validate_prepared_start(physical, first["start"], continuous_motion=first["continuous_motion"])
                if model_execution:
                    if (sorted(row["id"] for row in old_physical["component_bounds"])
                            != sorted(row["id"] for row in current_physical["component_bounds"])):
                        raise ValueError("Observed collision population changed during preparation: " + rid)
                    spatial_fields = ("current_pose", "joint_positions", "component_bounds", "footprint")
                    spatial_changed = spatial_changed or any(old_physical.get(key) != current_physical.get(key)
                                                              for key in spatial_fields)
                    spatial_changed = spatial_changed or any(
                        old_physical["observation_state"].get(key) != current_physical["observation_state"].get(key)
                        for key in ("current_pose", "joint_positions"))
                    for physical in (old_physical, current_physical):
                        for field in (*spatial_fields, "idle", "moving_links"):
                            physical.pop(field, None)
                        for field in ("current_pose", "joint_positions"):
                            physical["observation_state"].pop(field, None)
                    for value in (geometry, fresh_geometry):
                        for field in ("component_bounds", "footprint"):
                            value["resources"][rid].pop(field, None)
                if old_physical != current_physical:
                    raise ValueError("Physical scene changed during admission preparation: " + rid)
            for name, old in checkpoint.get("parts", {}).items():
                parts = [deepcopy(old), deepcopy(fresh["parts"][name])]
                spatial_changed = spatial_changed or parts[0].get("pose") != parts[1].get("pose")
                for part in parts:
                    for field in ("simulation_time", "simulation_stamp", "observed_monotonic"):
                        part.pop(field, None)
                    if isinstance(part.get("geometry_source"), dict):
                        part["geometry_source"].pop("simulation_time", None)
                        if model_execution:
                            part["geometry_source"].pop("models_fingerprint", None)
                    if model_execution:
                        part.pop("pose", None)
                if parts[0] != parts[1]:
                    raise ValueError("Observed part or geometry source changed during preparation: " + name)
                if model_execution:
                    for value in (geometry, fresh_geometry):
                        value["parts"][name].pop("footprint", None)
            if geometry != fresh_geometry:
                raise ValueError("Fixed physical geometry changed during admission preparation")
            if model_execution and spatial_changed:
                # Re-observation may differ numerically under the declared stationary
                # model. Re-ground every AP and the complete continuation; never
                # substitute proximity tolerances or a subset of region predicates.
                current_proof = await asyncio.to_thread(self._ground, task, prepared, fresh)
                for field in ("rules", "clock_version", "owner_program", "valuations"):
                    if fingerprint(proof.get(field)) != fingerprint(current_proof.get(field)):
                        raise ValueError("Fresh modeled AP continuation changed during preparation: " + field)
                boundaries = [{key: deepcopy(row.get(key)) for key in (
                    "time_exact", "phase", "continuous_interval", "rule_cells")}
                    for row in proof["observations"]]
                current_boundaries = [{key: deepcopy(row.get(key)) for key in (
                    "time_exact", "phase", "continuous_interval", "rule_cells")}
                    for row in current_proof["observations"]]
                if boundaries != current_boundaries:
                    raise ValueError("Fresh modeled physical observation boundaries changed during preparation")
                expected_aps = {rule["rule_id"]: {ap["label"] for ap in rule["aps"]}
                                for rule in current_proof["rules"]}
                initial = current_proof["valuations"][0]
                if (set(initial) != set(expected_aps)
                        or any(set(initial[key]) != labels for key, labels in expected_aps.items())
                        or any(type(value) is not bool and not (
                            isinstance(value, list) and len(value) == 1 and type(value[0]) is bool)
                            for values in initial.values() for value in values.values())):
                    raise ValueError("Fresh initial AP truth is incomplete or unknown")
                current_composition = current_proof.get("common_composition", {})
                if (current_composition.get("status") != "allowed" or not any(
                        choice.get("kind") == "start" and choice.get("status") == "allowed"
                        and choice.get("event_ids") == [task["outline_id"]]
                        for choice in current_composition.get("choices", []))):
                    raise ValueError("Fresh common composition has no winning immediate start")
                current_verdict = self.monitor.check(current_proof, budget=Budget(seconds=30))
                if current_verdict.get("status") != "allowed":
                    raise ValueError("Fresh physical monitor continuation is unavailable")
                proof, composition, verdict = current_proof, current_composition, current_verdict
            program = proof["owner_program"]
            request = {key: task[key] for key in ("task_id", "outline_id", "resource_id", "resource_jid")}
            request.update(recovery_id=reference["recovery_id"], primitive_steps=deepcopy(task["primitive_steps"]),
                event_programs=[{"schedule_id": reference["problem_id"], "start_time": 0.,
                    "primitive_steps": deepcopy(task["primitive_steps"]), "step_results": program["step_results"]}])
            request["program_hash"] = fingerprint(request)
            evidence = await owner.prepare_recovery_composition_evidence(request)
            if evidence.get("status") != "prepared":
                raise ValueError(evidence.get("reason", "Live execution preparation unavailable"))
            with self.lock:
                if self.holds():
                    return {"status": "held", "reason": "active_physical_program_requires_completion"}
                if identity != self._identity():
                    raise ValueError("stale_live_admission_snapshot")
                if nominal and native["snapshot_revision"] != admission._revision():
                    raise ValueError("stale_native_admission_snapshot")
                if not commit:
                    return {"status": "allowed", "committed": False, "physical_proof": verdict,
                            "common_composition": deepcopy(composition), "execution_coverage": coverage}
                token = fingerprint([self.runtime.context.run_id, reference, evidence["preparation_id"]])
                reservation = reserve_prepared_program(ledger=self.regions, prepared=proof,
                    resource_id=task["resource_id"], task_id=task["task_id"], token=token, observation=observation)
                try:
                    authorized = owner.recovery_composition_evidence_provider.authorize_preparation(
                        preparation_id=evidence["preparation_id"], reservation_token=token,
                        command_ids=evidence["command_ids"], expected_revision=evidence["command_ledger_revision"])
                    if authorized is not True:
                        raise ValueError("exact_command_reservation_not_authorized")
                except (ValueError, KeyError, TypeError, RuntimeError):
                    self.regions.finish(token, success=False)
                    self.invalidate("exact_command_reservation_not_authorized")
                    raise
                grant = {"recovery_composition_ref": deepcopy(reference),
                    "preparation_id": evidence["preparation_id"], "schedule_id": reference["problem_id"],
                    "primitive_steps": deepcopy(task["primitive_steps"]),
                    "resolved_primitive_steps": [{"primitive": row["primitive"], "params": deepcopy(row["resolved_params"])}
                                                 for row in program["step_results"]],
                    "step_results": deepcopy(program["step_results"]), "region_reservation_token": token,
                    "run_id": self.runtime.context.run_id,
                    "model_execution_assumptions": deepcopy(proof.get("model_execution_assumptions")),
                    "physical_execution_verified": False}
                if nominal:
                    grant["nominal_task"] = deepcopy(task["nominal_task"])
                    try:
                        committed = admission.commit_prepared(task["nominal_task"], native)
                    except (ValueError, KeyError, TypeError, RuntimeError):
                        self.regions.finish(token, success=False)
                        self.invalidate("Nominal admission commitment failed")
                        raise
                    if committed.get("status") != "allowed":
                        self.regions.finish(token, success=False)
                        self.invalidate(committed.get("reason", "Nominal admission commitment failed"))
                        return committed
                session["grants"][task["task_id"]] = {"grant": grant, "proof": verdict,
                    "reservation": reservation, "owner": owner, "executions": {}, "complete": False}
                return {"status": "allowed", "committed": True, "recovery_composition_grant": deepcopy(grant),
                        "physical_proof": verdict, "region_reservation": reservation,
                        "common_composition": deepcopy(composition), "execution_coverage": coverage}
        except (ValueError, KeyError, TypeError, StopIteration, RuntimeError, AnalysisLimit) as exc:
            result = {"status": "inconclusive", "reason": str(exc)}
            if isinstance(prepared, dict):
                result["owner_preparation"] = deepcopy(prepared)
            if isinstance(composition, dict):
                result["common_composition"] = deepcopy(composition)
            if isinstance(coverage, dict):
                result["execution_coverage"] = deepcopy(coverage)
            self.last_results[event.get("task_id", "")] = result
            return result


    def _record_execution(self, *, task_id: str, step_index: int, result: dict) -> None:
        with self.lock:
            for session in self.sessions.values():
                work = session["grants"].get(task_id)
                if work is not None:
                    if step_index in work["executions"]:
                        raise ValueError("Duplicate owner execution evidence")
                    if not work["executions"]:
                        self.regions.activate(work["grant"]["region_reservation_token"])
                    work["executions"][step_index] = deepcopy(result)
                    return
            raise ValueError("Owner executed an unregistered physical task")


    def _observe_prepared_motion(self, record: dict, *, sender: str) -> dict:
        """Commit physical history only from authenticated retained owner execution."""
        with self.lock:
            for session in self.sessions.values():
                task = session["tasks"].get(record.get("task_id"))
                if task is None or sender not in {task["resource_jid"], session["product_jid"]}:
                    continue
                if sender == task["resource_jid"] and (
                        record.get("resource_jid") != sender
                        or record.get("function_name") != task["function_name"]):
                    return {"status": "ignored", "reason": "Live completion resource or function changed"}
                work = session["grants"].get(task["task_id"])
                if work is None or work["complete"]:
                    continue
                reference = record.get("recovery_composition_ref")
                if reference is None:
                    reference = record.get("params", {}).get("recovery_composition_ref")
                if (reference != work["grant"]["recovery_composition_ref"]
                        or record.get("run_id") != self.runtime.context.run_id):
                    return {"status": "ignored", "reason": "Live completion identity or run changed"}
                if sender == session["product_jid"] and record.get("status") != "recovery_acknowledgement":
                    continue
                status = (record.get("observations", {}).get("status")
                          if record.get("status") == "recovery_acknowledgement" else record.get("status"))
                if status == "running":
                    self._plan_event(task, work, "start", status)
                    return {"status": "observed", "kind": "start"}
                if status not in {"completed", "done", "failed", "error", "cancelled"} and not str(status).startswith("failed"):
                    continue
                count = len(task["primitive_steps"])
                success = (status in {"completed", "done"} and set(work["executions"]) == set(range(count))
                           and all(row.get("success") is True for row in work["executions"].values()))
                if status in {"completed", "done"} and not success:
                    return {"status": "inconclusive", "reason": "Owner completion evidence is incomplete"}
                if not success and not any(row.get("observations") for row in work["executions"].values()):
                    reason = "Physical failure has no retained owner observation; execution remains unknown"
                    self.invalidate(reason)
                    return {"status": "inconclusive", "reason": reason,
                            "physical_history_revision": self.monitor.revision}
                if work["executions"]:
                    self._plan_event(task, work, "start", "running")
                self._plan_event(task, work, "done" if success else "fail", status)
                self.monitor.commit(work["proof"], success=success,
                    execution_evidence={"task_id": task["task_id"], "owner_jid": task["resource_jid"],
                                        "steps": deepcopy(work["executions"]), "status": status})
                self.regions.finish(work["grant"]["region_reservation_token"], success=success)
                work["complete"] = True
                if success:
                    session["completed_tasks"].add(task["task_id"])
                    session["complete"] = set(session["tasks"]) <= session["completed_tasks"]
                else:
                    session["invalid_reason"] = "Physical execution failed; observed clearance and revalidation required"
                return {"status": "observed", "success": success, "physical_history_revision": self.monitor.revision}
        return {"status": "ignored", "reason": "No authenticated live physical completion"}


    def _plan_event(self, task: dict, work: dict, kind: str, status: str) -> None:
        if task["function_name"] != "execute_recovery_macro" or kind in work.setdefault("plan_events", set()):
            return
        monitor = getattr(self.cca, "plan_fsa_monitor", None)
        if monitor is None:
            raise ValueError("Recovery plan monitor disappeared during physical execution")
        monitor.process_event(event_type=kind, task_id=task["task_id"],
                              function_name=task["function_name"], resource_jid=task["resource_jid"], status=status)
        work["plan_events"].add(kind)


    async def check_nominal(self, task: dict, product_jid: str, *, commit: bool) -> dict:
        """Use the same physical grant and native continuation for a nominal task."""
        try:
            with self.lock:
                if (product_jid != self.runtime.product_jid
                        or self.runtime.context.pending_tasks.get(task["task_id"]) != task
                        or task.get("run_id") != self.runtime.context.run_id):
                    raise ValueError("Nominal task is not the authenticated pending runtime task")
            owner = next(row for row in self.runtime.resource_agents if row.agent_name == task["resource_id"])
            resolve = getattr(owner, "prepare_nominal_safety_program", None)
            if not callable(resolve):
                raise ValueError("Nominal owner has no complete primitive preparation contract")
            value = resolve(deepcopy(task))
            if inspect.isawaitable(value):
                value = await value
            steps = value["primitive_steps"]
            scope = "nominal_" + self.runtime.context.run_id
            from cais_spade_llm.agents.central_controller.predefined_safety_runtime import (
                predefined_scope,
            )

            predefined_scope(self.cca, scope)
            request = {"recovery_id": "nominal_" + task["task_id"], "recovery_safety_scope_id": scope,
                "tasks": [{"task_id": task["task_id"], "outline_id": task["task_id"],
                    "resource_id": task["resource_id"], "resource_jid": self.runtime.jids[task["resource_id"]],
                    "function_name": task["event_name"], "params": deepcopy(task["parameters"]),
                    "nominal_task": deepcopy(task), "primitive_steps": steps}]}
            registration = await self.register(request, product_jid=product_jid)
            if registration["status"] != "allowed":
                return registration
            return await self.check({"task_id": task["task_id"], "resource_jid": self.runtime.jids[task["resource_id"]],
                "function_name": task["event_name"], "params": {**deepcopy(task["parameters"]),
                    "recovery_composition_ref": registration["task_refs"][task["task_id"]]}},
                sender=self.runtime.jids[task["resource_id"]], commit=commit)
        except (ValueError, KeyError, TypeError, StopIteration) as exc:
            return {"status": "inconclusive", "reason": str(exc)}


    async def execute_nominal(self, *, task: dict, grant: dict) -> dict:
        """Execute the exact owner program and require its actual nominal completion evidence."""
        if grant.get("nominal_task") != task or grant.get("run_id") != self.runtime.context.run_id:
            raise ValueError("Nominal execution differs from its live physical grant")
        owner = next(row for row in self.runtime.resource_agents if row.agent_name == task["resource_id"])
        finish = getattr(owner, "observe_prepared_nominal_completion", None)
        if not callable(finish):
            raise ValueError("Nominal physical completion observer unavailable")
        results = []
        for index, row in enumerate(grant["resolved_primitive_steps"]):
            results.append(await owner.execute_recovery_composition_step(
                primitive=row["primitive"], params=row["params"], task_id=task["task_id"],
                step_index=index, grant=grant, owner=owner))
            if results[-1].get("success") is not True:
                return {"status": "failed", "task_id": task["task_id"], "physical_steps": results}
        result = finish(deepcopy(task), deepcopy(results))
        return await result if inspect.isawaitable(result) else result



LiveSafetyRuntime = RecoveryCompositionAdmission
