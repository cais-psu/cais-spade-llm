"""Runtime environmental matching behind the existing Start System surface."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from uuid import uuid4

from spade.behaviour import CyclicBehaviour
from spade.template import Template

from cais_spade_llm.agents.shared_information.environment_capabilities import (
    CapabilityReplyInbox,
    explore,
    match_intake,
    message,
)
from cais_spade_llm.agents.shared_information.local_dispatch import send_agent_message
from cais_spade_llm.product.environment import EnvironmentProductContext, fingerprint
from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.resources.environment_models import (
    feasibility,
    matches_requirement,
    process_json,
    project_transition,
)

from cais_spade_llm.recovery_framework.reports import LatestReport

logger = logging.getLogger(__name__)
RUN_DIRECTORY = ROOT / "cais_spade_llm/monitor/environment_runs"
_prepared: dict | None = None
_prepared_controllers: dict[str, dict] = {}
_prepared_controllers_lock = threading.Lock()
_preparation_generation = 0


def _preparation_binding(
    setup: dict,
    inputs: dict,
    launch_identity,
    source_fingerprints: dict,
) -> dict:
    """Build the immutable identity for resource preparation ownership."""
    return {
        "configuration_fingerprint": fingerprint([setup, inputs]),
        "launch_identity": deepcopy(launch_identity),
        "source_fingerprints": deepcopy(source_fingerprints),
    }


def _discard_prepared_controllers() -> int:
    """Shut down controllers which were prepared but never claimed."""
    global _preparation_generation
    with _prepared_controllers_lock:
        _preparation_generation += 1
        generation = _preparation_generation
        entries = list(_prepared_controllers.values())
        _prepared_controllers.clear()
    for entry in entries:
        for controller in entry["controllers"].values():
            try:
                controller.shutdown()
            except (RuntimeError, OSError):
                logger.warning("Could not shut down an unclaimed prepared controller")
    return generation


def claim_prepared_environment_controllers(prepared: dict) -> dict:
    """Transfer exactly-bound UR5e controllers to their resource agents once."""
    token = str(prepared.get("prepared_controller_token") or "")
    if not token:
        return {}
    with _prepared_controllers_lock:
        entry = _prepared_controllers.pop(token, None)
    if entry is None:
        return {}
    from cais_spade_llm.recovery_framework.startup import configuration_fingerprints

    current_fingerprints = configuration_fingerprints(prepared["setup"])
    if current_fingerprints != prepared.get("source_fingerprints"):
        for controller in entry["controllers"].values():
            controller.shutdown()
        raise ValueError("Configuration changed during UR5e preparation")
    expected = _preparation_binding(
        prepared["setup"], prepared["inputs"], prepared.get("launch_identity"),
        current_fingerprints,
    )
    if entry["binding"] != expected:
        for controller in entry["controllers"].values():
            controller.shutdown()
        raise ValueError("Prepared UR5e controllers do not match this Start request")
    return entry["controllers"]


def _prepare_environment_controllers(
    context,
    setup: dict,
    launch_identity,
    source_fingerprints: dict,
    generation: int,
) -> tuple[str, list[str]]:
    """Prepare only the UR5e resources present in the selected runtime."""
    from cais_spade_llm.recovery_framework.workflow_execution import _robot_configuration
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        GazeboPickPlaceController,
    )

    robot_rows = {
        row["resource_id"]: row
        for row in context.inputs["scene"]["robots"]
        if row["resource_id"] in context.models
    }
    controllers = {}
    try:
        for resource_id, robot in robot_rows.items():
            config, named_positions, _static = _robot_configuration(
                context.inputs["scene"], robot
            )
            controller = GazeboPickPlaceController(
                robot_name=resource_id,
                node_name=config["node_name"],
                controller_config=config,
                named_positions=named_positions,
                execution_mode="simulation",
                arm_joint_names=config["arm_joint_names"],
                arm_trajectory_topic=config["arm_trajectory_topic"],
                joint_states_topic="/joint_states",
            )
            controllers[resource_id.lower()] = controller
            if not controller.wait_for_services(timeout_sec=30.0):
                detail = str(controller._last_failure_message or "controller not ready")
                raise RuntimeError(f"{resource_id} preparation failed: {detail}")
    except (OSError, RuntimeError, ValueError):
        for controller in controllers.values():
            controller.shutdown()
        raise
    token = uuid4().hex
    binding = _preparation_binding(
        setup, context.inputs, launch_identity, source_fingerprints
    )
    stale = False
    with _prepared_controllers_lock:
        if generation != _preparation_generation:
            stale = True
        else:
            _prepared_controllers[token] = {
                "binding": binding,
                "controllers": controllers,
            }
    if stale:
        for controller in controllers.values():
            controller.shutdown()
        raise RuntimeError("UR5e preparation was cancelled before ownership transfer")
    return token, list(controllers)


def prepare_environment_start(
    setup: dict | None,
    *,
    prewarm_controllers: bool = False,
    launch_identity=None,
    requested_at_unix: float | None = None,
    diagnostic_cca_bypass: bool = False,
) -> None:
    """Capture saved matching inputs without searching or commanding a resource.

    Args:
        setup: Saved experiment settings, or None to clear prepared controllers.
        prewarm_controllers: Prepare the selected simulation controllers.
        launch_identity: Identity of the simulation owned by SystemBridge.
        requested_at_unix: Time of the originating Start System request.
        diagnostic_cca_bypass: Skip CCA approvals for this simulation only.
    """
    global _prepared
    generation = _discard_prepared_controllers()
    if setup is None:
        _prepared = None
        return
    if type(diagnostic_cca_bypass) is not bool or (
        diagnostic_cca_bypass
        and (setup.get("execution_mode") != "simulation" or not launch_identity
             or not prewarm_controllers)
    ):
        raise ValueError("CCA bypass requires an explicitly prepared simulation launch")
    from cais_spade_llm.ui.recovery_setup import validate_setup

    inputs = validate_setup(setup)
    if "completion_conditions" in inputs["product_order"]:
        raise ValueError("Explicit delivery orders use their existing delivery runtime")
    # Validate the product contract before replacing any prepared startup input.
    context = EnvironmentProductContext(
        inputs["scene"], inputs["product_order"], inputs["geometry"], setup["permitted_resources"]
    )
    source_fingerprints = {}
    if prewarm_controllers:
        from cais_spade_llm.recovery_framework.startup import configuration_fingerprints
        source_fingerprints = configuration_fingerprints(setup)
    prepared = {
        "diagnostic_cca_bypass": diagnostic_cca_bypass,
        "setup": deepcopy(setup),
        "inputs": context.inputs,
        "launch_identity": deepcopy(launch_identity),
        "source_fingerprints": source_fingerprints,
        "startup_timing": {
            "start_requested_at_unix": (
                time.time() if requested_at_unix is None else float(requested_at_unix)
            )
        },
    }
    if prewarm_controllers:
        started = time.monotonic()
        token, resources = _prepare_environment_controllers(
            context, setup, launch_identity, source_fingerprints, generation
        )
        prepared["prepared_controller_token"] = token
        prepared["prepared_resources"] = resources
        prepared["startup_timing"]["controller_preparation_wall_time_sec"] = (
            time.monotonic() - started
        )
    _prepared = prepared


def prepared_environment_start() -> dict | None:
    """Return the independent factory input for the requested Start System."""
    return deepcopy(_prepared)


def record_environment_startup_ready(bridge) -> None:
    """Record agent readiness against the originating explicit Start click."""
    ready_at = time.time()
    for product in getattr(bridge, "product_agents", ()):
        runtime = getattr(product, "environment_runtime", None)
        if runtime is None:
            continue
        runtime.startup_timing["agents_ready_at_unix"] = ready_at
        requested = runtime.startup_timing.get("start_requested_at_unix")
        if requested is not None:
            runtime.startup_timing["start_to_readiness_wall_time_sec"] = ready_at - requested
        runtime.save()


class EnvironmentRuntime:
    """Connect ProductAgent discovery, CCA checks, task dispatch, and evidence."""

    def __init__(self, agent, prepared: dict, resources: list) -> None:
        self.diagnostic_cca_bypass = prepared.get("diagnostic_cca_bypass", False)
        if type(self.diagnostic_cca_bypass) is not bool or (
            self.diagnostic_cca_bypass
            and (prepared["setup"].get("execution_mode") != "simulation"
                 or not prepared.get("launch_identity"))
        ):
            raise ValueError("CCA bypass requires an explicitly prepared simulation launch")
        self.agent = agent
        self.product_jid = str(agent.jid).split("/", 1)[0]
        self.context = EnvironmentProductContext(
            **prepared["inputs"], permitted_resources=prepared["setup"]["permitted_resources"]
        )
        self.jids = {
            resource.agent_name: str(resource.jid).split("/", 1)[0] for resource in resources
        }
        if set(self.jids) != set(self.context.models):
            raise ValueError("Environmental matching requires every configured ResourceAgent")
        self.replies: asyncio.Queue = asyncio.Queue()
        self.max_steps = 64
        self.max_search_states = 50_000
        self.stopped = False
        self.resource_agents = list(resources)
        self.active_executions: set[asyncio.Task] = set()
        self.started_wall = time.monotonic()
        self.startup_timing = deepcopy(prepared.get("startup_timing", {}))
        self.planning_wall_time_sec = 0.0
        self.reporting_timing: list[dict] = []
        self.message_loop_timing = deque(maxlen=600)
        self.message_loop_sample_count = 0
        self.message_loop_max_delay_wall_time_sec = 0.0
        self._calculation_executor = None
        self._calculation_stop = threading.Event()
        self._report_inputs = deepcopy(self.context.inputs)
        self._report_initial_products = deepcopy(self.context.initial_product_states)
        self._report_history = {field: [] for field in ('explorations', 'negotiations', 'transitions')}
        self._report_process_models = None
        self._report_processes = None
        self._report_sequence = 0
        self._persisted_sequence = 0
        self._report_lock = threading.Lock()
        self._report_task: asyncio.Task | None = None
        self._pending_report: tuple | None = None
        self.outcome = {"status": "prepared"}
        self.reports = LatestReport(RUN_DIRECTORY, self.context.run_id)
        self.path = self.reports.path
        for resource in resources:
            resource.environment_runtime = self
            resource.environment_context = self.context.resources[resource.agent_name]
        from cais_spade_llm.recovery_framework.workflow_execution import (
            bind_environment_executors,
        )

        bind_environment_executors(self, resources)
        agent.part_tracker = deepcopy(self.context.part_tracker)

    async def calculate_capability(self, function, *args, request: dict, resource_id: str):
        """Evaluate detached capability inputs using at most two owned workers."""
        if self.stopped:
            raise asyncio.CancelledError()
        if self._calculation_executor is None:
            self._calculation_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="environment_capability")
        queued = time.monotonic()
        timing = {"kind": "capability_calculation", "request_id": request["request_id"],
                  "branch_id": request["branch_id"], "resource_id": resource_id}

        def calculate():
            started = time.monotonic()
            timing["queue_wall_time_sec"] = started - queued
            try:
                if self._calculation_stop.is_set():
                    return None
                return function(*args, stopped=self._calculation_stop.is_set)
            finally:
                timing["calculation_wall_time_sec"] = time.monotonic() - started

        try:
            return await asyncio.get_running_loop().run_in_executor(self._calculation_executor, calculate)
        finally:
            self.context.negotiations.append({**timing, "timestamp": time.time()})

    def _close_calculations(self) -> None:
        """Prevent queued work from running after this runtime is stopped."""
        self._calculation_stop.set()
        if self._calculation_executor is not None:
            self._calculation_executor.shutdown(wait=False, cancel_futures=True)
            self._calculation_executor = None

    def snapshot(self) -> dict:
        """Expose read-only runtime models, environmental graph, and progress."""
        activity = {}
        for resource in self.resource_agents:
            tasks = [
                deepcopy(task)
                for task in self.context.pending_tasks.values()
                if task["resource_id"] == resource.agent_name
            ]
            progress = deepcopy(getattr(getattr(resource, "workflow_worker", None), "progress", None))
            activity[resource.agent_name] = {
                "status": "active" if tasks else "idle",
                "tasks": tasks,
                "progress": progress,
            }
        completed_components = [
            part
            for part in self.context.selected_parts
            if self.context.part_tracker[part].get("state") == "assembled"
        ]
        return {
            "diagnostic_cca_bypass": self.diagnostic_cca_bypass,
            'collision_checks_bypassed': [
                *[robot['resource_id'] for robot in self.context.inputs['scene']['robots']
                  if robot.get('cartesian_motion', {}).get('avoid_collisions') is False],
                *(['KMR'] if self.context.inputs['scene']['KMR']['task_execution'].get('avoid_collisions') is False else []),
            ],
            "models": {
                rid: resource.descriptor() for rid, resource in self.context.resources.items()
            },
            "environment_model": deepcopy(self.context.environment_model),
            "outcome": deepcopy(self.outcome),
            "product_states": deepcopy(self.context.part_tracker),
            "initial_product_states": deepcopy(self.context.initial_product_states),
            "requirements": deepcopy(self.context.requirements),
            "resource_activity": activity,
            "machining_countdowns": {
                rid: row["progress"]
                for rid, row in activity.items()
                if rid in {"M1", "M2"} and row["progress"] is not None
            },
            "completed_components": completed_components,
            "component_progress": {
                "completed": len(completed_components),
                "total": len(self.context.selected_parts),
            },
            "timings": self._timing_summary(),
            "startup_timing": deepcopy(self.startup_timing),
            **({"processPlan": deepcopy(self.context.product_order["processPlan"])}
               if self.context.schema_version == 3 else {}),
            "revision": self.context.revision,
            "run_id": self.context.run_id,
        }

    def _timing_summary(self) -> dict:
        totals = {
            "planning_sec": self.planning_wall_time_sec,
            "motion_sec": 0.0,
            "waiting_sec": 0.0,
            "simulation_sec": 0.0,
            "wall_clock_sec": time.monotonic() - self.started_wall,
        }
        for transition in self.context.transitions:
            timing = dict(transition.get("observations", {}).get("timing", {}))
            totals["planning_sec"] += float(
                timing.get("planning_sec", timing.get("planning_wall_time_sec", 0.0)) or 0.0
            )
            totals["motion_sec"] += float(
                timing.get("motion_sec", timing.get("trajectory_duration_sec", 0.0)) or 0.0
            )
            totals["waiting_sec"] += float(timing.get("waiting_sec", 0.0) or 0.0)
            totals["simulation_sec"] += float(
                timing.get("simulation_sec", timing.get("simulation_time_sec", 0.0)) or 0.0
            )
        return totals

    def snapshot_revision(self) -> tuple:
        """Read display revisions without copying resource models or path graphs."""
        environment = self.context.environment_model
        return (
            self.context.run_id,
            self.context.revision,
            tuple(
                (
                    rid,
                    resource.revision,
                    resource.model.get("configuration_revision"),
                    fingerprint(resource.model.get("current_configuration", {})),
                    resource.evidence,
                    tuple(sorted(resource.executors)),
                )
                for rid, resource in self.context.resources.items()
            ),
            environment.get("request_id"),
            environment.get("status"),
            environment.get("closed"),
            *(
                len(environment.get(field, []))
                for field in ("nodes", "edges", "bids", "rejections")
            ),
            id(environment.get("selected_path")),
            id(self.outcome),
        )

    def _report_snapshot(self) -> tuple:
        """Capture mutable state without copying immutable models and history."""
        started = time.monotonic()
        context = self.context
        context.revisions()
        # Completed history and graph entries are append-only. Outcomes are
        # replaced, never edited. The writer may copy these after capture.
        environment_model = {
            key: list(value) if isinstance(value, list) else deepcopy(value)
            for key, value in context.environment_model.items()
        }
        report = {
            "schema_version": context.schema_version, "run_id": context.run_id,
            "inputs": self._report_inputs,
            "permitted_resources": list(context.permitted_resources),
            "models": {rid: resource._revision_inputs[0] for rid, resource in context.resources.items()},
            "executable_tasks": {rid: sorted(resource.executors) for rid, resource in context.resources.items()},
            "initial_product_states": self._report_initial_products,
            **{field: list(getattr(context, field)) for field in ('explorations', 'negotiations', 'transitions')},
            "environment_model": environment_model,
            "pending_tasks": deepcopy(context.pending_tasks), "reservations": dict(context.reservations),
            "final_valuation": context.snapshot(), "final_product_states": deepcopy(context.part_tracker),
            "revision": context.revision,
            "diagnostic_cca_bypass": self.diagnostic_cca_bypass,
            'collision_checks_bypassed': [
                *[robot['resource_id'] for robot in self.context.inputs['scene']['robots']
                  if robot.get('cartesian_motion', {}).get('avoid_collisions') is False],
                *(['KMR'] if self.context.inputs['scene']['KMR']['task_execution'].get('avoid_collisions') is False else []),
            ],
            "startup_timing": deepcopy(self.startup_timing),
            "outcome": self.outcome,
            "reporting_timing": list(self.reporting_timing),
            "message_loop_timing": list(self.message_loop_timing),
            "message_loop_sample_count": self.message_loop_sample_count,
            "message_loop_max_delay_wall_time_sec": self.message_loop_max_delay_wall_time_sec,
        }
        self._report_sequence += 1
        return self._report_sequence, report, time.monotonic() - started

    def _materialize_report(self, report: dict) -> dict:
        """Copy the captured report outside acknowledgement and dispatch handling."""
        memo = {id(report[key]): report[key] for key in ('inputs', 'initial_product_states')}
        memo.update({id(model): model for model in report['models'].values()})
        for field, previous in self._report_history.items():
            for row, (source, copied) in zip(report[field], previous):
                if row is source:
                    memo[id(row)] = copied
        materialized = deepcopy(report, memo)
        for field in self._report_history:
            self._report_history[field] = list(zip(report[field], materialized[field]))
        return materialized

    def _persist_report(self, snapshot: tuple) -> None:
        sequence, report, snapshot_duration = snapshot
        with self._report_lock:
            if sequence <= self._persisted_sequence:
                return
            started = time.monotonic()
            report = self._materialize_report(report)
            materialized = time.monotonic()
            models = report['models']
            if models != self._report_process_models:
                names = {event['event_name'] for model in models.values() for event in model['events']}
                self._report_processes = {name: process_json(models, name) for name in sorted(names)}
                self._report_process_models = models
            write_started = time.monotonic()
            self.reports.save(report, self._report_processes)
            self._persisted_sequence = sequence
            self.reporting_timing.append({"timestamp": time.time(),
                                          "snapshot_wall_time_sec": snapshot_duration,
                                          "materialization_wall_time_sec": materialized - started,
                                          "process_export_wall_time_sec": write_started - materialized,
                                          "write_wall_time_sec": time.monotonic() - write_started,
                                          "persistence_wall_time_sec": time.monotonic() - started})

    def save(self) -> None:
        """Persist a complete snapshot, superseding any older queued report."""
        self._persist_report(self._report_snapshot())

    def queue_save(self) -> None:
        """Queue the latest immutable report without waiting in the agent inbox."""
        if self._report_task is not None and self._report_task.done():
            self._report_task.result()
        self._pending_report = self._report_snapshot()
        if self._report_task is None or self._report_task.done():
            self._report_task = asyncio.create_task(self._write_reports())

    async def _write_reports(self) -> None:
        while self._pending_report is not None:
            snapshot, self._pending_report = self._pending_report, None
            await asyncio.to_thread(self._persist_report, snapshot)

    async def flush_reports(self) -> None:
        """Finish queued persistence before closing or exporting the run."""
        if self._report_task is not None:
            await asyncio.shield(self._report_task)

    async def _measure_message_loop(self) -> None:
        """Sample scheduling delay using wall time, independently of Gazebo time."""
        while not self.stopped:
            expected = time.monotonic() + 0.1
            await asyncio.sleep(0.1)
            delay = max(0., time.monotonic() - expected)
            self.message_loop_sample_count += 1
            self.message_loop_max_delay_wall_time_sec = max(self.message_loop_max_delay_wall_time_sec, delay)
            self.message_loop_timing.append({
                "timestamp": time.time(), "delay_wall_time_sec": delay,
            })

    def stop(self, reason: str = "Stopped by operator") -> None:
        """Close discovery and suppress execution without altering acknowledged state."""
        self.stopped = True
        self._close_calculations()
        partial = {}
        for resource in self.resource_agents:
            worker = getattr(resource, "workflow_worker", None)
            if worker is not None and worker.last_result is not None:
                partial[resource.agent_name] = deepcopy(worker.last_result)
            controller = getattr(resource, "_controller", None)
            cancel_goal = getattr(controller, "_cancel_simulation_goal", None)
            if callable(cancel_goal):
                try:
                    cancel_goal()
                except (RuntimeError, ValueError, OSError):
                    logger.warning("Could not cancel %s controller goal", resource.agent_name)
        cancelled = [
            self.context.cancel_pending(task_id)
            for task_id in tuple(self.context.pending_tasks)
        ]
        if self.outcome.get("status") != "completed":
            self.outcome = {
                "status": "stopped",
                "reason": reason,
                "cancelled_tasks": [task for task in cancelled if task is not None],
                "partial_execution": partial,
            }
        self.context.environment_model["closed"] = True
        for model in self.context.exploration_models.values():
            model["closed"] = True
        for execution in tuple(self.active_executions):
            execution.get_loop().call_soon_threadsafe(execution.cancel)

    async def cancel_owned(self) -> None:
        """Stop persistent workers after execution coroutines have been cancelled."""
        workers = [
            getattr(resource, "workflow_worker", None) for resource in self.resource_agents
        ]
        workers.append(getattr(self, "assembly_worker", None))
        await asyncio.gather(
            *(worker.cancel() for worker in workers if worker is not None),
            return_exceptions=True,
        )

    def install(self, agent) -> None:
        """Install matching and execution inboxes instead of the legacy order planner."""
        replies = Template(metadata={"type": "capability_reply"})
        agent.add_behaviour(CapabilityReplyInbox(), replies)
        execution = (
            Template(metadata={"type": "plan_safety_result"})
            | Template(metadata={"type": "ack"})
            | Template(metadata={"type": "replan_request"})
        )
        agent.add_behaviour(EnvironmentProductLoop(), execution)

    def set_plan(self, task: dict) -> None:
        """Build the existing DAG/FSA interface for one rechecked next task."""
        name = f"environment_{self.context.revision + 1}"
        actor = self.context.resources[task["resource_id"]]
        after, _ = project_transition(
            self.context.models,
            self.context.snapshot(),
            self.context.part_tracker,
            task,
            self.context.product_name,
            self.context.requirements,
        )
        node = {
            "id": name,
            "type": "task",
            "requirement_id": "environment",
            "function_name": task["event_name"],
            "params": deepcopy(task["parameters"]),
            "resource_jid": self.jids[task["resource_id"]],
            "predecessors": [],
            "successors": [],
            "sequence_index": 0,
            "status": "pending",
            "environment_task": deepcopy(task),
            "in_state": actor.valuation.get("resource_state", "idle"),
            "out_state": after[task["resource_id"]].get("resource_state", "idle"),
        }
        self.agent.process_planner.nodes = [node]
        self.agent.process_planner.compile_global_fsa()

    def set_plans(self, tasks: list[dict]) -> None:
        """Expose one concurrently dispatched batch through the existing FSA surface."""
        nodes = []
        for index, task in enumerate(tasks):
            actor = self.context.resources[task["resource_id"]]
            after, _ = project_transition(
                self.context.models,
                self.context.snapshot(),
                self.context.part_tracker,
                task,
                self.context.product_name,
                self.context.requirements,
            )
            nodes.append(
                {
                    "id": task.get("task_id", f"environment_batch_{index + 1}"),
                    "type": "task",
                    "requirement_id": "environment",
                    "function_name": task["event_name"],
                    "params": deepcopy(task["parameters"]),
                    "resource_jid": self.jids[task["resource_id"]],
                    "predecessors": [],
                    "successors": [],
                    "sequence_index": index,
                    "status": "pending",
                    "environment_task": deepcopy(task),
                    "in_state": actor.valuation.get("resource_state", "idle"),
                    "out_state": after[task["resource_id"]].get("resource_state", "idle"),
                }
            )
        self.agent.process_planner.nodes = nodes
        self.agent.process_planner.compile_global_fsa()


class EnvironmentProductLoop(CyclicBehaviour):
    """Explore at runtime, then execute only supported and safety-checked tasks."""

    def _report_kickoff(self, runtime: EnvironmentRuntime) -> None:
        """Release Start System after CCA accepts the first executable work."""
        if getattr(self, "_kickoff_reported", False):
            return
        self._kickoff_reported = True
        self.agent._set_kickoff_result(
            success=True,
            message=(
                "Environmental execution started after CCA validation; "
                "progress remains available through SystemBridge."
            ),
            retries_used=0,
            retries_max=0,
            violations=[],
        )

    async def run(self) -> None:
        runtime = self.agent.environment_runtime
        signature = fingerprint(
            [
                runtime.context.revision,
                runtime.context.revisions(),
                runtime.context.permitted_resources,
                runtime.context.geometry,
            ]
        )
        if runtime.stopped or getattr(self, "_last_attempt", None) == signature:
            await asyncio.sleep(0.25)
            return
        self._last_attempt = signature
        try:
            await self.work(runtime)
        except (
            ValueError,
            KeyError,
            TypeError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError,
        ) as exc:
            runtime.stop(str(exc))
            await runtime.cancel_owned()
            runtime.outcome = {"status": "blocked", "reason": str(exc)}
            logger.warning("Environmental run blocked: %s", exc)
        finally:
            self._last_attempt = fingerprint(
                [
                    runtime.context.revision,
                    runtime.context.revisions(),
                    runtime.context.permitted_resources,
                    runtime.context.geometry,
                ]
            )
            runtime.queue_save()
            await runtime.flush_reports()
            if not getattr(self, "_kickoff_reported", False):
                self._report_kickoff(runtime)

    async def work(self, runtime: EnvironmentRuntime) -> None:
        """Negotiate ready goals and dispatch again after each acknowledgement."""
        context = runtime.context
        discoveries: dict[str, asyncio.Task] = {}
        attempted: dict[str, str] = {}
        failures: dict[str, dict] = {}
        ready: dict[str, dict] = {}
        dependencies: dict[str, dict] = {}
        intake_timeouts = 0
        runtime.intake_assignments = {}
        runtime.admitted_parts = set()
        runtime.waiting_since = {}
        runtime.retained_paths = getattr(runtime, "retained_paths", {})
        self._deferred_acks = []
        self._plan_decision = None
        self._awaiting_plan_request = None
        approval = None
        monitor = asyncio.create_task(runtime._measure_message_loop())
        try:
            await runtime.prepare_execution()
            while not runtime.stopped:
                if approval is not None:
                    if time.monotonic() - approval["requested_monotonic"] > 60:
                        raise TimeoutError("Timed out awaiting plan_safety_result")
                    if self._plan_decision is not None:
                        decision = self._plan_decision
                        self._plan_decision = None
                        self._awaiting_plan_request = None
                        if not await self._dispatch_approved(runtime, approval, decision, attempted):
                            return
                        approval = None
                all_goals = self._negotiation_goals(runtime)
                for key, *_ in all_goals:
                    runtime.waiting_since.setdefault(key, time.monotonic())
                waiting = [part for key, part, *_ in all_goals
                           if key == part and context.part_tracker[part]["location"] == "Storage"]
                source_waiting = self._source_waiting(context, all_goals, runtime.admitted_parts)
                deferred = {part for parts in source_waiting.values() for part in parts}
                goals = [goal for goal in all_goals if goal[0] not in waiting and goal[0] not in deferred]
                for parts in source_waiting.values():
                    if any(part in discoveries or part in ready for part in parts):
                        continue
                    for part in sorted(parts, key=lambda name: runtime.waiting_since[name]):
                        if self._intake_ready(context, part):
                            goals.append(next(goal for goal in all_goals if goal[0] == part))
                            break
                if (waiting and not any(part in discoveries or part in ready for part in waiting)
                        and self._intake_ready(context, waiting[0])):
                    goals.append(("intake", waiting[0], None, None))
                revisions = context.revisions()
                for key, part, desired, resource_goal in goals:
                    runtime.waiting_since.setdefault(key, time.monotonic())
                    if key in discoveries or key in ready:
                        continue
                    signature = self._goal_signature(context, key, dependencies.get(key), revisions)
                    if attempted.get(key) == signature:
                        continue
                    attempted[key] = signature
                    discoveries[key] = asyncio.create_task(
                        match_intake(runtime, self, waiting) if key == "intake" else
                        self._negotiate_goal(runtime, key, part, desired, resource_goal)
                    )
                for key, discovery in tuple(discoveries.items()):
                    if not discovery.done():
                        continue
                    discoveries.pop(key)
                    result = discovery.result()
                    dependencies[key] = result.get("dependencies", {})
                    if result.get("status") == "stale":
                        attempted.pop(key, None)
                        if key in waiting:
                            attempted.pop("intake", None)
                        continue
                    if key == "intake":
                        intake_timeouts = intake_timeouts + 1 if result.get("status") == "timeout" else 0
                        if 0 < intake_timeouts < 3:
                            # A missing reply says nothing about machine availability.
                            # Renew the request instead of caching an empty offer set.
                            attempted.pop(key, None)
                            continue
                    attempted[key] = self._goal_signature(context, key, dependencies[key], revisions)
                    if key == "intake":
                        offers = [offer for offer in result.get("offers", []) if offer["available"]]
                        if offers:
                            selected = min(offers, key=lambda offer: (
                                not offer["executable"],
                                self._machine_workload(runtime, offer["resource_id"]),
                                runtime.waiting_since[offer["part_name"]],
                            ))
                            part = selected["part_name"]
                            desired = next(goal[2] for goal in all_goals if goal[0] == part)
                            discoveries[part] = asyncio.create_task(
                                self._negotiate_goal(runtime, part, part, desired, None)
                            )
                            context.negotiations.append({
                                "kind": "selection", "scope": "intake", "request_id": result["request_id"],
                                "offer": deepcopy(selected), "timestamp": time.time(),
                            })
                        else:
                            failures[key] = result
                        continue
                    if result.get("status") == "planned" and result.get("executable"):
                        failures.pop(key, None)
                        ready[key] = result
                    else:
                        failures[key] = result
                pending = []
                new_admissions = set()
                for key in sorted(ready, key=lambda key: self._ready_priority(runtime, key, ready[key])):
                    if approval is not None:
                        break
                    result = ready[key]
                    task = result["tasks"][0]
                    try:
                        prepared = context.prepare(task)
                    except ValueError as exc:
                        reason = str(exc)
                        if reason.startswith("Task conflicts with active reservations:"):
                            if result.get("waiting_reason") != reason:
                                context.negotiations.append({
                                    "kind": "waiting", "request_id": task.get("offer_request_id"),
                                    "part_name": task.get("part_name"), "reason": reason,
                                    "timestamp": time.time(),
                                })
                                result["waiting_reason"] = reason
                            continue
                        ready.pop(key)
                        attempted.pop(key, None)
                        if key in waiting:
                            attempted.pop("intake", None)
                        context.negotiations.append({
                            "kind": "offer_rejected", "request_id": task.get("offer_request_id"),
                            "resource_id": task["resource_id"], "reason": reason,
                            "timestamp": time.time(),
                        })
                    else:
                        ready.pop(key)
                        pending.append(prepared)
                        if key == task["part_name"]:
                            new_admissions.add(prepared["task_id"])
                        machine = self._offered_machine(runtime, result)
                        if machine is not None and context.part_tracker[task["part_name"]]["location"] == "Storage":
                            runtime.intake_assignments[task["part_name"]] = machine
                        context.negotiations.append({
                            "kind": "dispatch", "request_id": task.get("offer_request_id"),
                            "task_id": prepared["task_id"], "resource_id": task["resource_id"],
                            "event_name": task["event_name"], "part_name": task.get("part_name"),
                            "offered_machine": machine, "timestamp": time.time(),
                            "waiting_wall_time_sec": time.monotonic() - runtime.waiting_since.pop(key, time.monotonic()),
                        })
                if pending:
                    runtime.set_plans(list(context.pending_tasks.values()))
                    request_id = uuid4().hex
                    approval = {
                        "request_id": request_id, "tasks": pending, "new_admissions": new_admissions,
                        "requested_monotonic": time.monotonic(), "requested_at_unix": time.time(),
                    }
                    if runtime.diagnostic_cca_bypass:
                        decision = {
                            "ok": True, "request_id": request_id,
                            "diagnostic_cca_bypass": True,
                        }
                        if not await self._dispatch_approved(runtime, approval, decision, attempted):
                            return
                        approval = None
                    else:
                        payload = self.agent._build_plan_validation_payload(
                            skip_revalidation=False, request_id=request_id,
                            validation_scope="active_window", composition_backend="explicit_fsa_dfa",
                        )
                        self._awaiting_plan_request = request_id
                        await send_agent_message(self, message(self.agent.cca_jid, "plan_safety_check", payload))
                if not discoveries and not context.pending_tasks:
                    if context.outstanding() is None and _robots_at_home(context):
                        context.environment_model.update(status="completed", closed=True)
                        runtime.outcome = {"status": "completed", "tasks": []}
                        return
                    if not ready and all(
                        attempted.get(key) == self._goal_signature(context, key, dependencies.get(key), revisions)
                        for key, *_ in goals
                    ):
                        unavailable = [result for result in failures.values()
                                       if result.get("tasks") and not result.get("executable")]
                        runtime.outcome = {
                            "status": "execution_unavailable" if unavailable else "blocked",
                            "reason": "No validated RA offer establishes remaining requirements",
                            "executable": False,
                            "tasks": [deepcopy(task) for result in unavailable for task in result["tasks"]],
                            "details": deepcopy(failures),
                        }
                        return
                if not await self._collect_acknowledgement(runtime):
                    return
        except asyncio.CancelledError:
            runtime.stop("Environmental work cancelled")
            await runtime.cancel_owned()
            raise
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            self._awaiting_plan_request = None
            self._plan_decision = None
            if approval is not None:
                for task in approval["tasks"]:
                    context.cancel_pending(task["task_id"])
            for discovery in discoveries.values():
                discovery.cancel()
            await asyncio.gather(*discoveries.values(), return_exceptions=True)
            if runtime.outcome.get("status") == "completed":
                runtime._close_calculations()

    async def _dispatch_approved(self, runtime, approval: dict, decision: dict, attempted: dict) -> bool:
        """Recheck and send approved work without another scheduling cycle."""
        context = runtime.context
        context.negotiations.append({
            "kind": "CCA_bypassed" if runtime.diagnostic_cca_bypass else "CCA",
            "request_id": approval["request_id"],
            "task_ids": [task["task_id"] for task in approval["tasks"]],
            "requested_at_unix": approval["requested_at_unix"],
            "waiting_wall_time_sec": time.monotonic() - approval["requested_monotonic"],
            "timestamp": time.time(), "decision": deepcopy(decision),
        })
        if decision.get("ok") is not True:
            runtime.stop("CCA rejected negotiated work")
            await runtime.cancel_owned()
            runtime.outcome = {"status": "blocked", "reason": "CCA rejected negotiated work", "details": decision}
            return False
        self._report_kickoff(runtime)
        for task in approval["tasks"]:
            if runtime.stopped:
                return False
            if (context.pending_for(task["task_id"]) != task
                    or not context.relevant_revisions_match(task) or not context.allows_task(task)):
                context.cancel_pending(task["task_id"])
                attempted.clear()
                continue
            context.negotiations.append({
                "kind": "task_sent", "task_id": task["task_id"],
                "resource_id": task["resource_id"], "event_name": task["event_name"],
                "timestamp": time.time(),
            })
            await send_agent_message(self, message(runtime.jids[task["resource_id"]], "task", task))
            if task["task_id"] in approval["new_admissions"]:
                runtime.admitted_parts.add(task["part_name"])
        if runtime.stopped:
            return False
        runtime.outcome = {"status": "executing", "tasks": deepcopy(list(context.pending_tasks.values()))}
        runtime.queue_save()
        return True

    @staticmethod
    def _source_waiting(context, goals: list[tuple], admitted: set[str]) -> dict[str, list[str]]:
        """Group untouched initial outputs for incremental resource-owned pickup."""
        sources: dict[str, list[str]] = {}
        for key, part, _desired, resource_goal in goals:
            if resource_goal is not None or key != part or part in admitted:
                continue
            source = context.initial_product_states[part]["location"]
            if (source in context.resources and source != "Storage"
                    and context.part_tracker[part]["location"] == source):
                sources.setdefault(source, []).append(part)
        return sources

    @staticmethod
    def _intake_ready(context, part: str) -> bool:
        """Check an enabled resource-owned handoff from the current source."""
        source = context.contact_resource(part)
        desired = next(step for step in context.requirements[part]
                       if not matches_requirement(context.part_tracker[part], step))
        for rid in context.resources[source].model["neighbors"]:
            actor = context.resources[rid]
            for offer in actor.alternatives(context, context.snapshot(), context.part_tracker, part, desired):
                if (offer["status"] == "FEASIBLE" and offer["executable"]
                        and (offer["products"][part]["location"] != source
                             or (offer["task"]["event_name"] == "pick_approach"
                                 and offer["task"]["parameters"].get("origin_resource_location") == source))
                        and not context._task_reservations(offer["task"], part).intersection(context.reservations)):
                    return True
        return False

    @staticmethod
    def _machine_workload(runtime, machine: str | None) -> int:
        return sum(assigned == machine and runtime.context.part_tracker[part]["state"] != "assembled"
                   for part, assigned in runtime.intake_assignments.items())

    @staticmethod
    def _goal_signature(context, key: str, dependencies: dict | None, revisions: dict) -> str:
        observed = {rid: revisions[rid] for rid in dependencies} if dependencies else revisions
        return fingerprint([observed, context.part_tracker.get(key), context.permitted_resources])

    @staticmethod
    def _offered_machine(runtime, result: dict) -> str | None:
        for task in result["tasks"]:
            if task["resource_id"] in runtime.context.machine_ids:
                return task["resource_id"]
            for field in ("target_resource", "destination_location"):
                resource = task["parameters"].get(field)
                if resource in runtime.context.machine_ids:
                    return resource
        cached = runtime.retained_paths.get(result["tasks"][0].get("part_name"), {})
        for task in cached.get("path", []):
            if task["resource_id"] in runtime.context.machine_ids:
                return task["resource_id"]
        return None

    def _ready_priority(self, runtime, key: str, result: dict) -> tuple:
        context = runtime.context
        task = result["tasks"][0]
        actor = context.resources[task["resource_id"]]
        age = runtime.waiting_since.get(key, time.monotonic())
        if key.startswith("resource:") or actor.valuation.get("held_part") is not None:
            return (0, 0, age)
        if context.part_tracker[task["part_name"]]["location"] != "Storage":
            return (1, 0, age)
        machine = self._offered_machine(runtime, result)
        workload = self._machine_workload(runtime, machine)
        return (2, workload, age)

    def _negotiation_goals(self, runtime: EnvironmentRuntime) -> list[tuple]:
        """Read unmet effects and resource goals without prescribing a route."""
        context = runtime.context
        goals = []
        for part in context.selected_parts:
            if f"part:{part}" in context.reservations:
                continue
            desired = next((step for step in context.requirements[part]
                            if not matches_requirement(context.part_tracker[part], step)), None)
            if desired is not None:
                goals.append((part, part, desired, None))
        for resource_goal in _robot_home_requirements(context):
            rid = resource_goal["resource_id"]
            state = context.resources[rid].valuation
            if (state["held_part"] is not None
                    or state["resource_state"] not in {"idle", "placed"}
                    or f"resource:{rid}" in context.reservations):
                continue
            if all(state[field] == value for field, value in resource_goal["values"].items()):
                continue
            part = context.selected_parts[0]
            goals.append((f"resource:{rid}", part, context.requirements[part][-1], resource_goal))
        return goals

    async def _negotiate_goal(self, runtime, key, part, desired, resource_goal):
        """Ask the owning RA to revalidate a retained offer or discover a new path."""
        started = time.monotonic()
        cache = runtime.retained_paths.get(key, {})
        desired_id = fingerprint([desired, resource_goal])
        capabilities = {rid: [resource.model, sorted(resource.executors)]
                        for rid, resource in runtime.context.resources.items()}
        try:
            if (cache.get("desired") == desired_id and cache.get("path")
                    and all(capabilities.get(rid) == value
                            for rid, value in cache.get("capabilities", {}).items())
                    and cache.get("permitted_resources") == runtime.context.permitted_resources):
                context = runtime.context
                valuation = context.snapshot()
                while cache["path"]:
                    task = cache["path"][0]
                    try:
                        after, products = project_transition(
                            context.models, valuation, context.part_tracker, task,
                            context.product_name, context.requirements,
                        )
                    except ValueError:
                        break
                    if after != valuation or products != context.part_tracker:
                        break
                    # Another handling cycle may already have acknowledged this
                    # resource effect while this part was still upstream.
                    cache["path"].pop(0)
                    context.negotiations.append({
                        "kind": "retained_step_satisfied", "part_name": part,
                        "resource_id": task["resource_id"], "event_name": task["event_name"],
                        "revision": context.revision, "timestamp": time.time(),
                    })
                if cache["path"]:
                    result = await explore(
                        runtime, self, part=part, desired=desired, resource_goal=resource_goal,
                        candidate=cache["path"][0], timeout=15,
                    )
                    if result.get("status") == "planned" and result.get("executable"):
                        result["tasks"][0]["part_name"] = part
                        return result
                    if result.get("status") == "stale":
                        return result
            result = await explore(runtime, self, part=part, desired=desired, resource_goal=resource_goal)
            if result.get("status") == "planned":
                for task in result["tasks"]:
                    task["part_name"] = part
                if result.get("executable"):
                    runtime.retained_paths[key] = {
                        "desired": desired_id, "path": deepcopy(result["tasks"]),
                        "capabilities": {
                            rid: deepcopy(capabilities[rid])
                            for rid in {peer for task in result["tasks"]
                                        for peer in runtime.context._task_participants(task)}
                        },
                        "permitted_resources": deepcopy(runtime.context.permitted_resources),
                    }
                else:
                    runtime.retained_paths.pop(key, None)
            return result
        finally:
            runtime.planning_wall_time_sec += time.monotonic() - started

    async def _collect_acknowledgement(self, runtime: EnvironmentRuntime) -> bool:
        """Commit one completed task without waiting for unrelated active work."""
        if runtime.stopped:
            return False
        msg = self._deferred_acks.pop(0) if self._deferred_acks else await self.receive(timeout=0.05)
        if msg is None:
            return True
        sender = str(msg.sender).split("/", 1)[0]
        if (not runtime.diagnostic_cca_bypass and msg.metadata.get("type") == "replan_request"
                and sender == str(self.agent.cca_jid).split("/", 1)[0]):
            runtime.stop("CCA interrupted execution")
            return False
        if (msg.metadata.get("type") == "plan_safety_result"
                and sender == str(self.agent.cca_jid).split("/", 1)[0]):
            payload = json.loads(msg.body)
            expected = getattr(self, "_awaiting_plan_request", None)
            if expected is not None and payload.get("request_id") == expected:
                self._plan_decision = payload
            return True
        if msg.metadata.get("type") != "ack":
            return True
        payload = json.loads(msg.body)
        task = runtime.context.pending_for(payload.get("task_id"))
        if task is None or sender != runtime.jids[task["resource_id"]]:
            return True
        runtime.context.negotiations.append({
            "kind": "ack_received", "task_id": task["task_id"],
            "resource_id": task["resource_id"], "status": payload.get("status"),
            "timestamp": time.time(),
        })
        if payload.get("status") in {"accepted", "running"}:
            return True
        if payload.get("status") != "completed":
            runtime.stop("Resource did not complete negotiated work")
            await runtime.cancel_owned()
            runtime.outcome = {"status": "blocked", "reason": "Resource did not complete negotiated work", "details": payload}
            return False
        runtime.context.acknowledge(payload["acknowledgement"])
        for cached in runtime.retained_paths.values():
            path = cached.get("path", [])
            if path and all(path[0].get(key) == task.get(key)
                            for key in ("resource_id", "event_id", "parameters", "part_name")):
                path.pop(0)
        self.agent.part_tracker = deepcopy(runtime.context.part_tracker)
        runtime.context.negotiations.append({
            "kind": "acknowledgement", "task_id": task["task_id"],
            "resource_id": task["resource_id"], "revision": runtime.context.revision,
            "timestamp": time.time(),
        })
        for node in self.agent.process_planner.nodes:
            if node["id"] == task["task_id"]:
                node["status"] = "completed"
        runtime.queue_save()
        return True

    async def receive_from(
        self, sender: str, kind: str, *, timeout: float, request_id: str | None = None
    ) -> dict:
        """Authenticate control replies and honor CCA interruptions while waiting."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = await self.receive(timeout=min(1, deadline - time.monotonic()))
            if self.agent.environment_runtime.stopped:
                raise ValueError("Environmental run stopped")
            if msg is None:
                continue
            if kind != "ack" and msg.metadata.get("type") == "ack":
                self._deferred_acks.append(msg)
                continue
            actual = str(msg.sender).split("/", 1)[0]
            if (
                msg.metadata.get("type") == "replan_request"
                and actual == str(self.agent.cca_jid).split("/", 1)[0]
            ):
                self.agent.environment_runtime.stop("CCA interrupted execution")
                raise ValueError("CCA interrupted execution")
            if msg.metadata.get("type") == kind and actual == str(sender).split("/", 1)[0]:
                payload = json.loads(msg.body)
                if request_id is None or payload.get("request_id") == request_id:
                    return payload
        raise TimeoutError(f"Timed out awaiting {kind}")


def _robot_home_requirements(context: EnvironmentProductContext) -> list[dict]:
    """Read empty home requirements from each robot's owned marked state."""
    goals = []
    for rid, resource in context.resources.items():
        if "held_part" not in resource.model["state_variables"]:
            continue
        for condition in resource.model["marked_state_conditions"]:
            goals.append({
                "resource_id": rid,
                "values": {field: rule["equals"] for field, rule in condition.items()},
            })
    return goals


def _robots_at_home(context: EnvironmentProductContext) -> bool:
    """Require acknowledged idle, empty home states for every configured robot."""
    return all(
        all(context.resources[goal["resource_id"]].valuation[field] == value
            for field, value in goal["values"].items())
        for goal in _robot_home_requirements(context)
    )


def _kmr_at_storage(context: EnvironmentProductContext) -> bool:
    """Require an empty, idle KMR at its configured home location."""
    state = context.snapshot()["KMR"]
    return (
        state["resource_location"] == "Storage"
        and state["resource_state"] == "idle"
        and state["held_part"] is None
    )


async def execute_environment_task(behaviour, msg, task: dict) -> None:
    """Execute a declared task after resource checks and the selected CCA mode."""
    agent = behaviour.agent
    runtime = getattr(agent, "environment_runtime", None)
    if runtime is None or str(msg.sender).split("/", 1)[0] != runtime.product_jid:
        return
    context, actor = runtime.context, agent.environment_context
    task_id = task.get("task_id")

    async def ack(status: str, **extra) -> None:
        context.negotiations.append({
            "kind": "ack_sent", "task_id": task_id, "resource_id": actor.resource_id,
            "status": status, "timestamp": time.time(),
        })
        await send_agent_message(
            behaviour,
            message(runtime.product_jid, "ack", {"task_id": task_id, "status": status, **extra}),
        )

    observations = None
    context.negotiations.append({
        "kind": "task_received", "task_id": task_id, "resource_id": actor.resource_id,
        "timestamp": time.time(),
    })
    try:
        if task_id in actor.validated_completions:
            expected = {**task, "status": "completed"}
            if expected != actor.validated_completions[task_id]:
                raise ValueError("Conflicting repeated task")
            await ack("completed", acknowledgement=expected)
            return
        if (
            runtime.stopped
            or context.pending_for(task_id) != task
            or task["resource_id"] != actor.resource_id
        ):
            raise ValueError("Task does not match current prepared execution")
        name = task["event_name"]
        if name not in actor.executors or not context.relevant_revisions_match(task):
            raise ValueError("Controller unavailable or resource configuration changed")
        part = task["parameters"].get("part_name") or task.get("part_name") or context.selected_parts[0]
        status, reasons = feasibility(actor.model, task, context.geometry.get(part, {}))
        if status != "FEASIBLE":
            raise ValueError("; ".join(reasons))
        project_transition(
            context.models,
            context.snapshot(),
            context.part_tracker,
            task,
            context.product_name,
            context.requirements,
        )
        if (
            await actor.start_validators[name](
                deepcopy(task), context.snapshot(), context.geometry.get(part, {})
            )
            is not True
        ):
            raise ValueError("Resource start conditions have not been validated")
        if runtime.diagnostic_cca_bypass:
            decision = "allow"
            context.negotiations.append({
                "kind": "resource_safety_bypassed", "task_id": task_id,
                "resource_id": actor.resource_id, "timestamp": time.time(),
            })
        else:
            await send_agent_message(
                behaviour,
                message(
                    agent.cca_jid,
                    "resource_event",
                    {
                        "task_id": task_id,
                        "resource_jid": str(agent.jid),
                        "function_name": name,
                        "params": task["parameters"],
                        "status": "safety_check",
                    },
                ),
            )
            context.negotiations.append({
                "kind": "resource_safety_requested", "task_id": task_id,
                "resource_id": actor.resource_id, "timestamp": time.time(),
            })
            decision = await asyncio.wait_for(agent._wait_for_safety_decision(task_id), 60)
            context.negotiations.append({
                "kind": "resource_safety_decision", "task_id": task_id,
                "resource_id": actor.resource_id, "timestamp": time.time(), "decision": decision,
            })
        if (
            decision != "allow"
            or runtime.stopped
            or not context.relevant_revisions_match(task)
        ):
            raise ValueError("CCA permission missing or execution context changed")
        await ack("running")
        if not runtime.diagnostic_cca_bypass:
            await send_agent_message(
                behaviour,
                message(
                    agent.cca_jid,
                    "resource_event",
                    {
                        "task_id": task_id,
                        "resource_jid": str(agent.jid),
                        "function_name": name,
                        "params": task["parameters"],
                        "status": "running",
                    },
                ),
            )
        context.negotiations.append({
            "kind": "execution_started", "task_id": task_id,
            "resource_id": actor.resource_id, "event_name": name,
            "part_name": part, "timestamp": time.time(),
        })
        execution = asyncio.create_task(actor.executors[name](deepcopy(task)))
        runtime.active_executions.add(execution)
        try:
            observations = await asyncio.wait_for(execution, agent.tool_timeout_s)
        finally:
            runtime.active_executions.discard(execution)
        if runtime.stopped or actor.completion_validators[name](task, observations) is not True:
            raise ValueError("Resource completion evidence did not validate")
        if not context.relevant_revisions_match(task):
            raise ValueError("Resource state changed during execution")
        context.negotiations.append({
            "kind": "execution_completed", "task_id": task_id,
            "resource_id": actor.resource_id, "event_name": name,
            "part_name": part, "timestamp": time.time(),
        })
        completed = {**task, "status": "completed"}
        actor.validated_completions[task_id] = completed
        actor.completion_observations[task_id] = deepcopy(observations)
        if not runtime.diagnostic_cca_bypass:
            await send_agent_message(
                behaviour,
                message(
                    agent.cca_jid,
                    "resource_event",
                    {
                        "task_id": task_id,
                        "resource_jid": str(agent.jid),
                        "function_name": name,
                        "params": task["parameters"],
                        "status": "completed",
                    },
                ),
            )
        await ack("completed", acknowledgement=completed)
    except (ValueError, KeyError, TypeError, RuntimeError, OSError, TimeoutError) as exc:
        if observations is not None:
            actor.completion_observations[task_id] = deepcopy(observations)
        await ack(
            "blocked",
            reason=str(exc),
            execution_evidence=deepcopy(observations),
        )
