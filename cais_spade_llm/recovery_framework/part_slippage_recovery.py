from __future__ import annotations

"""ProductAgent-owned stages for the functional Part slippage experiment."""

import asyncio
import hashlib
import json
import logging
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from openai import OpenAIError

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.capture_files import (
    RAW_SCHEMA_VERSION,
    pending_nodes,
    write_capture,
)
from cais_spade_llm.recovery_framework.capture_files import completed_tasks as _completed_tasks
from cais_spade_llm.recovery_framework.diagnostics import write_record
from cais_spade_llm.recovery_framework.failure_grounding import (
    apply_grounding,
    belief_key,
    belief_rows,
    fault_identity,
    ground_failure,
    has_raw_failure_facts,
    restore_beliefs,
    update_beliefs,
)
from cais_spade_llm.recovery_framework.recovery_observations import (
    capture_observations,
    observation_fingerprint,
    observation_summary,
    offline_preparation_missing,
)
from cais_spade_llm.ui.recovery_setup import FAILURE_SCENARIOS

logger = logging.getLogger(__name__)
# A first capture can miss an observation; retakes refine grounding until it settles.
_GROUNDING_RETAKES = 3


def _recorded_checkpoint(observations: dict) -> dict:
    failure = observations.get("failure", {})
    result = {"name": failure["checkpoint"]} if isinstance(failure.get("checkpoint"), str) else {}
    result.update({key: deepcopy(failure[key]) for key in (
        "simulation_started_sec", "simulation_elapsed_sec", "processing_time_sec", "processing_fraction",
        "process", "result", "process_completed", "observation_status") if key in failure})
    task_id = failure.get("task_id")
    acknowledgement = _completed_tasks(observations).get(task_id, {})
    if acknowledgement:
        result["completed_task"] = {key: deepcopy(acknowledgement[key]) for key in
            ("task_id", "resource_id", "event_name", "status") if key in acknowledgement}
    if observations.get("processing_observations"):
        from cais_spade_llm.resources.machine.machine_jobs import public_processing_observations
        result["processing_observations"] = {rid: public_processing_observations(record)
            for rid, record in observations["processing_observations"].items()}
    return result


def _prompt_safety_rules(selected_files: list[str]) -> list[dict]:
    rules, requirements = [], {}
    for source_path in selected_files:
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("NEEDS_CONTEXT: selected safety specification requires a file path")
        path = Path(source_path)
        if not path.is_absolute():
            path = ROOT / path
        try:
            content = path.read_bytes()
            payload = json.loads(content)
        except (OSError, ValueError) as exc:
            raise ValueError(f"NEEDS_CONTEXT: cannot read selected safety specification {source_path}: {exc}") from exc
        catalog = payload.get("catalog") if isinstance(payload, dict) else None
        specifications = catalog.get("specifications") if isinstance(catalog, dict) else None
        if not isinstance(specifications, list) or not specifications:
            raise ValueError(f"NEEDS_CONTEXT: selected safety specification {source_path} requires catalog.specifications")
        for specification in specifications:
            if not isinstance(specification, dict) or any(
                not isinstance(specification.get(field), str) or not specification[field].strip()
                for field in ("id", "requirement")
            ):
                raise ValueError(f"NEEDS_CONTEXT: selected safety specification {source_path} requires an exact id and requirement")
            rule_id, requirement = specification["id"], specification["requirement"]
            if rule_id in requirements and requirements[rule_id] != requirement:
                raise ValueError(f"NEEDS_CONTEXT: selected safety specifications disagree on requirement for {rule_id}")
            requirements[rule_id] = requirement
            rules.append({"id": rule_id, "requirement": requirement, "source_path": source_path,
                          "source_sha256": hashlib.sha256(content).hexdigest()})
    return rules


def diagnostic_setup(setup: dict) -> bool:
    """Recognize the explicit simulation-only recovery policy.

    Args:
        setup: The settings captured before the simulation controllers were prepared.

    Returns:
        Whether this Part slippage run explicitly defers formal safety checks.
    """
    return (
        setup.get("execution_mode") == "simulation"
        and setup.get("diagnostic_cca_bypass") is True
        and setup.get("runtime_recovery_validation_policy") == "no_validation"
        and setup.get("selected_safety_file") == "__NONE__"
        and (setup.get("failure_scenario") or {}).get("scenario") in FAILURE_SCENARIOS
    )


class PartSlippageRecovery:
    """Retain generation, operator review and execution on the ProductAgent loop."""

    def __init__(self, runtime: Any, *, observations: dict | None = None) -> None:
        """Attach to an explicitly prepared diagnostic simulation runtime."""
        if not runtime.diagnostic_cca_bypass or not diagnostic_setup(runtime.setup):
            raise ValueError("Part slippage recovery requires the diagnostic simulation policy")
        self.runtime = runtime
        self.agent = runtime.agent
        reports = getattr(runtime, "reports", None)
        directory = reports.directory if reports is not None else Path(runtime.path)
        self.path = directory / "recovery" / runtime.context.run_id
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._task: asyncio.Task | None = None
        self._generation = 0
        self._stopped = False
        self._stage_stop: str | None = None
        self._offline_cancel = threading.Event()
        self._generate_on_begin: bool | None = None
        self._defer_observation_save = False
        self._failure: dict = {}
        self._prepared: dict = {}
        self._outline_request: dict = {}
        self._proposal: dict = {}
        self.offline = observations is not None
        self._observations = deepcopy(observations or {})
        self._context_revision: int | None = None
        self._state = {
            "run_id": runtime.context.run_id,
            "source": "offline" if self.offline else "gazebo",
            "scenario": runtime.setup["failure_scenario"]["scenario"],
            "observation_path": "", "observations": {},
            "selected_safety_files": deepcopy(runtime.setup.get("selected_safety_files", [])),
            "status": "running_example",
            "message": f"Running the example to the observed {runtime.setup['failure_scenario']['scenario']} checkpoint.",
            "outline_only": False,
            "operator_assistance": [],
            "attempt_provenance": {}, "provenance_path": "",
            "outline_mode": "manual" if self.offline else runtime.setup.get("runtime_recovery_mode", "auto"),
            "primitive_mode": "manual" if self.offline else runtime.setup.get("runtime_recovery_primitive_mode", "auto"),
            "outline_status": "waiting",
            "primitive_status": "waiting",
            "safety_status": "skipped",
            "diagnostic_cca_bypass": True,
            "validation_policy": "no_validation",
            "artifact_directory": "",
            "outline_artifact_directory": "", "primitive_artifact_directory": "",
            "outline": [], "primitives": [], "execution": [], "revision": 0,
            "imported_recovery": False,
        }

    def snapshot(self) -> dict:
        """Return detached progress without starting or advancing any work."""
        with self._lock:
            state = deepcopy(self._state)
            if self._outline_request:
                from .primitive_composition import composition_readiness

                state["composition_readiness"] = composition_readiness(self._outline_request)
            return state


    def _publish(self, **values: Any) -> None:
        with self._lock:
            self._state.update(deepcopy(values))
            self._state["revision"] += 1
            write_record(self.path / "run.json", self._state)
        self.runtime.queue_save()

    def _schedule(self, operation) -> None:
        if self._stopped:
            raise ValueError("Reset example before starting another recovery")
        if self._task is not None and not self._task.done():
            raise ValueError("A recovery action is already running")
        async def run() -> None:
            try:
                await operation()
            except asyncio.CancelledError:
                # Offline workers have drained here, so the stopped stage can publish.
                if self._stage_stop is not None and not self._stopped:
                    self._finish_stage_stop()
                raise
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError, OpenAIError) as exc:
                if self._defer_observation_save and isinstance(exc, ValueError) and str(exc).startswith("NEEDS_CONTEXT:"):
                    if not self._stopped:
                        self._publish(status="NEEDS_CONTEXT", outline_status="NEEDS_CONTEXT",
                                      outline_safety_status="unresolved", execution_authorized=False, message=str(exc))
                        logger.info("Offline recovery needs recorded context: %s", exc)
                    return
                logger.exception("Part slippage recovery could not complete")
                if not self._stopped:
                    self._record_error(f"{type(exc).__name__}: {exc}")
            finally:
                # A stage stop must not cancel the next operation's previews or workers.
                if not self._stopped:
                    self._stage_stop = None
                    if self._offline_cancel.is_set():
                        self._offline_cancel = threading.Event()

        self._task = asyncio.create_task(run(), name="part_slippage_recovery")
        self._task.add_done_callback(self._task_finished)

    def _task_finished(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("Recovery task failed", exc_info=(type(error), error, error.__traceback__))
            if not self._stopped:
                self._record_error(f"{type(error).__name__}: {error}")

    def _record_error(self, message: str) -> None:
        updates = {f"{stage}_status": "failed" for stage in ("outline", "primitive")
                   if self._state[f"{stage}_status"] in {"generating", "preparing"}}
        if self._state["status"] == "preparing_outline":
            updates["outline_status"] = "failed"
        if self._state.get("validation_contract_version") == 2 and updates.get("outline_status") == "failed":
            updates["outline_safety_status"] = "unresolved"
        directory = self._state.get("artifact_directory")
        if directory:
            write_record(Path(directory) / "stage_error.json", {
                "status": self._state["status"], "error": message, "safety_status": "skipped", **updates,
            })
        self._publish(status="blocked", message=message, **updates)

    def _save_observations(self) -> None:
        path = self.path / "observations.json"
        write_capture(path, self._observations)
        self._publish(observation_path=str(path), observations=observation_summary(self._observations))

    def begin(self, failure: dict, *, generate_outline: bool | None = None,
              defer_observation_save: bool = False) -> None:
        """Prepare once; optionally defer generation until an explicit action.

        Args:
            failure: Exact recorded reconciled failure.
            generate_outline: Whether this operator action includes outline generation.
            defer_observation_save: Save a detached Offline capture in the worker before preparation.
        """
        if defer_observation_save and not self.offline:
            raise ValueError("Deferred observation saving requires Offline recovery")
        if self._failure or self._stopped:
            return
        self._failure = deepcopy(failure)
        self._generate_on_begin = generate_outline
        self._defer_observation_save = defer_observation_save
        if not self.offline:
            checkpoint = getattr(self.runtime, "recovery_checkpoint_observations", None)
            self._observations = (deepcopy(checkpoint) if checkpoint is not None else
                                  capture_observations(self.runtime, failure, include_physical=False))
            if checkpoint is not None:
                self._observations["failure"] = deepcopy(failure)
                self._observations["processing_shutdown"] = deepcopy(failure.get("processing_shutdown", {}))
        if not defer_observation_save:
            self._save_observations()
        self._publish(status="preparing_outline", message=f"Preparing recovery from the observed {self._state['scenario']}.")
        self._schedule(self._begin)

    async def _begin(self) -> None:
        if self._defer_observation_save:
            await self._offline_work(self._save_observations)
        await self.runtime.nominal_idle.wait()
        # The same gate as offline grounding: only failures that record the agents' state are grounded.
        grounds_beliefs = (not self.offline and self._observations.get("schema_version") == RAW_SCHEMA_VERSION
                           and self._failure.get("injection_status") == "completed"
                           and has_raw_failure_facts({"failure": self._failure}))
        if not self.offline and self._observations.get("physical_evidence", {}).get("status") == "pending":
            # Persist reconciled facts before a slow or failing ROS observation;
            # its owner agents must remain able to service their event loop.
            self._observations = capture_observations(self.runtime, self._failure, include_physical=False)
            self._save_observations()
            physical = await self._capture_physical_evidence()
            if grounds_beliefs:
                physical = await self._ground_live_capture(physical)
            settings = self.runtime.context.inputs.get("scene", {}).get("safety_preparation", {}).get("outline_preview")
            if isinstance(settings, dict):
                from cais_spade_llm.recovery_framework.recovery_motion_preview import capture_preview_scene

                try:
                    # MoveIt is captured separately. Its asynchronous state is
                    # never labelled as the same physics sample merely because
                    # both read-only requests completed in this attempt.
                    scene_capture = await capture_preview_scene(execution_mode=self.runtime.setup["execution_mode"],
                        planning_scene_service=settings["planning_scene_service"],
                        preview_parameter_service=settings["preview_parameter_service"],
                        timeout_seconds=settings["timeout_seconds"])
                    self._observations["planning_scene_capture"] = {
                        **scene_capture, "status": "captured", "compatibility": "not_checked",
                        "physical_evidence_fingerprint": physical.get("evidence_fingerprint"),
                    }
                except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
                    self._observations["planning_scene_capture"] = {
                        "status": "NEEDS_CONTEXT", "reason": str(exc), "compatibility": "not_checked",
                        "physical_evidence_fingerprint": physical.get("evidence_fingerprint"),
                        "source": "read-only GetPlanningScene and preview model parameters",
                    }
                if self._stopped:
                    raise asyncio.CancelledError()
                self._save_observations()
        # A frozen checkpoint cannot be retaken, so its beliefs keep their acknowledged
        # values; offline grounding applies the grounded fault before the first LLM turn.
        if self._failure.get("injection_status") != "completed":
            raise ValueError("Failure observations require reconciliation before recovery: " +
                             str(self._failure.get("error") or self._failure.get("injection_status")))
        compatibility_missing = []
        if self._observations.get("processing_observations"):
            from cais_spade_llm.recovery_framework.recovery_observations import processing_capture_compatibility

            compatibility = processing_capture_compatibility(self._observations)
            compatibility_missing = list(dict.fromkeys(row["reason"] for row in compatibility["checks"]
                                                       if row["status"] != "supported"))
            if compatibility_missing and not self.offline:
                raise ValueError("NEEDS_CONTEXT: " + "; ".join(compatibility_missing))
        await self._prepare()
        if self._stopped:
            raise asyncio.CancelledError()
        message = (f"{self._state['scenario']} observed and saved. Generate an unvalidated outline from the saved "
                   "observations in Offline.")
        if compatibility_missing:
            message += " Processing capture compatibility: " + "; ".join(compatibility_missing)
        self._publish(status="outline_ready", outline_status="ready", message=message)

    def _new_artifact_directory(self) -> Path:
        self._generation += 1
        directory = self.path / f"generation_{self._generation:04d}"
        directory.mkdir(parents=True, exist_ok=False)
        self._publish(artifact_directory=str(directory), composition_validation_version=None,
                      composition_feasibility="not_checked", composition_findings=[], execution_authorized=False)
        return directory

    async def _prepare(self, feedback: str = "") -> None:
        from cais_spade_llm.agents.intelligent_product.replanner.failure_context import (
            build_failure_event,
        )
        from cais_spade_llm.resources.robot.robot_profile import _robot_snapshot_builder
        from cais_spade_llm.resources.gazebo_programs import saved_robot_definition
        from cais_spade_llm.resources.robot.robot_tasks import robot_task_registry
        from cais_spade_llm.recovery_framework.recovery_observations import captured_resource_snapshots

        runtime, context = self.runtime, self.runtime.context
        directory = self._new_artifact_directory()
        self._publish(outline_artifact_directory=str(directory / "recovery_outline"),
                      primitive_artifact_directory=str(directory / "recovery_primitves"))
        saved = self._observations.get("prepared_recovery_request")
        if saved and (self.offline or self._observations.get("offline_ready")):
            # The capture already holds its prepared request; drafts read it from the saved observations.
            return
        if self.offline:
            raise ValueError("NEEDS_CONTEXT: saved capture has no prepared_recovery_request; see missing observations")
        configuration = runtime.conveyor_fault.configuration
        rid = configuration["resource_id"]
        completed = _completed_tasks(self._observations)
        failed = next((task for task in self._failure.get("pending_tasks", [])
                       if task["resource_id"] == rid and task.get("task_id") not in completed), {})
        part = self._failure.get("part_name", configuration.get("part_name"))
        if part is None and "part_name" not in self._failure and "part_name" not in configuration:
            part = failed.get("parameters", {}).get("part_name")
        if not isinstance(part, str) or part not in context.part_tracker:
            raise ValueError("NEEDS_CONTEXT: failure must identify its recorded affected part")
        resource_states, recorded_snapshots = {}, captured_resource_snapshots(self._observations)
        robot_ids = {row["resource_id"] for row in context.inputs["scene"]["robots"]}
        for resource in runtime.resource_agents:
            values = self._observations["resource_states"][resource.agent_name]
            if resource.agent_name in robot_ids:
                physical = self._observations.get("physical_evidence", {}).get("snapshot", {}).get("resources", {}).get(resource.agent_name, {})
                if "held_part" in physical and "held_part" in values and physical["held_part"] != values["held_part"]:
                    raise ValueError(f"NEEDS_CONTEXT: recorded physical and symbolic held_part conflict for {resource.agent_name}")
                pose = physical.get("current_pose")
                if isinstance(pose, list) and len(pose) == 7:
                    pose = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), pose, strict=True))
                snapshot = _robot_snapshot_builder(resource, recorded_pose=pose if isinstance(pose, dict) else {})
                if "held_part" in physical and snapshot.get("held_part") != physical["held_part"]:
                    raise ValueError(f"NEEDS_CONTEXT: recorded physical and owner held_part conflict for {resource.agent_name}")
                recorded_snapshots[str(resource.jid)] = snapshot
                if isinstance(pose, dict):
                    self._observations.setdefault("robot_poses", {})[resource.agent_name] = deepcopy(pose)
            else:
                snapshot = recorded_snapshots.get(str(resource.jid))
                if not isinstance(snapshot, dict):
                    raise ValueError(f"NEEDS_CONTEXT: {resource.agent_name} captured owner snapshot is unavailable")
                recorded_snapshots[str(resource.jid)] = deepcopy(snapshot)
            if hasattr(context, "unavailable_resources"):
                if resource.agent_name in context.unavailable_resources:
                    recorded_snapshots[str(resource.jid)]["availability"] = "unavailable"
                else:
                    recorded_snapshots[str(resource.jid)].setdefault("availability", "available")
            resource_states[str(resource.jid)] = deepcopy(values)
            if hasattr(context, "unavailable_resources"):
                resource_states[str(resource.jid)]["availability"] = recorded_snapshots[str(resource.jid)]["availability"]
            if hasattr(resource, "_current_state"):
                resource_states[str(resource.jid)]["current_state"] = deepcopy(resource._current_state)
            elif "resource_state" in values:
                resource_states[str(resource.jid)]["current_state"] = deepcopy(values["resource_state"])
            if "resource_location" in values:
                resource_states[str(resource.jid)]["current_location"] = deepcopy(values["resource_location"])
        tracker = deepcopy(context.part_tracker)
        for name, row in tracker.items():
            if name in context.initial_product_states:
                row.setdefault("origin_resource_location", context.initial_product_states[name].get("location"))
        self.agent.part_tracker = deepcopy(tracker)
        self.agent.process_planner.nodes = self._pending_nodes()
        self._observations["pending_nodes"] = deepcopy(self.agent.process_planner.nodes)
        self._save_observations()
        tools_catalog = []
        for resource in runtime.resource_agents:
            if resource.agent_name not in robot_ids:
                model_getter = getattr(resource, "recovery_des_model", None)
                if callable(model_getter):
                    model = model_getter(snapshot=recorded_snapshots[str(resource.jid)])
                    for event in model.get("events", []):
                        tools_catalog.append({"function": event["event_name"],
                            "function_owner_agent": str(resource.jid),
                            "parameter_bindings": deepcopy(event.get("parameter_bindings", {}))})
                continue
            for function in resource.executables:
                if function in robot_task_registry():
                    definition = saved_robot_definition(context.inputs["scene"], resource.agent_name, function)
                    tools_catalog.append({"function": function, "function_owner_agent": str(resource.jid),
                                          **definition.tool_frontmatter()})
        failure_event = build_failure_event(
            failed_task_id=failed.get("task_id", ""), failed_resource_jid=runtime.jids[rid],
            failed_function_name=failed.get("event_name", ""),
            final_status="failed:" + self._state["scenario"],
            part_name=part, observations={**deepcopy(self._failure), "checkpoint_context": _recorded_checkpoint(self._observations)},
            affected_entities=[{"entity_type": "part", "entity_id": part,
                                "state": tracker[part]["state"], "location": tracker[part]["location"]}],
            state_before=self._failure.get("resource_values_before", {}).get(rid),
            state_after={**deepcopy(context.resources[rid].valuation),
                         "part_state": tracker[part]["state"], "part_location": tracker[part]["location"]},
        )
        states = {name: row.get("state") for name, row in tracker.items()}
        locations = {name: row.get("location") for name, row in tracker.items()}
        planner = self.agent.process_planner
        stuck = planner._build_resource_search_state(
            resource_jid=runtime.jids[rid], resource_states=resource_states,
            default_resource_state="", part_states=states, part_locations=locations,
        )
        prepared = await planner.prepare_recovery_request(
            stuck_state=stuck, P_id=list(context.selected_parts), ra_jid=runtime.jids[rid],
            goal_state="", tools_catalog=tools_catalog,
            part_tracker=tracker, obligation_targets=[], recovery_feedback=feedback,
            resource_states=resource_states, default_resource_state="", part_states=states,
            part_locations=locations, recovery_safety_context={}, failure_context=failure_event,
            recorded_resource_snapshots=recorded_snapshots,
        )
        if self._stopped:
            raise asyncio.CancelledError()
        if context.revision != self._observations["context_revision"]:
            raise ValueError("NEEDS_CONTEXT: runtime observations changed during preparation")
        for jid in recorded_snapshots:
            entry = prepared.get("recovery_resources", {}).get(jid)
            if isinstance(entry, dict) and "current_pose" in recorded_snapshots[jid]:
                entry.setdefault("snapshot_sources", {})["current_pose"] = {
                    "kind": "observed", "ref": f"resource_evidence/{jid}/snapshot/current_pose",
                    "run_id": self._observations.get("run_id"), "captured_at": self._observations.get("captured_at"),
                    "revision": self._observations.get("context_revision")}
        # Grounded once here, the request is saved with its grounding in beliefs.json.
        apply_grounding(prepared, self._observations)
        self._observations.update(
            prepared_recovery_request=deepcopy(prepared),
            llm_model=getattr(self.agent, "model", None),
            llm_reasoning_effort=getattr(self.agent, "reasoning_effort", None),
            product_jid=str(getattr(self.agent, "jid", "")),
        )
        missing = offline_preparation_missing(self._observations)
        self._observations.update(offline_ready=not missing, missing=missing)
        self._save_observations()
        if self._observations.get("planning_scene_capture", {}).get("native_model_parameters"):
            from cais_spade_llm.recovery_framework.recovery_observations import save_native_preview_model_from_capture

            await save_native_preview_model_from_capture(self.path / "observations.json",
                                                        cancel_event=self._offline_cancel)
        if missing:
            raise ValueError("NEEDS_CONTEXT: " + ", ".join(missing))


    async def _capture_physical_evidence(self) -> dict:
        from cais_spade_llm.recovery_framework.recovery_observations import (
            capture_recovery_physical_evidence, capture_resource_evidence,
        )

        revision = self.runtime.context.revision
        capture = asyncio.create_task(asyncio.to_thread(capture_recovery_physical_evidence, self.runtime))
        try:
            physical = await asyncio.shield(capture)
        except asyncio.CancelledError:
            # Observation interfaces cannot be cancelled as robot commands.
            # Drain the owned worker and discard its result after Stop.
            await asyncio.gather(capture, return_exceptions=True)
            raise
        if self._stopped:
            raise asyncio.CancelledError()
        if self.runtime.context.revision != revision:
            raise ValueError("NEEDS_CONTEXT: runtime observations changed during physical capture; retry preparation")
        self._observations["physical_evidence"] = physical
        self._observations["resource_evidence"] = capture_resource_evidence(self.runtime, physical)
        self._save_observations()
        return physical

    async def _ground_live_capture(self, physical: dict) -> dict:
        """Update the product's beliefs from the grounded fault until a retaken capture agrees.

        Failure injection leaves the beliefs at their acknowledged values, so
        execution monitoring applies the grounded fault state before preparation.
        A first capture can lack an observation that a retaken one supplies, so
        each grounding replaces the previous belief changes until two consecutive
        captures ground the same changes. The original beliefs are restored if
        they do not settle or the grounded fault itself changes.

        Args:
            physical: Physical evidence captured before grounding.

        Returns:
            Physical evidence captured with the settled beliefs.

        Raises:
            ValueError: No failure is detectable, or the grounding does not settle.
        """
        context = self.runtime.context
        grounded = self._grounded_fault()
        passes = [self._grounding_pass(grounded)]
        if not grounded["belief_updates"]:
            return physical
        rows, applied = None, None
        try:
            for _ in range(_GROUNDING_RETAKES):
                if rows is not None:
                    restore_beliefs(context, self._observations, rows)
                rows = belief_rows(context, self._observations, grounded, saved=rows)
                update_beliefs(context, self._observations, grounded)
                applied = grounded
                self._observations["context_revision"] = context.revision
                self._save_observations()
                # Belief-dependent views were derived before this grounding; retake them.
                physical = await self._capture_physical_evidence()
                grounded = ground_failure(self._observations)
                passes.append(self._grounding_pass(grounded))
                write_record(self.path / "grounding_passes.json", {"passes": passes})
                if fault_identity(grounded)[:2] != fault_identity(applied)[:2]:
                    raise ValueError("NEEDS_CONTEXT: the retaken physical capture grounds a different fault")
                if belief_key(grounded) == belief_key(applied):
                    return physical
            raise ValueError("NEEDS_CONTEXT: grounded beliefs did not settle across retaken physical captures")
        except (ValueError, asyncio.CancelledError):
            if rows is not None:
                restore_beliefs(context, self._observations, rows)
                self._save_observations()
            raise

    def _grounding_pass(self, grounded: dict) -> dict:
        physical = self._observations.get("physical_evidence") or {}
        event = grounded["fault_event"]
        return {"context_revision": self._observations.get("context_revision"),
                "unresolved": deepcopy(physical.get("unresolved", [])),
                "signals": [row["rule"] + ":" + str(row.get("resource_id") or row.get("part_name"))
                            for row in event["signals"]],
                "faulted_resources": deepcopy(event["faulted_resources"]), "failed_task_id": event["failed_task_id"],
                "belief_updates": deepcopy(grounded["belief_updates"])}

    def _grounded_fault(self) -> dict:
        grounded = ground_failure(self._observations)
        if not grounded["fault_event"]["detected"]:
            raise ValueError("NEEDS_CONTEXT: no failure is detectable from the recorded execution-monitoring facts")
        return grounded

    def _pending_nodes(self) -> list[dict]:
        return pending_nodes(self._failure, self._observations, run_id=self.runtime.context.run_id,
                             jids=self.runtime.jids)

    async def _run_stage(self, stage: str) -> dict:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import (
            execute_multi_turn_recovery,
        )

        generation = self._generation
        self._prepared["_stop_after_multi_turn_phase"] = stage
        if self.offline and stage == "primitive":
            self.agent._composition_cancel_event = self._offline_cancel
        provider_turns = 0
        while True:
            if self._stopped or generation != self._generation:
                raise asyncio.CancelledError()
            prepared = self._prepared
            if provider_turns >= 80:
                raise ValueError(f"{stage} exceeded its generation budget")
            provider_turns += 1
            if self.offline:
                def generate() -> dict:
                    # The provider turn and its CPU validation own this copy.
                    # Stop may retain artifacts, but cannot publish its state.
                    working = deepcopy(self._prepared)
                    asyncio.run(execute_multi_turn_recovery(self.agent.process_planner, working,
                        session_state=working.get("multi_turn_session_state")))
                    return working

                prepared = await self._offline_work(generate)
                self._prepared = prepared
            else:
                await execute_multi_turn_recovery(self.agent.process_planner, prepared,
                    session_state=prepared.get("multi_turn_session_state"))
            if self._stopped or generation != self._generation:
                raise asyncio.CancelledError()
            session = prepared.get("multi_turn_session_state", {})
            if stage == "primitive":
                findings = deepcopy(session.get("primitive_rejection_feedback", []))
                self._publish(primitives=deepcopy(session.get("accepted_primitive_program", [])),
                    composition_findings=findings,
                    composition_feasibility=(findings[0].get("feasibility_status", "NEEDS_CONTEXT") if findings else "in_progress"),
                    message=f"Validated {len(session.get('accepted_primitive_program', []))} of "
                            f"{len(session.get('accepted_outline_prefix', []))} recovery events.")
            if (stage == "outline" and session.get("current_phase") in {"primitive_generation", "finalize"}
                    and session.get("accepted_outline_prefix")):
                return session
            final = session.get("final_output", {})
            if stage == "primitive" and final.get("primitive_program_complete") is True:
                return session
            if session.get("status") not in {
                "paused_after_outline_turn", "ready_for_primitive_generation", "paused_after_primitive_turn",
                "paused_after_primitive_generation",
            }:
                reasons = "; ".join(row.get("reason", "") for row in session.get("primitive_rejection_feedback", [])) if stage == "primitive" else ""
                raise ValueError(f"{stage}: {reasons or session.get('status', 'No completed output')}")

    async def _offline_work(self, operation):
        """Drain detached work before releasing ownership or publishing results."""
        generation = self._generation
        if self._stopped:
            raise asyncio.CancelledError()
        worker = asyncio.create_task(asyncio.to_thread(operation))
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            self._offline_cancel.set()
            # Native preview cancellation checks this event. Pure calculations
            # finish privately; another attempt cannot share their owners yet.
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except (OSError, ValueError, KeyError, TypeError, RuntimeError, OpenAIError):
                    break
            if not worker.cancelled():
                worker.exception()
            raise
        if self._stopped or generation != self._generation:
            raise asyncio.CancelledError()
        return result


    async def _continue_outline(self) -> None:
        from .primitive_composition import start_composition_attempt

        await start_composition_attempt(self)
        self._publish(status="primitive_ready", primitive_status="ready",
                      message="Primitive composition is ready to generate.")
        if not self.offline and self._state["primitive_mode"] == "auto":
            await self._generate_primitives()

    async def _generate_primitives(self) -> None:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import (
            build_multi_turn_recovery_proposal,
        )

        self._publish(status="generating_primitives", primitive_status="generating",
                      message="Generating primitive composition.")
        session = await self._run_stage("primitive")
        adapted = build_multi_turn_recovery_proposal(
            final_output_payload=session["final_output"], prepared_recovery_request=self._prepared,
        )
        if not adapted.get("accepted"):
            raise ValueError(adapted.get("reason") or "Primitive composition is incomplete")
        self._proposal = adapted["recovery_proposal"]
        self._proposal["offline_recovery"] = self.offline
        self._proposal["validation_contract_version"] = self._prepared.get("validation_contract_version")
        if self._prepared.get("composition_validation_version") == 1:
            self._proposal["composition_validation_version"] = 1
        if "primitive_support_semantics" in self._prepared:
            self._proposal["primitive_support_semantics"] = self._prepared["primitive_support_semantics"]
        self._proposal["execution_authorized"] = False
        self._proposal["source_observation_fingerprint"] = observation_fingerprint(self._observations)
        write_record(Path(self._state["artifact_directory"]) / "proposal.json", self._proposal)
        write_record(Path(self._state["artifact_directory"]) / "composition.json", {
            "composition_validation_version": self._prepared.get("composition_validation_version"),
            "source_outline_fingerprint": self._prepared.get("composition_source_fingerprint"),
            "accepted_outline_prefix": self._outline_request["multi_turn_session_state"]["accepted_outline_prefix"],
            "accepted_primitive_program": session.get("accepted_primitive_program", []),
            "composition_feasibility": "FEASIBLE", "execution_authorized": False,
            "cca_composition_status": "not_checked"})
        self._publish(status="offline_completed" if self.offline else "ready_to_execute", primitive_status="completed",
                      primitives=session.get("accepted_primitive_program", []),
                      composition_validation_version=self._prepared.get("composition_validation_version"),
                      composition_feasibility="FEASIBLE", execution_authorized=False,
                      message=("Composition completed for review. Execution requires the subsequent runtime CCA integration."
                               if self._prepared.get("validation_contract_version") == 2
                               or self._prepared.get("composition_validation_version") == 1 else
                               "Offline composition completed. Test in Gazebo before execution." if self.offline
                               else "Primitive composition completed. Review it, then Execute recovery."))

    def action(self, action: str, *, feedback: str = "") -> dict:
        """Schedule one explicit operator action and return immediately."""
        status = self._state["status"]
        if feedback:
            self._publish(operator_assistance=[*self._state["operator_assistance"],
                {"action": action, "feedback": feedback, "generation": self._generation}])
        if action == "execute" and (self._state.get("validation_contract_version") == 2
                                    or self._state.get("composition_validation_version") == 1):
            raise ValueError("Outline validation does not authorize execution; runtime CCA integration is pending")
        composition_stopped = self._state["primitive_status"] == "stopped"
        operations = {
            "prepare": ({"blocked"}, self._begin),
            "continue_outline": ({"outline_review"}, self._continue_outline),
            "generate_primitives": ({"primitive_ready", "outline_review"} if self.offline or composition_stopped
                                    else {"primitive_ready"}, self._generate_primitives_from_review),
            "execute": ({"ready_to_execute"}, self._execute),
        }
        if action == "regenerate_primitives":
            if status not in {"outline_review", "primitive_ready", "ready_to_execute", "offline_completed", "blocked"}:
                raise ValueError("Wait for the current stage before regenerating")

            async def regenerate() -> None:
                if not self._outline_request:
                    raise ValueError("Generate the recovery outline first")
                from .primitive_composition import start_composition_attempt

                await start_composition_attempt(self, feedback=feedback)
                await self._generate_primitives()

            self._schedule(regenerate)
        else:
            if action not in operations or status not in operations[action][0]:
                raise ValueError(f"{action} is unavailable while {status}")
            self._schedule(operations[action][1])
        return self.snapshot()


    async def _generate_primitives_from_review(self) -> None:
        if self._state["status"] == "outline_review":
            await self._continue_outline()
        # Gazebo Auto already composed inside Continue.
        if self._state["status"] == "primitive_ready":
            await self._generate_primitives()

    async def _execute(self) -> None:
        from cais_spade_llm.recovery_framework.slippage_execution import execute_generated_recovery

        if (self._state.get("validation_contract_version") == 2 or self._proposal.get("validation_contract_version") == 2
                or self._state.get("composition_validation_version") == 1
                or self._proposal.get("composition_validation_version") == 1):
            raise ValueError("Outline validation does not authorize execution; runtime CCA integration is pending")
        if self.offline or self._proposal.get("offline_recovery"):
            raise ValueError("Offline proposals require Test in Gazebo before execution")
        if self.runtime.context.revision != self._context_revision:
            raise ValueError("Observed execution context changed; regenerate the recovery outline")
        if getattr(self, "_imported_proposal", None) is not None:
            from cais_spade_llm.recovery_framework.saved_recovery_execution import (
                validate_saved_proposal,
            )

            await validate_saved_proposal(self, self._imported_proposal, self._imported_observations)
        self._publish(status="executing", message="Executing recovery primitives.")
        await execute_generated_recovery(self)
        if self._stopped:
            raise asyncio.CancelledError()
        runtime = self.runtime
        with runtime.context.admission_lock:
            runtime.stopped = False
            runtime._calculation_stop.clear()
            runtime.retained_paths = {}
            runtime.context.environment_model["closed"] = False
            runtime.context.revision += 1
            runtime.outcome = {"status": "prepared", "reason": "Recovery completed; resuming assembly"}
        self._publish(status="resuming", message="Recovery completed. Resuming unfinished assembly.")

    def nominal_finished(self) -> None:
        """Record the observed nominal outcome after recovery releases the scheduler."""
        if self._state["status"] != "resuming":
            return
        outcome = self.runtime.outcome
        if outcome.get("status") == "completed":
            self._publish(status="completed", message="Recovery and remaining assembly completed.",
                          outcome=outcome)
        elif outcome.get("status") in {"blocked", "execution_unavailable", "held"}:
            self._publish(status="blocked", message=outcome.get("reason", "Assembly could not resume"),
                          outcome=outcome)

    def generating_stage(self) -> str:
        """Name the generation stage that ``stop_generation`` can cancel.

        Returns:
            ``"outline"``, ``"primitive"``, or ``""`` when no stage can be stopped.
        """
        if self._stopped or self._stage_stop is not None or self._task is None or self._task.done():
            return ""
        if self._state["primitive_status"] in {"preparing", "generating"}:
            return "primitive"
        if self._state["outline_status"] == "generating":
            return "outline"
        return ""

    def stop_generation(self) -> dict:
        """Cancel the running outline or composition stage and keep this run usable.

        Offline workers finish their in-flight provider call first, and its
        response is retained. A stopped composition keeps the accepted outline.

        Returns:
            The detached run snapshot.

        Raises:
            ValueError: No outline or composition generation is running.
        """
        if self._stage_stop is not None:
            return self.snapshot()
        stage = self.generating_stage()
        if not stage:
            raise ValueError("No recovery generation is running")
        self._stage_stop = stage
        self._offline_cancel.set()
        self._generation += 1
        self._task.get_loop().call_soon_threadsafe(self._task.cancel)
        self._publish(message="Stopping after the current LLM call; captured responses are retained.")
        return self.snapshot()

    def _finish_stage_stop(self) -> None:
        if self._stage_stop == "outline":
            self._publish(status="outline_ready", outline_status="stopped", outline=[], primitive_status="waiting",
                          message="Recovery outline stopped. Generate starts a fresh outline attempt.")
        else:
            self._publish(status="outline_review", primitive_status="stopped", primitives=[], composition_findings=[],
                          composition_feasibility="not_checked", execution_authorized=False,
                          message="Primitive composition stopped. The accepted outline is kept; "
                                  "Generate starts a new composition attempt.")

    def stop(self, reason: str = "Stopped by operator") -> None:
        """Invalidate pending generation and cancel this run's owned operation."""
        self._stopped = True
        self._offline_cancel.set()
        self._generation += 1
        if self._task is not None and not self._task.done():
            self._task.get_loop().call_soon_threadsafe(self._task.cancel)
        if self._state["status"] not in {"completed", "offline_completed"}:
            self._publish(status="stopped", message=reason)
