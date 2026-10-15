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

from cais_spade_llm.recovery_framework import ROOT, fingerprint
from cais_spade_llm.recovery_framework.capture_files import (
    RAW_SCHEMA_VERSION,
    pending_nodes,
    write_capture,
)
from cais_spade_llm.recovery_framework.capture_files import completed_tasks as _completed_tasks
from cais_spade_llm.recovery_framework.diagnostics import write_record
from cais_spade_llm.recovery_framework.failure_grounding import (
    apply_grounding,
    fault_identity,
    ground_failure,
    has_raw_failure_facts,
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
            session = self._prepared.get("multi_turn_session_state", {})
            state["can_prepare_gazebo_evidence"] = bool(
                not self._stopped and state.get("validation_contract_version") == 2
                and session.get("pending_outline_validation") and state["status"] == "blocked")
            if self._outline_request:
                from .primitive_composition import composition_readiness

                state["composition_readiness"] = composition_readiness(self._outline_request)
            return state

    async def retry_outline_validation(self, evidence: dict) -> None:
        """Retry a saved proposal using new evidence, without another LLM call."""
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import (
            retry_pending_event,
        )
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
            archive_nominal_continuation, write_recovery_validation_record,
        )

        if self._stopped:
            raise asyncio.CancelledError()
        session = self._prepared.get("multi_turn_session_state", {})
        if not session.get("pending_outline_validation"):
            raise ValueError("There is no pending outline proposal to validate")
        generation = self._generation
        working = deepcopy(session)
        supplement = evidence.get("resource_evidence_capture")
        if supplement is not None:
            from cais_spade_llm.recovery_framework.outline_gazebo_evidence import checkpoint_differences
            from cais_spade_llm.recovery_framework.recovery_observations import observation_fingerprint

            if evidence.get("source_observation_fingerprint") != observation_fingerprint(self._observations):
                raise ValueError("Supplementary validation evidence belongs to another observation capture")
            differences = checkpoint_differences(self._observations, supplement)
            if differences:
                raise ValueError("Supplementary validation evidence changes the checkpoint: " + "; ".join(differences))
            owners = getattr(self.agent, "resources", {})
            for jid, facts in supplement.get("resource_evidence", {}).items():
                if jid not in self._prepared.get("recovery_resources", {}):
                    raise ValueError("Supplementary validation evidence identifies an unknown resource")
                owner = owners.get(jid) if isinstance(owners, dict) else None
                install = getattr(owner, "install_recovery_resource_evidence", None)
                verified = (install(observations=supplement, source_fingerprint=evidence["source_observation_fingerprint"])
                            if callable(install) else deepcopy(facts))
                working.setdefault("validation_resource_context", {}).setdefault(jid, {})["resource_evidence"] = verified
            capture_record = {"source_observation_fingerprint": evidence["source_observation_fingerprint"],
                              "observations": deepcopy(supplement)}
            captures = working.setdefault("validation_resource_evidence_captures", [])
            if capture_record not in captures:
                captures.append(capture_record)
        if "projected_physical_evidence" not in working:
            working["projected_physical_evidence"] = deepcopy(self._prepared.get("physical_evidence") or {})
        physical = working["projected_physical_evidence"]
        preview_physical = evidence.get("physical_evidence") or {}
        template_fields = ("planning_scene_capture", "planning_scene_configuration")
        if any(field in preview_physical for field in template_fields):
            from cais_spade_llm.recovery_framework.outline_gazebo_evidence import _saved_planning_scene

            preview_observations = self._offline_preview_observations() if self.offline else self._observations
            captured = _saved_planning_scene(preview_observations)
            configuration = preview_observations.get("inputs", {}).get("scene", {})
            expected = {"planning_scene_capture": captured, "planning_scene_configuration": {
                key: deepcopy(configuration[key]) for key in ("robots", "planning_scene_mounting_contacts") if key in configuration}}
            if captured is None or any(field not in preview_physical or preview_physical[field] != expected[field]
                                       for field in template_fields):
                raise ValueError("Preview planning scene template differs from the immutable observations")
            pending_scene = working["pending_outline_validation"].get("projected_physical_evidence")
            for target in (physical, pending_scene):
                if target is None:
                    continue
                for field in template_fields:
                    if field in target and target[field] != expected[field]:
                        raise ValueError("Preview changes an existing planning scene template")
                    target[field] = deepcopy(expected[field])
        for field in ("scene_object_bindings", "support_collision_object_bindings"):
            value = (evidence.get("physical_evidence") or {}).get(field)
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ValueError(f"Preview {field} must contain recorded mappings")
            for key, binding in value.items():
                recorded = physical.setdefault(field, {})
                if key in recorded and recorded[key] != binding:
                    raise ValueError(f"Preview changes an existing {field} mapping")
                recorded[key] = deepcopy(binding)
        preview_models = (evidence.get("physical_evidence") or {}).get("resource_models", evidence.get("resource_models", {}))
        for rid, model in preview_models.items():
            current = physical.setdefault("resource_models", {}).setdefault(rid, {})
            for field in ("event_projection_contracts", "named_joint_poses"):
                for key, value in model.get(field, {}).items():
                    recorded = current.setdefault(field, {})
                    if key in recorded and recorded[key] != value:
                        raise ValueError("Preview changes an existing owner model declaration")
                    recorded[key] = deepcopy(value)
        for record in evidence.get("event_evidence", []):
            if record not in physical.setdefault("event_evidence", []):
                physical["event_evidence"].append(deepcopy(record))
        if evidence.get("physical_capability_rejected") is True:
            from cais_spade_llm.recovery_framework.outline_gazebo_evidence import pending_outline_event
            from cais_spade_llm.recovery_framework.outline_validation import rejected_outline_capability_projection

            task = pending_outline_event(self)
            entry = self._prepared["recovery_resources"][task["resource_jid"]]
            pending = working["pending_outline_validation"]
            snapshot = pending.get("recovery_snapshot", working.get("derived_resource_snapshots", {}).get(
                task["resource_jid"], entry["recovery_snapshot"]))
            scene = pending.get("projected_physical_evidence", physical)
            owners = getattr(self.agent, "resources", {})
            owner = owners.get(task["resource_jid"]) if isinstance(owners, dict) else None
            if owner is None:
                owner = next((resource for resource in getattr(self.runtime, "resource_agents", [])
                              if str(resource.jid) == task["resource_jid"]), None)
            if owner is None:
                raise ValueError("Rejected capability requires its owning resource evidence")
            record = {key: deepcopy(evidence[key]) for key in ("status", "validation_status", "physical_capability_rejected",
                "capability_result", "event_record_binding", "resource_evidence_fingerprint", "static_capabilities_fingerprint",
                "motion_requests", "task", "reason") if key in evidence}
            rejected_outline_capability_projection(record=record, task=task, recovery_snapshot=snapshot,
                recovery_des_model=entry["recovery_des_model"], physical_evidence=scene,
                resource_evidence=owner.get_recovery_resource_evidence(), static_capabilities=owner.static_capabilities)
            if record not in physical.setdefault("rejected_capability_evidence", []):
                physical["rejected_capability_evidence"].append(record)
        if evidence.get("physical_motion_unresolved") is True:
            from cais_spade_llm.recovery_framework.outline_gazebo_evidence import pending_outline_event
            from cais_spade_llm.recovery_framework.outline_validation import incomplete_outline_motion_projection

            task = pending_outline_event(self)
            entry = self._prepared["recovery_resources"][task["resource_jid"]]
            pending = working["pending_outline_validation"]
            snapshot = pending.get("recovery_snapshot", working.get("derived_resource_snapshots", {}).get(
                task["resource_jid"], entry["recovery_snapshot"]))
            scene = pending.get("projected_physical_evidence", physical)
            owners = getattr(self.agent, "resources", {})
            owner = owners.get(task["resource_jid"]) if isinstance(owners, dict) else None
            if owner is None:
                owner = next((resource for resource in getattr(self.runtime, "resource_agents", [])
                              if str(resource.jid) == task["resource_jid"]), None)
            if owner is None:
                raise ValueError("Incomplete motion requires its owning resource evidence")
            record = {key: deepcopy(evidence[key]) for key in ("status", "validation_status", "physical_motion_unresolved",
                "incomplete_motion", "event_record_binding", "resource_evidence_fingerprint", "task", "reason") if key in evidence}
            incomplete_outline_motion_projection(record=record, task=task, recovery_snapshot=snapshot,
                recovery_des_model=entry["recovery_des_model"], physical_evidence=scene,
                resource_evidence=owner.get_recovery_resource_evidence())
            if record not in physical.setdefault("incomplete_event_evidence", []):
                physical["incomplete_event_evidence"].append(record)
        if evidence.get("physical_motion_rejected") is True:
            from cais_spade_llm.recovery_framework.outline_gazebo_evidence import pending_outline_event
            from cais_spade_llm.recovery_framework.outline_validation import rejected_outline_motion_projection

            pending = working["pending_outline_validation"]
            task = pending_outline_event(self)
            entry = self._prepared["recovery_resources"][task["resource_jid"]]
            snapshot = pending.get("recovery_snapshot", working.get("derived_resource_snapshots", {}).get(
                task["resource_jid"], entry["recovery_snapshot"]))
            scene = pending.get("projected_physical_evidence", physical)
            record = {key: deepcopy(evidence[key]) for key in ("status", "validation_status", "physical_motion_rejected",
                "rejected_motion", "event_record_binding", "task", "reason", "continuous_mating_rejected",
                "mating_rejection", "mating_replay_preparation", "rejected_stage_index", "native_requests", "motion_results",
                "physical_endpoint_rejected", "endpoint_rejection", "endpoint_replay_preparation", "nominal_snap_result")
                if key in evidence}
            if (record.get("continuous_mating_rejected") is True or record.get("physical_endpoint_rejected") is True) and scene is not physical:
                scene = deepcopy(scene)
                for field in ("scene_object_bindings", "support_collision_object_bindings"):
                    for key, value in physical.get(field, {}).items():
                        target = scene.setdefault(field, {})
                        if key in target and target[key] != value:
                            raise ValueError("Preview changes a pending motion object mapping")
                        target[key] = deepcopy(value)
                for rid, model in physical.get("resource_models", {}).items():
                    for field in ("event_projection_contracts", "named_joint_poses"):
                        for key, value in model.get(field, {}).items():
                            target = scene.setdefault("resource_models", {}).setdefault(rid, {}).setdefault(field, {})
                            if key in target and target[key] != value:
                                raise ValueError("Preview changes a pending motion owner declaration")
                            target[key] = deepcopy(value)
            rejected_outline_motion_projection(record=record, task=task, recovery_snapshot=snapshot,
                recovery_des_model=entry["recovery_des_model"], physical_evidence=scene)
            if record not in physical.setdefault("rejected_event_evidence", []):
                physical["rejected_event_evidence"].append(record)
        async def validate() -> tuple[str, dict]:
            return await retry_pending_event(
                prepared=self._prepared, session=working, planner=self.agent.process_planner)

        if self.offline:
            def validate_saved() -> tuple[str, dict]:
                return asyncio.run(retry_pending_event(prepared=deepcopy(self._prepared),
                    session=working, planner=self.agent.process_planner))

            decision, turn = await self._offline_work(validate_saved)
        else:
            decision, turn = await validate()
        if self._stopped or generation != self._generation:
            raise asyncio.CancelledError()
        directory = Path(self._state["artifact_directory"])
        turns = directory / "recovery_outline" / "turns"
        turns.mkdir(parents=True, exist_ok=True)
        source_turn = turn.get("source_turn_index") or turn.get("turn_index") or session.get("turn_index", 0)
        turn.update(turn_index=source_turn, phase="outline", decision=decision,
                    validation_only=True, request_sent=False)
        def persist() -> None:
            if archive_nominal_continuation(self._prepared, working, turn, decision=decision) is None:
                write_recovery_validation_record(turns.parent, turn)
            write_record(directory / "validation_resume.json", {"multi_turn_session_state": working})

        if self.offline:
            await self._offline_work(persist)
        else:
            persist()
        self._prepared["multi_turn_session_state"] = working
        if decision == "outline_ready":
            working["current_phase"] = "primitive_generation"
            self._outline_request = deepcopy(self._prepared)
            write_record(directory / "outline_request.json", self._outline_request)
            write_record(directory / "outline.json", {"outline": working["accepted_outline_prefix"]})
            self._publish(status="outline_review", outline_status="completed", outline=working["accepted_outline_prefix"],
                outline_safety_status="passed" if self._prepared.get("outline_safety_specifications") else "not_selected",
                message="Outline validation completed. Review before continuing; execution remains disabled.")
        elif decision in {"need_next_task", "need_revision"}:
            self._publish(status="outline_ready", outline_status="ready", outline=working.get("accepted_outline_prefix", []),
                message="Validation retried. Generate continues the outline from the recorded result.")
        else:
            self._publish(status="blocked", outline_status="NEEDS_CONTEXT", outline=working.get("accepted_outline_prefix", []),
                message="NEEDS_CONTEXT: " + str(working.get("outline_validation_findings") or "Required evidence is unavailable"))

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
        elif grounds_beliefs and self._ground_beliefs()["belief_updates"]:
            raise ValueError("NEEDS_CONTEXT: grounding changed beliefs recorded in a frozen checkpoint capture; "
                             "retake the checkpoint")
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
        message = f"{self._state['scenario']} observed. Recovery outline is ready to generate."
        if compatibility_missing:
            message += " Physical validation remains NEEDS_CONTEXT: " + "; ".join(compatibility_missing)
        self._publish(status="outline_ready", outline_status="ready", message=message)
        if self._generate_on_begin is True or (not self.offline and self._generate_on_begin is None and self._state["outline_mode"] == "auto"):
            await self._generate_outline()

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
            if self.offline:
                def install():
                    missing = offline_preparation_missing(self._observations)
                    if missing:
                        raise ValueError("NEEDS_CONTEXT: " + ", ".join(missing))
                    self._install_prepared(deepcopy(saved), directory, feedback)

                await self._offline_work(install)
            else:
                self._install_prepared(deepcopy(saved), directory, feedback)
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
        self._install_prepared(prepared, directory, feedback)

    def _install_prepared(
        self, prepared: dict, directory: Path, feedback: str, *, new_attempt: bool = True,
        source_directory: Path | None = None, require_nominal_complete: bool = False,
    ) -> None:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import (
            _build_phase_prompt,
            _seed_grounded_part_pose_observations,
            build_multi_turn_session_seed,
        )

        if self._stopped:
            raise asyncio.CancelledError()
        context = self.runtime.context
        if require_nominal_complete and (new_attempt or prepared.get("validation_contract_version") != 2
                or prepared.get("nominal_continuation_artifact_version") != 1
                or not isinstance(prepared.get("saved_pending_nodes"), list)):
            raise ValueError("Saved outline requires its recorded task index and bound nominal continuation")
        if not new_attempt and prepared.get("nominal_continuation_artifact_version") == 1:
            from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
                nominal_continuation_context_binding, read_nominal_continuation,
            )

            if prepared.get("source_observation_fingerprint") != observation_fingerprint(self._observations):
                raise ValueError("Saved nominal continuation belongs to another observation capture")
            saved_session = prepared.get("multi_turn_session_state", {})
            nominal = saved_session.get("nominal_continuation_evidence", {})
            if require_nominal_complete and (not isinstance(nominal, dict) or "artifact_reference" not in nominal):
                raise ValueError("Saved outline requires its bound complete nominal continuation")
            if "artifact_reference" in nominal:
                if source_directory is None:
                    raise ValueError("Saved nominal continuation requires its verified source generation")
                proof = read_nominal_continuation(nominal, directory=source_directory / "recovery_outline",
                    context_binding=nominal_continuation_context_binding(saved_session, prepared))
                if require_nominal_complete and (proof.get("complete") is not True or proof.get("status") != "passed"
                        or proof.get("execution_authorized") is not False
                        or not isinstance(proof.get("tasks"), list)
                        or not isinstance(proof.get("replaced_task_ids"), list)
                        or not isinstance(proof.get("replacement_evidence"), dict)):
                    raise ValueError("Saved outline nominal continuation is incomplete or does not retain execution control")
        restore = getattr(self.agent, "restore_recovery_resource_evidence", None)
        if callable(restore):
            restore({} if new_attempt else prepared.get("multi_turn_session_state", {}),
                    source_fingerprint=observation_fingerprint(self._observations))
        if not new_attempt:
            with self._lock:
                self._state.pop("primitive_support_semantics", None)
            self._prepared = deepcopy(prepared)
            self._context_revision = self._observations.get("context_revision", context.revision)
            self._proposal = {}
            self._imported_proposal = None
            self._imported_observations = None
            self._outline_request = {}
            provenance = deepcopy(prepared.get("attempt_provenance", {}))
            if provenance:
                write_record(directory / "attempt_provenance.json", provenance)
            self._publish(
                validation_contract_version=prepared.get("validation_contract_version"),
                **({"primitive_support_semantics": prepared["primitive_support_semantics"]}
                   if "primitive_support_semantics" in prepared else {}),
                imported_recovery=False, execution_authorized=False,
                attempt_provenance=provenance,
                provenance_path=str(directory / "attempt_provenance.json") if provenance else "",
            )
            return
        prepared.pop("multi_turn_session_state", None)
        prepared["nominal_continuation_artifact_version"] = 1
        prepared["source_observation_fingerprint"] = observation_fingerprint(self._observations)
        prepared["outline_only"] = self._state["outline_only"]
        prepared["operator_assistance"] = deepcopy(self._state["operator_assistance"])
        recorded_failure = prepared.get("failure_context_raw", {}).get("failure_context", {})
        for target in (prepared.setdefault("llm_input", {}).setdefault("fault_event", {}),
                       prepared.setdefault("context_summary", {}).setdefault("fault_event", {})):
            if isinstance(recorded_failure.get("failure_mode"), str):
                target["failure_type"] = recorded_failure["failure_mode"]
            checkpoint = _recorded_checkpoint(self._observations)
            if checkpoint:
                target["checkpoint"] = checkpoint
        for field in ("unavailable_resources", "nominal_permitted_resources", "recovery_permitted_resources"):
            if field in self._observations:
                prepared[field] = deepcopy(self._observations[field])
        prepared["offline_recovery"] = self.offline
        if self.offline:
            from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_resource_adapter import (
                adapt_recovery_resource_snapshot, recovery_resource_capabilities,
            )
            from cais_spade_llm.resources.resource_profile import resource_snapshot_declared_fields
            from cais_spade_llm.resources.resource_primitives import build_synthesis_primitive_catalog

            for jid, entry in prepared.get("recovery_resources", {}).items():
                evidence = self._observations.get("resource_evidence", {}).get(jid, {})
                if not isinstance(evidence.get("snapshot"), dict):
                    continue
                owner = self.agent.resources[jid]
                owner_evidence = owner.get_recovery_resource_evidence()
                recorded_model = entry.get("recovery_des_model") or {}
                if (not recorded_model.get("descriptor_fingerprint")
                        and not (entry.get("resource_type") != "robot" and
                                 owner_evidence.get("environment_models", {}).get(owner_evidence.get("resource_id")))):
                    # Older captures retain physical facts for resources without
                    # a recovery descriptor. A dependent event must request that
                    # descriptor; unrelated resources do not block preparation.
                    entry["resource_evidence"] = owner_evidence
                    continue
                prepare_capabilities = getattr(owner, "prepare_current_scene_capabilities", None)
                if callable(prepare_capabilities):
                    calculation = prepare_capabilities()
                    if calculation:
                        prepared.setdefault("scene_capability_calculations", {})[jid] = calculation
                snapshot = owner.get_recovery_snapshot()
                model = owner.recovery_des_model(snapshot=snapshot)
                variables = model.get("state_variables", {})
                entry["recovery_snapshot"] = adapt_recovery_resource_snapshot(
                    resource_jid=jid, resource_type=entry["resource_type"], snapshot=snapshot,
                    modeled_state=entry.get("modeled_state", {}), state_variables=variables)
                entry["recovery_des_model"] = model
                entry["recovery_des_model_fingerprint"] = model["descriptor_fingerprint"]
                entry["resource_core"] = deepcopy(entry["recovery_snapshot"]["resource_core"])
                entry["resource_facets"] = deepcopy(entry["recovery_snapshot"]["resource_facets"])
                entry["static_capabilities"] = deepcopy(owner.static_capabilities)
                entry["resource_evidence"] = owner.get_recovery_resource_evidence()
                if entry.get("resource_type") != "robot" and evidence.get("environment_models"):
                    catalog = deepcopy(model.get("primitive_catalog", []))
                    entry["execution_primitive_catalog"] = catalog
                    entry["primitive_catalog"] = build_synthesis_primitive_catalog(primitive_catalog=catalog) or []
                    entry["recovery_adapter"] = recovery_resource_capabilities(
                        entry["resource_type"], primitive_catalog=catalog)
                facts = resource_snapshot_declared_fields(snapshot, variables)
                for row in prepared.get("llm_input", {}).get("observed_runtime_state", {}).get("resources", []):
                    if row.get("resource_jid") == jid:
                        for name, declaration in variables.items():
                            if declaration.get("scope", "resource") == "resource":
                                row.pop(name, None)
                        row.update(deepcopy(facts))
                if jid == prepared.get("ra_jid"):
                    prepared["recovery_snapshot"] = deepcopy(entry["recovery_snapshot"])
        from cais_spade_llm.recovery_framework.outline_validation import (
            OUTLINE_VALIDATION_CONTRACT_VERSION,
            load_outline_specifications,
            outline_evidence_requirements,
        )

        prepared["validation_contract_version"] = OUTLINE_VALIDATION_CONTRACT_VERSION
        from .outline_evidence_policy import OUTLINE_EVIDENCE_POLICY_VERSION

        prepared["outline_evidence_policy_version"] = OUTLINE_EVIDENCE_POLICY_VERSION
        prepared.setdefault("recovery_session", {}).update(
            outline_mode="incremental_candidates_validated", recovery_selection_mode="neurosymbolic",
            outline_context_retrieval=True, action_horizon="1", candidate_bound=5, candidate_count="auto")
        prepared["primitive_support_semantics"] = "successor_primitive_union"
        prepared["nominal_continuation_semantics"] = "recorded_unfinished_tasks"
        prepared["public_safety_context_version"] = 1
        prepared["physical_evidence"] = deepcopy(self._observations.get("physical_evidence") or {})
        prepared["physical_evidence"]["physical_preview_contract_version"] = 7
        prepared["physical_evidence"]["owner_projection_clock_version"] = 1
        physical = prepared["physical_evidence"]
        for jid, calculation in prepared.get("scene_capability_calculations", {}).items():
            rid = calculation["resource_id"]
            if rid in physical.get("resource_models", {}):
                physical["resource_models"][rid]["static_capabilities"] = deepcopy(calculation["static_capabilities"])
        from .outline_endpoint_evidence import POLICY_VERSION

        physical["endpoint_evidence_policy_version"] = POLICY_VERSION
        physical["outline_evidence_policy_version"] = OUTLINE_EVIDENCE_POLICY_VERSION
        description = (self._observations.get("planning_scene_capture") or {}).get("robot_description")
        if description is not None:
            physical["endpoint_robot_description"] = description
        if "processing_observations" in self._observations:
            from cais_spade_llm.recovery_framework.recovery_observations import (
                processing_capture_compatibility, recorded_processing_fields,
            )

            physical["machine_job_control_version"] = 1
            physical["processing_observations"] = deepcopy(self._observations["processing_observations"])
            physical["unavailable_resources"] = deepcopy(self._observations.get("unavailable_resources", []))
            physical["processing_observation_source"] = {
                "run_id": self._observations["run_id"], "path": "failure/processing_observations",
                "kind": "failure-time owner records; shutdown is recorded separately"}
            physical["processing_capture_compatibility"] = processing_capture_compatibility(self._observations)
            for rid, record in physical["processing_observations"].items():
                collections = (physical.get("snapshot", {}).get("resources", {}),
                               physical.get("environment_valuation", {}))
                if not any(rid in collection for collection in collections):
                    continue
                fields = recorded_processing_fields(record, resource_id=rid,
                    run_id=self._observations["run_id"], inputs=self._observations.get("inputs") or {})
                for collection in collections:
                    if rid in collection:
                        collection[rid].update(deepcopy(fields))
                physical.setdefault("processing_field_sources", {})[rid] = {
                    "fields": sorted(fields), "source": f"/processing_observations/{rid}",
                    "revision": record["revision"], "simulation_time": record.get("simulation_clock_sec"),
                    "kind": "recorded machine owner fields; physical measurements remain separate"}
        for rid, jid in physical.get("jids", {}).items():
            entry = prepared.get("recovery_resources", {}).get(jid, {})
            if "static_capabilities" not in entry:
                continue
            model = physical.setdefault("resource_models", {}).setdefault(rid, {})
            recorded = entry["static_capabilities"]
            if "static_capabilities" in model and model["static_capabilities"] != recorded:
                raise ValueError("Recorded resource static_capabilities conflict with the physical model")
            model["static_capabilities"] = deepcopy(recorded)
        for jid, evidence in self._observations.get("resource_evidence", {}).items():
            if jid in prepared.get("recovery_resources", {}):
                prepared["recovery_resources"][jid]["resource_evidence"] = (
                    self.agent.resources[jid].get_recovery_resource_evidence() if self.offline else deepcopy(evidence))
        if "pending_nodes" in self._observations:
            prepared["saved_pending_nodes"] = deepcopy(self._observations["pending_nodes"])
        prepared["completed_task_ids"] = deepcopy(self._observations.get("completed_task_ids") or [])
        prepared["observation_jids"] = deepcopy(self._observations.get("jids") or {})
        prepared["monitor_history"] = deepcopy(self._observations.get("monitor_history") or {})
        prepared["outline_monitor_context"] = deepcopy(self._observations.get("outline_monitor_context") or {})
        prepared["outline_safety_specifications"] = load_outline_specifications(
            self.runtime.setup.get("selected_safety_files", []), ROOT)
        from cais_spade_llm.recovery_framework.recovery_observations import restore_recorded_resource_geometry

        prepared["physical_evidence"]["outline_safety_specifications"] = deepcopy(prepared["outline_safety_specifications"])
        prepared["physical_evidence"] = restore_recorded_resource_geometry(
            prepared["physical_evidence"], self._observations.get("inputs") or {})
        write_record(directory / "recorded_geometry_supplements.json",
                     prepared["physical_evidence"].get("recorded_geometry_supplements", {}))
        prepared["outline_evidence_requirements"] = outline_evidence_requirements(prepared)
        write_record(directory / "outline_evidence_requirements.json", prepared["outline_evidence_requirements"])
        self._publish(validation_contract_version=OUTLINE_VALIDATION_CONTRACT_VERSION,
                      primitive_support_semantics=prepared["primitive_support_semantics"],
                      outline_safety_status="pending", execution_authorized=False)
        prepared["source_observation_captured_at"] = self._observations.get("captured_at")
        prepared["source_observation_run_id"] = self._observations.get("run_id")
        prepared["source_context_revision"] = self._observations.get("context_revision", context.revision)
        prepared["recovery_session"].update(
            reasoning_mode="multi_turn", max_turns=80,
            outline_context_retrieval=True,
            session_id=f"{context.run_id}_{self._generation:04d}",
        )
        prepared["recovery_feedback"] = feedback
        prepared["recovery_debug"].update(artifact_directory=str(directory), per_turn_debug_dir=str(directory))
        prepared["prompt_safety_rules"] = _prompt_safety_rules(self.runtime.setup.get("selected_safety_files", []))
        from cais_spade_llm.recovery_framework.diagnostics import validator_fingerprint

        source = getattr(self.runtime, "source_observation_path", None) if self.offline else self.path / "observations.json"
        source_sha256 = None
        if source and Path(source).is_file():
            with Path(source).open("rb") as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                source_sha256 = digest.hexdigest()
        provenance = {
            "version": 1,
            "capture": {"run_id": self._observations.get("run_id"),
                        "captured_at": self._observations.get("captured_at"),
                        "context_revision": self._observations.get("context_revision"),
                        "observation_fingerprint": observation_fingerprint(self._observations),
                        "source_sha256": source_sha256},
            "configuration_fingerprint": fingerprint(self._observations["inputs"]) if "inputs" in self._observations else None,
            "model_fingerprint": self._observations.get("planning_scene_capture", {}).get("model_fingerprint"),
            "safety_specifications": [
                {"path": name, "sha256": hashlib.sha256((Path(name) if Path(name).is_absolute() else ROOT / name).read_bytes()).hexdigest()}
                for name in self.runtime.setup.get("selected_safety_files", [])],
            "implementation_fingerprint": validator_fingerprint(),
            "operator_assistance": deepcopy(self._state["operator_assistance"]),
            "unassisted": not feedback and not self._state["operator_assistance"],
        }
        prepared["attempt_provenance"] = provenance
        write_record(directory / "attempt_provenance.json", provenance)
        self._publish(attempt_provenance=provenance, provenance_path=str(directory / "attempt_provenance.json"))
        write_record(directory / "prompt_safety_rules.json", {
            "prompt_safety_rules": prepared["prompt_safety_rules"], "safety_status": "skipped",
        })
        if self.offline:
            tracker = prepared.setdefault("part_tracker", {})
            grounded_parts = prepared.get("grounding_context", {}).get("parts", {})
            captured_parts = self._observations.get("part_tracker", {})
            for name in grounded_parts.keys() | captured_parts.keys():
                recorded = captured_parts.get(name) or grounded_parts.get(name, {})
                entry = tracker.setdefault(name, deepcopy(recorded))
                pose = recorded.get("observed_pose") or grounded_parts.get(name, {}).get("observed_pose")
                if not entry.get("observed_pose") and pose:
                    entry["observed_pose"] = deepcopy(pose)
        from cais_spade_llm.recovery_framework.recovery_observations import restore_recorded_public_facts

        write_record(directory / "recorded_public_fact_sources.json",
                     restore_recorded_public_facts(prepared, self._observations))
        if self.offline:
            grounded = apply_grounding(prepared, self._observations)
            if grounded is not None:
                write_record(directory / "failure_grounding.json", grounded)
        prepared["multi_turn_session_seed"] = build_multi_turn_session_seed(prepared)
        prepared["multi_turn_session_seed"]["public_safety_context_version"] = prepared["public_safety_context_version"]
        if self.offline:
            _seed_grounded_part_pose_observations(prepared, prepared["multi_turn_session_seed"])
        preview, prompt = _build_phase_prompt(prepared, prepared["multi_turn_session_seed"])
        write_record(directory / "input_preview.json", {
            "request_sent": False, "preview_status": "not sent", "prompt_input": preview, "prompt": prompt,
        })
        write_record(directory / "inputs.json", {
            "run_id": context.run_id, "safety_status": "skipped", "failure": self._failure,
            "prepared_recovery_request": prepared,
            "llm_settings": deepcopy(getattr(self.agent, "llm_settings", {})),
            "source_observation_fingerprint": observation_fingerprint(self._observations),
        })
        self._prepared = prepared
        self._context_revision = self._observations.get("context_revision", context.revision)
        self._proposal = {}
        self._imported_proposal = None
        self._imported_observations = None
        self._outline_request = {}
        self._publish(imported_recovery=False)

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
        grounded = self._ground_beliefs()
        if not grounded["belief_updates"]:
            return physical
        # Belief-dependent views were derived before grounding; retake them.
        physical = await self._capture_physical_evidence()
        if fault_identity(ground_failure(self._observations)) != fault_identity(grounded):
            raise ValueError("NEEDS_CONTEXT: the retaken physical capture grounds a different fault")
        return physical

    def _ground_beliefs(self) -> dict:
        """Ground the observed fault from raw facts and update the product's beliefs.

        Failure injection leaves the beliefs at their acknowledged values, so
        execution monitoring applies the grounded fault state before preparation.

        Returns:
            The grounding, whose ``belief_updates`` were applied.

        Raises:
            ValueError: No failure is detectable from the recorded facts.
        """
        grounded = ground_failure(self._observations)
        if not grounded["fault_event"]["detected"]:
            raise ValueError("NEEDS_CONTEXT: no failure is detectable from the recorded execution-monitoring facts")
        if update_beliefs(self.runtime.context, self._observations, grounded):
            self._observations["context_revision"] = self.runtime.context.revision
        self._save_observations()
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
        preview_attempts: set[str] = set()
        provider_turns = 0
        while True:
            if self._stopped or generation != self._generation:
                raise asyncio.CancelledError()
            prepared = self._prepared
            session = prepared.get("multi_turn_session_state", {})
            if self.offline and stage == "outline" and session.get("pending_outline_validation"):
                await self._prepare_offline_outline_evidence(preview_attempts)
                session = self._prepared.get("multi_turn_session_state", {})
                if (session.get("current_phase") in {"primitive_generation", "finalize"}
                        and session.get("accepted_outline_prefix")):
                    return session
                if session.get("pending_outline_validation"):
                    continue
                self._publish(status="generating_outline", outline_status="generating",
                              message="Continuing recovery outline after recorded validation.")
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
            if self.offline and stage == "outline" and session.get("pending_outline_validation"):
                continue
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

    def _offline_preview_observations(self, archive: dict | None = None) -> dict:
        """Read a verified private model view without modifying saved inputs."""
        from cais_spade_llm.recovery_framework.recovery_observations import (
            native_preview_observations, read_native_preview_model,
        )

        source = getattr(self.runtime, "source_observation_path", None)
        if not source:
            # Existing callers without a companion retain their exact capture.
            return self._observations
        if archive is None:
            archive = read_native_preview_model(source, self._observations)
        return native_preview_observations(self._observations, archive)

    async def _prepare_offline_outline_evidence(self, attempted: set[str]) -> None:
        """Calculate saved-model evidence and retry the unchanged pending task."""
        from .outline_endpoint_evidence import endpoint_policy

        if endpoint_policy(self._prepared.get("physical_evidence") or {}):
            session = self._prepared.get("multi_turn_session_state", {})
            raise ValueError("NEEDS_CONTEXT: recovery outline validation requires additional facts or corrected candidate bindings. "
                + str(session.get("outline_validation_findings") or "Required validation evidence is unavailable"))
        from cais_spade_llm.recovery_framework.outline_gazebo_evidence import (
            pending_outline_event, prepare_owner_event_evidence,
        )
        from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_recovery_motion_from_files
        from cais_spade_llm.recovery_framework.recovery_observations import (
            _file_sha256, fingerprint, read_native_preview_model,
        )
        from cais_spade_llm.resources.recovery_geometry import outline_event_binding

        source = getattr(self.runtime, "source_observation_path", None)
        if not source:
            raise ValueError("NEEDS_CONTEXT: the saved observation path is required")
        session = self._prepared["multi_turn_session_state"]
        pending = session["pending_outline_validation"]
        if pending.get("kind") == "nominal" and not isinstance(pending.get("task"), dict):
            from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
                nominal_continuation_context_binding, read_nominal_continuation,
            )

            proof = read_nominal_continuation(session.get("nominal_continuation_evidence") or {},
                directory=Path(self._state["artifact_directory"]) / "recovery_outline",
                context_binding=nominal_continuation_context_binding(session, self._prepared))
            reasons = [str(row["reason"]) for row in proof.get("failures", []) if row.get("reason")]
            raise ValueError("NEEDS_CONTEXT: " + str(proof.get("reason", "Nominal continuation is unresolved"))
                             + (": " + "; ".join(dict.fromkeys(reasons)) if reasons else ""))
        task = pending_outline_event(self)
        entry = self._prepared["recovery_resources"][task["resource_jid"]]
        snapshot = deepcopy(pending.get("recovery_snapshot", session.get("derived_resource_snapshots", {}).get(
            task["resource_jid"], entry["recovery_snapshot"])))
        physical = deepcopy(pending.get("projected_physical_evidence", session.get(
            "projected_physical_evidence", self._prepared["physical_evidence"])))
        owner = self.agent.resources[task["resource_jid"]]
        resource_evidence = owner.get_recovery_resource_evidence()
        archive = None
        observations = deepcopy(self._observations)
        from cais_spade_llm.recovery_framework.recovery_observations import projection_capture_compatibility

        compatibility = projection_capture_compatibility(physical=physical, task=task,
                                                         resource_evidence=resource_evidence)
        if (resource_evidence.get("resource_type") in {"robot", "kmr"}
                and compatibility["status"] != "NEEDS_CONTEXT"):
            archive = read_native_preview_model(source, self._observations)
            observations = self._offline_preview_observations(archive)
        else:
            recorded_sha256 = self._prepared.get("attempt_provenance", {}).get("capture", {}).get("source_sha256")
            if not recorded_sha256 or _file_sha256(Path(source)) != recorded_sha256:
                raise ValueError("NEEDS_CONTEXT: saved observations changed or lack their recorded source fingerprint")
        identity = fingerprint({"task": task, "binding": outline_event_binding(
            task, snapshot, entry["recovery_des_model"], physical), "resource_evidence": resource_evidence,
            "manifest_sha256": archive["manifest_sha256"] if archive is not None else None})
        if identity in attempted:
            raise ValueError("NEEDS_CONTEXT: saved-model calculations provide no additional evidence for this proposal: "
                + str(session.get("outline_validation_findings") or pending))
        attempted.add(identity)
        directory = Path(self._state["artifact_directory"]) / "offline_validation" / identity
        directory.mkdir(parents=True, exist_ok=True)
        self._publish(status="generating_outline", outline_status="generating",
                      message="Calculating physical evidence from the saved scene; Gazebo is not required.")

        async def preview(request, **kwargs) -> dict:
            if archive is None:
                raise ValueError("NEEDS_CONTEXT: this resource has no recorded native preview model")
            kwargs.pop("action_name", None)
            return await preview_recovery_motion_from_files(request, model_archive=archive,
                artifact_directory=directory, **kwargs)

        def calculate() -> dict:
            evidence = asyncio.run(prepare_owner_event_evidence(owner, task=task, recovery_snapshot=snapshot,
                recovery_des_model=deepcopy(entry["recovery_des_model"]), physical_evidence=physical,
                resource_evidence=resource_evidence, observations=observations, cancel_event=self._offline_cancel,
                preview_call=preview, allow_live_capture=False))
            write_record(directory / "evidence.json", evidence)
            return evidence

        evidence = await self._offline_work(calculate)
        if evidence.get("status") == "NEEDS_CONTEXT":
            from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
                write_recovery_validation_record,
            )

            write_recovery_validation_record(Path(self._state["artifact_directory"]) / "recovery_outline", {
                "turn_index": pending.get("source_turn_index", session.get("turn_index", 0)),
                "source_turn_index": pending.get("source_turn_index", session.get("turn_index", 0)),
                "phase": "outline", "decision": "need_context", "validation_only": True, "request_sent": False,
                "validation_stages": [{"validator_role": "ResourceAgent", "stage": "physical_projection",
                                       "evidence": deepcopy(evidence)}],
            })
            message = "NEEDS_CONTEXT: " + str(evidence.get("reason") or "Required saved physical evidence is unavailable")
            self._publish(status="blocked", outline_status="NEEDS_CONTEXT", outline_safety_status="unresolved",
                          message=message)
            raise ValueError(message)
        await self.retry_outline_validation(evidence)

    async def _generate_outline(self) -> None:
        self._publish(status="generating_outline", outline_status="generating", primitive_status="waiting",
                      outline=[], primitives=[], message="Generating recovery outline.")
        session = await self._run_stage("outline")
        self._outline_request = deepcopy(self._prepared)
        write_record(Path(self._state["artifact_directory"]) / "outline_request.json", self._outline_request)
        write_record(Path(self._state["artifact_directory"]) / "outline.json",
                     {"outline": session["accepted_outline_prefix"]})
        self._publish(status="outline_review", outline_status="completed",
                      outline=session["accepted_outline_prefix"],
                      outline_safety_status="passed" if self._prepared.get("outline_safety_specifications") else "not_selected",
                      message="Recovery outline completed with RA and CCA validation. Execution validation remains pending.")
        if not self.offline and not self._state["outline_only"] and self._state["outline_mode"] == "auto":
            await self._continue_outline()

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
        operations = {
            "prepare": ({"blocked"}, self._begin),
            "generate_outline": ({"outline_ready"}, self._generate_outline),
            "continue_outline": ({"outline_review"}, self._continue_outline),
            "generate_primitives": ({"primitive_ready", "outline_review"} if self.offline else {"primitive_ready"},
                                    self._generate_primitives_from_review),
            "execute": ({"ready_to_execute"}, self._execute),
        }
        if action in {"regenerate_outline", "regenerate_primitives"}:
            if status not in {"outline_review", "primitive_ready", "ready_to_execute", "offline_completed", "blocked"}:
                raise ValueError("Wait for the current stage before regenerating")

            async def regenerate() -> None:
                if action == "regenerate_outline":
                    await self._prepare(feedback)
                    await self._generate_outline()
                else:
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

    def stop(self, reason: str = "Stopped by operator") -> None:
        """Invalidate pending generation and cancel this run's owned operation."""
        self._stopped = True
        self._offline_cancel.set()
        self._generation += 1
        if self._task is not None and not self._task.done():
            self._task.get_loop().call_soon_threadsafe(self._task.cancel)
        if self._state["status"] not in {"completed", "offline_completed"}:
            self._publish(status="stopped", message=reason)
