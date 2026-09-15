from __future__ import annotations

"""Run the exact validated program through an owned simulation-only transport."""

import asyncio
import fcntl
import hashlib
import logging
import threading
import time
import uuid

from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .execution_state import (
    ACTIVE,
    assert_execution_available,
    assert_interaction_current,
    execution_busy,
    execution_custody,
    read_primitive_execution_diagnostic,
    reset_history,
    verified_reset,
)
from .primitive_composition import (
    _action,
    _attempt_paths,
    _evidence_value,
    _load_inputs,
    _parse_steps,
    _read_record,
    _recorded_request_inputs,
    _validate_steps,
)
from .program_validation import resolve_selected_values, validate_program
from .validation_scope import (
    GAZEBO_LINK_ATTACHER_SCOPE, is_observed_scope, is_pick_place_scope,
    read_validation_scope, supported_primitive_symbols,
)
from .refinement import _ACTIVE as ACTIVE_COMPOSITIONS
from .refinement_records import (
    append_record,
    fingerprint,
    owned_path,
    pin,
    read_pin,
    verify_evidence_tree,
    verify_record,
)
from ...adapters.gazebo_execution import (
    GazeboExecutionSession,
    GripperCommandError,
    fixture_instances,
    load_execution_profile,
    match_instance,
)
from ...adapters.dual_gazebo import (
    GazeboResetProbe, assert_reset_interlocks, read_dual_gazebo_started_at_ns, read_dual_gazebo_status,
    start_dual_gazebo, stop_dual_gazebo,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidatedProgram:
    """Retain the immutable program authority and its recorded validation inputs."""

    inputs: Any
    candidate_ref: dict[str, str]
    validation_ref: dict[str, str]
    result_ref: dict[str, str]
    steps: list[dict[str, Any]]
    report: dict[str, Any]
    robot: dict[str, Any]
    profile: dict[str, Any]
    binding_ref: dict[str, str] | None = None


def load_validated_program(root: Path) -> ValidatedProgram:
    """Reject drafts, stale authority, or mismatched candidate/report lineage."""
    inputs = _load_inputs(root)
    base_inputs = inputs
    runs = sorted((root / "composition/refinement_runs").glob("run_*"))
    if not runs or not (runs[-1] / "result.json").is_file():
        raise ValueError(
            "Run in Gazebo requires the current program to be validated_for_declared_scope."
        )
    directory = runs[-1]
    result_ref = pin(root, directory / "result.json")
    result = verify_record(root, result_ref)
    if result.get("status") != "validated_for_declared_scope":
        raise ValueError("Run in Gazebo requires the current program to be validated_for_declared_scope.")
    request_ref = result["request_ref"]
    request = verify_record(root, request_ref)
    scope = read_validation_scope(request["profile"])
    if (
        request_ref != pin(root, directory / "request.json")
        or request.get("record_type") != "PrimitiveRefinementRequest"
        or request["base_context_refs"] != inputs.context_refs
    ):
        raise ValueError("The validated run has changed authority.")
    # Execution verifies the selected run's actual dependencies. Rendering every
    # historical candidate is UI work and grows with unrelated failed attempts.
    for key in ("candidate_refs", "decision_refs", "validation_refs", "binding_refs", "event_refs"):
        for reference in result.get(key, []):
            read_pin(root, reference)
    events = [verify_record(root, reference) for reference in result["event_refs"]]
    recorded_events = {reference["ref"] for reference in result["event_refs"]}
    for path in directory.glob("event_*.json"):
        if path.relative_to(root).as_posix() not in recorded_events:
            terminal = verify_record(root, pin(root, path))
            if (path.name != f"event_{len(events) + 1:04d}.json"
                    or terminal.get("stage") != "finished"
                    or terminal.get("status") != result["status"]):
                raise ValueError("The validated run's events changed after completion.")
    candidate_ref, validation_ref = result["candidate_refs"][-1], result["validation_refs"][-1]
    attempts = _attempt_paths(root)
    if not attempts or candidate_ref["ref"] != (attempts[-1] / "candidate.json").relative_to(root).as_posix():
        raise ValueError("A newer composition attempt supersedes the validated program.")
    candidate = read_pin(root, candidate_ref)
    candidate_path = owned_path(root, candidate_ref["ref"])
    if _read_record(candidate_path) != candidate or candidate.get("status") != "proposed":
        raise ValueError("The validated candidate is not an unchanged RA proposal.")
    if candidate["request_ref"] != (attempts[-1] / "request.json").relative_to(root).as_posix():
        raise ValueError("The candidate references another attempt's request.")
    saved_request = _read_record(owned_path(root, candidate["request_ref"]))
    read_pin(root, {"ref": candidate["request_ref"], "sha256": candidate["request_sha256"]})
    if read_validation_scope(saved_request) != scope:
        raise ValueError("The authored program and refinement run have different validation scopes.")
    inputs = _recorded_request_inputs(inputs, saved_request)
    if saved_request["context_refs"] != inputs.context_refs:
        raise ValueError("The candidate request has different context authority.")
    if inputs.includes_execution_identifiers:
        raise ValueError(
            "Compose a new program with the current composition interface before execution."
        )
    report = verify_record(root, validation_ref)
    proposals = [event for event in events if event["stage"] == "proposal"]
    validations = [event for event in events if event["stage"] == "validation_result"]
    if (not proposals or proposals[-1]["candidate_ref"] != candidate_ref
            or not validations or validations[-1]["validation_ref"] != validation_ref):
        raise ValueError("The selected program differs from the latest proposal or validation.")
    exchanges = []
    for index, reference in enumerate(candidate["exchange_refs"], 1):
        expected = attempts[-1] / f"exchange_{index:04d}.json"
        if reference["ref"] != expected.relative_to(root).as_posix():
            raise ValueError("The candidate references another attempt's RA exchange.")
        exchanges.append(_read_record(owned_path(root, reference["ref"])))
        read_pin(root, reference)
    if not exchanges:
        raise ValueError("The candidate has no recorded RA submission.")
    submitted = _action(exchanges[-1]["response"])
    if submitted["kind"] != "propose" or _parse_steps(submitted["primitive_steps"]) != candidate["primitive_steps"]:
        raise ValueError("Candidate differs from the RA submission.")
    _validate_steps(candidate["primitive_steps"], inputs)
    binding_ref = result.get("binding_refs", [None])[-1] if result.get("binding_refs") else None
    bindings = [event["binding_ref"] for event in events if event["stage"] == "binding"]
    if report.get("binding_ref") != binding_ref or (bindings[-1] if bindings else None) != binding_ref:
        raise ValueError("The displayed binding differs from the validated program.")
    if binding_ref is not None:
        from .program_binding import read_program_binding

        binding, inputs = read_program_binding(base_inputs, binding_ref)
        if binding["candidate_ref"] != candidate_ref:
            raise ValueError("The validated binding belongs to another RA proposal.")
        steps = deepcopy(binding["primitive_steps"])
    else:
        steps = deepcopy(candidate["primitive_steps"])
    if (
        report["status"] != "passed"
        or report["scope"] != scope
        or report["candidate_ref"] != candidate_ref
        or report["candidate_fingerprint"] != fingerprint(steps)
        or any(step["primitive_symbol"] not in supported_primitive_symbols(scope) for step in steps)
    ):
        raise ValueError("The saved report does not validate this exact supported program.")
    for reference in report["evidence_refs"].values():
        verify_evidence_tree(root, reference)
    robot = verify_record(root, report["final_robot_context_ref"])
    if (
        robot.get("record_type") != "RobotValidationContext"
        or robot.get("resource_jid") != inputs.assignment.selected_resource_jid
    ):
        raise ValueError("The validated robot context belongs to a different resource.")
    return ValidatedProgram(
        inputs, candidate_ref, validation_ref, result_ref, steps, report, robot, request["profile"], binding_ref
    )


@dataclass
class _Session:
    task: asyncio.Task[Any]
    stop: threading.Event
    operation: str = "execution"
    diagnostic: dict[str, Any] | None = None


class PrimitiveExecutionRuntime:
    """Coordinate a whole saved program independently of the browser lifetime."""

    def __init__(
        self,
        *,
        robot_runtime: Any,
        capture_runtime: Any = None,
        profile: Mapping[str, Any] | None = None,
        session_factory: Any = GazeboExecutionSession,
        validator: Any = validate_program,
        share: Path | None = None,
        dual_gazebo: Any = None,
        reset_probe_factory: Any = GazeboResetProbe,
    ) -> None:
        """Inject simulation configuration and command transport for the saved program."""
        self.robot_runtime, self.capture_runtime = robot_runtime, capture_runtime
        self.profile = dict(profile) if profile is not None else load_execution_profile()
        self.session_factory, self.validator, self.share = session_factory, validator, share
        self.dual_gazebo, self.reset_probe_factory = dual_gazebo, reset_probe_factory

    async def run(
        self,
        root: Path,
        *,
        candidate_ref: str | None = None,
        binding_ref: str | None = None,
        progress: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Join duplicate clicks; permit only one owned execution across interactions."""
        return await self._operate(root, "execution", progress,
                                   lambda stop, emit: self._run(root.resolve(), stop, emit, candidate_ref, binding_ref))

    async def _operate(self, root: Path, operation: str, progress: Any, run: Any) -> dict[str, Any]:
        root = root.resolve()
        session = ACTIVE.get(root)
        if session is not None and not session.task.done():
            if session.operation != operation:
                raise ValueError("A different Gazebo operation is active.")
            return await asyncio.shield(session.task)
        if execution_busy() or any(not task.done() for task in tuple(ACTIVE_COMPOSITIONS.values())):
            raise ValueError("Another composition or Gazebo execution is active.")
        stop = threading.Event()
        async def emit(event: Mapping[str, Any]) -> None:
            ACTIVE[root].diagnostic = dict(event)
            if progress is not None:
                try:
                    await progress(event)
                except (RuntimeError, OSError, TypeError, ValueError):
                    logger.exception("Gazebo progress display detached; operation remains owned.")

        task = asyncio.create_task(run(stop, emit))
        ACTIVE[root] = _Session(task, stop, operation, {
            "status": "preparing" if operation == "execution" else "resetting",
            "message": "Preparing Gazebo execution." if operation == "execution" else "Preparing verified Gazebo reset.",
        })
        try:
            return await asyncio.shield(task)
        finally:
            # Browser cancellation must not release the execution guard.
            task.add_done_callback(
                lambda done: ACTIVE.pop(root, None)
                if ACTIVE.get(root) is not None and ACTIVE[root].task is done
                else None
            )

    def stop(self, root: Path) -> bool:
        """Request transport cancellation without abandoning a running worker."""
        session = ACTIVE.get(root.resolve())
        if session is None or session.task.done():
            return False
        session.stop.set()
        return True

    def diagnostic(self, root: Path) -> dict[str, Any]:
        """Read persisted progress without contacting a robot."""
        return read_primitive_execution_diagnostic(root)

    async def reset_simulation(
        self, root: Path, *, progress: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Reset both simulated robots and invalidate old interactions after verified replacement.

        Args:
            root: Interaction owning the immutable reset journal.
            progress: Optional UI observer; its lifetime does not own the reset.

        Returns:
            A persisted reset result. Only reset_completed clears historical custody.
        """
        if self.dual_gazebo is None:
            raise RuntimeError("The owned dual-Gazebo reset adapter is unavailable.")
        return await self._operate(root, "reset", progress,
                                   lambda stop, emit: self._reset(root.resolve(), stop, emit))

    async def _reset(self, root: Path, stop: threading.Event, progress: Any) -> dict[str, Any]:
        from .refinement import load_refinement_profile
        from ...adapters.in_process_robot_agent import (
            _ROBOT_AGENT_STARTUP_TIMEOUT_SECONDS, _ROBOT_AGENT_STARTUP_POLL_SECONDS,
        )

        with (root.parent / ".gazebo_execution.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another application owns Gazebo execution or composition.") from exc
            await asyncio.to_thread(assert_reset_interlocks, self.dual_gazebo)
            history = await asyncio.to_thread(reset_history, root.parent)
            required_joints = set()
            for historical in history:
                for reference in historical["records"]:
                    name = Path(reference["ref"]).name
                    owner = root.parent / historical["interaction"]
                    if name == "prepared.json":
                        prepared = await asyncio.to_thread(verify_record, owner, reference)
                        robot = await asyncio.to_thread(verify_record, owner, prepared["robot_context_ref"])
                    elif name in {"robot_initial.json", "robot_ready.json"}:
                        robot = await asyncio.to_thread(verify_record, owner, reference)
                    else:
                        continue
                    required_joints.update(robot["joint_state"]["names"])
            if not required_joints:
                raise ValueError("No verified historical robot feedback is available to verify both robots after reset.")
            profile = load_refinement_profile()
            probe_profile = {**self.profile, "state_max_age_sec": profile["state_max_age_sec"],
                             "joint_states_topics": sorted({resource["joint_states_topic"] for resource in profile["resources"].values()}),
                             "readiness_timeout_sec": _ROBOT_AGENT_STARTUP_TIMEOUT_SECONDS}
            directory = root / "execution" / f"reset_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
            request_ref = await asyncio.to_thread(append_record, root, directory, "request.json", {
                "record_type": "PrimitiveExecutionResetRequest", "created_at_ns": time.time_ns(),
                "history": history, "required_joints": sorted(required_joints),
                "invalidated_interactions": sorted(path.name for path in root.parent.iterdir() if path.is_dir()),
                "resources": sorted(profile["resources"]),
            })
            previous = None
            count = 0

            async def emit(message: str) -> None:
                nonlocal previous, count
                count += 1
                event = {"record_type": "PrimitiveExecutionResetEvent", "status": "resetting",
                         "message": message, "previous_event_ref": previous, "created_at_ns": time.time_ns()}
                previous = await asyncio.to_thread(append_record, root, directory, f"event_{count:04d}.json", event)
                await progress(event)

            def check() -> None:
                if stop.is_set():
                    raise RuntimeError("Gazebo reset interrupted; a verified reset is still required.")
                assert_reset_interlocks(self.dual_gazebo)

            result: dict[str, Any] = {"record_type": "PrimitiveExecutionResetResult", "request_ref": request_ref,
                                      "status": "reset_required", "baseline": None, "old_endpoints": None, "stopped": None}
            probe = self.reset_probe_factory(probe_profile, stop)
            try:
                await asyncio.to_thread(probe.__enter__)
                await emit("Checking the old Gazebo process and ROS endpoint identities.")
                result["old_endpoints"] = await asyncio.to_thread(probe.snapshot)
                await asyncio.to_thread(check)
                await emit("Stopping both robots and the shared Gazebo scene.")
                if (await asyncio.to_thread(read_dual_gazebo_status, self.dual_gazebo)).state != "stopped":
                    await asyncio.to_thread(stop_dual_gazebo, self.dual_gazebo)
                result["stopped"] = await asyncio.to_thread(probe.wait_stopped, self.dual_gazebo)
                await asyncio.to_thread(check)
                await emit("Old endpoints are gone. Starting a new shared Gazebo scene.")
                error = await asyncio.to_thread(start_dual_gazebo, self.dual_gazebo)
                if error:
                    raise RuntimeError(error)
                deadline = time.monotonic() + _ROBOT_AGENT_STARTUP_TIMEOUT_SECONDS
                while True:
                    await asyncio.to_thread(check)
                    ready, reason = await asyncio.to_thread(self.dual_gazebo.simulation_start_ready, force=True)
                    if ready:
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Gazebo reset readiness timed out: " + str(reason))
                    await asyncio.sleep(_ROBOT_AGENT_STARTUP_POLL_SECONDS)
                await emit("Verifying a replacement clock and fresh feedback for both robots.")
                result["baseline"] = await asyncio.to_thread(probe.wait_ready, result["old_endpoints"], sorted(required_joints))
                await asyncio.to_thread(check)
                if (await asyncio.to_thread(read_dual_gazebo_status, self.dual_gazebo)).state != "running":
                    raise RuntimeError("Gazebo stopped during reset verification.")
                result.update(status="reset_completed", message=(
                    "Both robots and the shared Gazebo scene were reset and verified. "
                    "Start a fresh interaction, capture new observations and RobotAgent context, then compose again. "
                    "Old programs cannot be replayed."
                ))
            except asyncio.CancelledError:
                stop.set()
                result["message"] = "Gazebo reset interrupted; a verified reset is still required."
            except (ImportError, OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
                result["message"] = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    await asyncio.to_thread(probe.__exit__)
                except (RuntimeError, OSError) as exc:
                    result.update(status="reset_required", message=f"Reset probe cleanup failed: {exc}")
            result.update(last_event_ref=previous, completed_at_ns=time.time_ns())
            reference = await asyncio.to_thread(append_record, root, directory, "result.json", result)
            if result["status"] == "reset_completed":
                await asyncio.to_thread(verified_reset, root.parent)
            await progress(result)
            return await asyncio.to_thread(verify_record, root, reference)

    async def _run(
        self, root: Path, stop: threading.Event, progress: Any, candidate_ref: str | None, binding_ref: str | None
    ) -> dict[str, Any]:
        lock_path = root.parent / ".gazebo_execution.lock"
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another application owns Gazebo execution.") from exc
            return await self._run_locked(root, stop, progress, candidate_ref, binding_ref)

    async def _run_locked(
        self, root: Path, stop: threading.Event, progress: Any, candidate_ref: str | None, binding_ref: str | None
    ) -> dict[str, Any]:
        started_at_ns, started = time.time_ns(), time.monotonic()
        preparation_timings: dict[str, float] = {}
        program = await asyncio.to_thread(load_validated_program, root)
        preparation_timings["load_program_sec"] = time.monotonic() - started
        if binding_ref is not None and (program.binding_ref or {}).get("ref") != binding_ref:
            raise ValueError("The displayed parameter binding changed. Refresh before Run in Gazebo.")
        if candidate_ref is not None and candidate_ref != program.candidate_ref["ref"]:
            raise ValueError(
                "The displayed program changed. Refresh the program before Run in Gazebo."
            )
        assignment = program.inputs.assignment
        if assignment.selected_execution_mode != "simulation":
            raise ValueError("Run in Gazebo supports simulation only.")
        gazebo_started_at_ns = 0
        if self.dual_gazebo is not None:
            gazebo = await asyncio.to_thread(read_dual_gazebo_status, self.dual_gazebo)
            if gazebo.blocked_reason:
                raise RuntimeError(gazebo.blocked_reason)
            if gazebo.state != "running":
                raise RuntimeError(
                    "Start Dual Gazebo Environment with its top Start control before Run in Gazebo."
                )
            gazebo_started_at_ns = read_dual_gazebo_started_at_ns(self.dual_gazebo)
        await asyncio.to_thread(assert_execution_available, root.parent, _started_at_ns=gazebo_started_at_ns)
        await asyncio.to_thread(assert_interaction_current, root, _started_at_ns=gazebo_started_at_ns)
        custody = await asyncio.to_thread(
            execution_custody, root.parent, assignment.selected_resource_jid,
            _started_at_ns=gazebo_started_at_ns,
        )
        stage_started = time.monotonic()
        authority = await self.robot_runtime.execution_configuration(assignment)
        preparation_timings["configuration_sec"] = time.monotonic() - stage_started
        configuration = authority["configuration"]
        transport_profile = {
            **self.profile,
            "joint_states_topic": program.profile["resources"][assignment.selected_resource_jid][
                "joint_states_topic"
            ],
            "state_max_age_sec": program.profile["state_max_age_sec"],
            "fk_orientation_tolerance_rad": program.profile["fk_orientation_tolerance_rad"],
        }
        directory = root / "execution" / ("run_" + str(time.time_ns()) + "_" + uuid.uuid4().hex[:8])
        refs = []
        previous_event = None
        event_count = 0
        completed = 0
        motion_dispatched = False
        command_dispatched = False
        custody_known = True
        gripper_request = None
        held = custody["held_part"] if custody is not None else program.robot.get("held_part")
        # Empty custody does not establish an open gripper. Only an acknowledged
        # gripper command supplies that semantic state to subsequent context.
        gripper_state = custody["gripper_state"] if custody is not None else None
        if held is not None and custody is None:
            raise ValueError("An initially held part requires verified execution custody.")
        request_ref = append_record(
            root,
            directory,
            "request.json",
            {
                "record_type": "PrimitiveExecutionRequest",
                "candidate_ref": program.candidate_ref,
                "binding_ref": program.binding_ref,
                "validation_ref": program.validation_ref,
                "refinement_result_ref": program.result_ref,
                "resource_jid": assignment.selected_resource_jid,
                "total_steps": len(program.steps),
                "profile": transport_profile,
                "validation_scope": program.report["scope"],
                "configuration_sha256": fingerprint(configuration),
                "fresh_simulation": False,
                "started_at_ns": started_at_ns,
                "created_at_ns": time.time_ns(),
            },
        )

        async def record(name: str, payload: Mapping[str, Any]) -> dict[str, str]:
            reference = await asyncio.to_thread(append_record, root, directory, name, payload)
            refs.append(reference)
            return reference

        async def emit(status: str, message: str, **details: Any) -> None:
            nonlocal previous_event, event_count
            event_count += 1
            event = {
                "record_type": "PrimitiveExecutionEvent",
                "status": status,
                "message": message,
                "previous_event_ref": previous_event,
                "created_at_ns": time.time_ns(),
                "elapsed_sec": time.monotonic() - started,
                **details,
            }
            previous_event = await asyncio.to_thread(
                append_record, root, directory, f"event_{event_count:04d}.json", event
            )
            if progress is not None:
                try:
                    await progress(event)
                except (RuntimeError, OSError, TypeError, ValueError):
                    logger.exception(
                        "Execution progress display detached; execution remains owned by the runtime."
                    )

        def check_stop() -> None:
            if stop.is_set():
                raise RuntimeError("Stop execution was requested.")

        status, message = "blocked", "Execution preparation did not complete."
        try:
            await emit("preparing", "Loading the saved primitive parameters and trajectories.")
            check_stop()
            evidence = {
                role: await asyncio.to_thread(verify_record, root, reference)
                for role, reference in program.report["evidence_refs"].items()
            }
            part = evidence["part"]
            from ...tools.exact_ref_resolver import approved_cad_path, _load_sources

            cad = _load_sources()[part["cad_context_ref"]]
            cad_path = approved_cad_path(part["cad_context_ref"])
            if hashlib.sha256(cad_path.read_bytes()).hexdigest() != cad["source_sha256"]:
                raise ValueError("The accepted CAD source changed.")
            if cad["units"] not in {"mm", "m"}:
                raise ValueError("Unsupported accepted CAD units.")
            if self.share is None:
                from ament_index_python.packages import get_package_share_directory

                share = Path(get_package_share_directory(self.profile["package"]))
            else:
                share = self.share
            instances = await asyncio.to_thread(
                fixture_instances, share, self.profile["world_file"], cad_path
            )
            checked = {item["step_index"]: item for item in program.report["checked_steps"]}
            placement_parent = (self.profile["placement_attachment"]
                                if program.report["scope"] == GAZEBO_LINK_ATTACHER_SCOPE and "goal" in evidence else None)
            # Compose stores raw trajectories. Apply the saved duration scale
            # once; do not plan or repeat Compose's joint-limit checks here.
            trajectories = {}
            for index, step in enumerate(program.steps, 1):
                if step["primitive_symbol"] != "move_cartesian":
                    continue
                trajectory = deepcopy(checked[index]["trajectory"])
                speed = checked[index]["resolved_params"].get(
                    "speed", program.robot["policy"]["trajectory_time_scale"]
                )
                trajectory["time_from_start_ns"] = [
                    int(stamp * speed) for stamp in trajectory["time_from_start_ns"]
                ]
                for field, divisor in (("velocities", speed), ("accelerations", speed * speed)):
                    trajectory[field] = [[number / divisor for number in row] for row in trajectory[field]]
                trajectories[index] = trajectory
            calculations = {
                index: (await asyncio.to_thread(verify_record, root, checked[index]["calculation_ref"]))["result"]
                for index, step in enumerate(program.steps, 1)
                if step["primitive_symbol"] in {"compute_pick_targets", "compute_place_targets"}
            }
            parameters = {
                index: deepcopy(checked[index]["resolved_params"])
                if "resolved_params" in checked[index] else resolve_selected_values(
                    step["params"],
                    read_evidence=lambda ref, pointer: _evidence_value(program.inputs, ref, pointer),
                    results=calculations,
                )
                for index, step in enumerate(program.steps, 1)
            }
            check_stop()
            await record(
                "prepared.json",
                {
                    "record_type": "PrimitiveExecutionPreparation",
                    "validation_ref": program.validation_ref,
                    "robot_context_ref": program.report["robot_context_ref"],
                    "trajectories": {str(index): value for index, value in trajectories.items()},
                    "preparation_timings_sec": dict(preparation_timings),
                    "created_at_ns": time.time_ns(),
                },
            )
            stage_started = time.monotonic()
            async with self.session_factory(configuration, transport_profile, stop) as transport:
                states = await transport.entity_states([item["model_name"] for item in instances])
                binding = match_instance(
                    instances,
                    states,
                    part,
                    cad_scale=0.001 if cad["units"] == "mm" else 1.0,
                    profile=self.profile,
                    interaction_root=root, validation_scope=program.report["scope"],
                )
                if placement_parent is not None:
                    if placement_parent["model_name"] == binding["model_name"]:
                        raise ValueError("The placement fixture cannot be the selected moving part.")
                    parents = await transport.entity_states([placement_parent["model_name"]])
                    if placement_parent["model_name"] not in parents:
                        raise ValueError("The configured placement fixture is unavailable in Gazebo.")
                binding_ref = await record(
                    "binding.json",
                    {
                        "record_type": "GazeboInstanceBinding",
                        "part_ref": program.report["evidence_refs"]["part"],
                        **binding,
                    },
                )
                preparation_timings["transport_and_binding_sec"] = time.monotonic() - stage_started
                previous_step_finished = None
                for index, step in enumerate(program.steps, 1):
                    step_started = time.monotonic()
                    symbol = step["primitive_symbol"]
                    check_stop()
                    params = deepcopy(parameters[index])
                    await emit(
                        "running",
                        f"Running step {index} of {len(program.steps)}: {symbol}",
                        step_index=index,
                        primitive_symbol=symbol,
                    )
                    await record(
                        f"step_{index:04d}_request.json",
                        {
                            "record_type": "PrimitiveExecutionStepRequest",
                            "step_index": index,
                            "primitive_symbol": symbol,
                            "resolved_params": params,
                            "binding_ref": binding_ref
                            if symbol in {"grasp_part", "release_part"}
                            else None,
                            "created_at_ns": time.time_ns(),
                            "preflight_elapsed_sec": time.monotonic() - step_started,
                            "preflight_feedback_elapsed_sec": None,
                            "idle_before_step_sec": (
                                time.monotonic() - previous_step_finished
                                if previous_step_finished is not None else None
                            ),
                        },
                    )
                    check_stop()
                    command_started_at_ns, command_started = time.time_ns(), time.monotonic()
                    motion_elapsed = completion_feedback_elapsed = None
                    if symbol in {"compute_pick_targets", "compute_place_targets"}:
                        output = deepcopy(calculations[index])
                    elif symbol == "move_cartesian":
                        command_dispatched = motion_dispatched = True
                        output = await transport.move(trajectories[index])
                        motion_elapsed = time.monotonic() - command_started
                        if output.get("success") is not True:
                            raise RuntimeError("Trajectory completion was not acknowledged.")
                    elif symbol == "grasp_part":
                        if held is not None:
                            raise ValueError(f"grasp_part requires held_part to be null; acknowledged held_part is {held!r}.")
                        command_dispatched = True
                        position = params.get("position", configuration["gripper"]["close"])
                        gripper_request = await record(
                            f"step_{index:04d}_gripper_request.json",
                            {"record_type": "PrimitiveExecutionGripperRequest", "step_index": index,
                             "joint": configuration["gripper"]["joint"], "target": position,
                             "created_at_ns": time.time_ns()},
                        )
                        gripper_state = None
                        grip = await transport.gripper_command(position)
                        await record(f"step_{index:04d}_gripper_result.json", {
                            "record_type": "PrimitiveExecutionGripperResult",
                            "request_ref": gripper_request, **grip,
                        })
                        gripper_request = None
                        if grip.get("success") is not True:
                            raise RuntimeError("Gripper completion was not acknowledged.")
                        gripper_state = "closed"
                        check_stop()
                        custody_known = False
                        output = await transport.attachment(binding, True)
                        if output.get("success") is not True or output.get("attached") is not True:
                            raise RuntimeError("Gazebo attachment was not acknowledged.")
                        output["gripper_feedback"] = grip
                        held = params["part_name"]
                        custody_known = True
                    else:
                        if held is None or params.get("part_name", held) != held:
                            raise ValueError("release_part does not match acknowledged held_part.")
                        command_dispatched = True
                        gripper_request = await record(
                            f"step_{index:04d}_gripper_request.json",
                            {"record_type": "PrimitiveExecutionGripperRequest", "step_index": index,
                             "joint": configuration["gripper"]["joint"],
                             "target": configuration["gripper"]["open"], "created_at_ns": time.time_ns()},
                        )
                        gripper_state = None
                        grip = await transport.gripper_command(configuration["gripper"]["open"])
                        await record(f"step_{index:04d}_gripper_result.json", {
                            "record_type": "PrimitiveExecutionGripperResult",
                            "request_ref": gripper_request, **grip,
                        })
                        gripper_request = None
                        if grip.get("success") is not True:
                            raise RuntimeError("Gripper completion was not acknowledged.")
                        gripper_state = "open"
                        check_stop()
                        custody_known = False
                        output = await transport.attachment(binding, False)
                        if output.get("success") is not True or output.get("attached") is not False:
                            raise RuntimeError("Gazebo detachment was not acknowledged.")
                        if placement_parent is not None:
                            check_stop()
                            # Freeze the pose reached by the authored primitives;
                            # a placement snap would conceal a wrong target.
                            placed = await transport.attachment(binding, True, parent=placement_parent)
                            if (placed.get("success") is not True or placed.get("attached") is not True
                                    or placed.get("parent") != placement_parent):
                                raise RuntimeError("Gazebo board attachment was not acknowledged.")
                            output["placement_attachment"] = placed
                        output["gripper_feedback"] = grip
                        held, custody_known = None, True
                    previous_step_finished = time.monotonic()
                    await record(
                        f"step_{index:04d}_result.json",
                        {
                            "record_type": "PrimitiveExecutionStepResult",
                            "step_index": index,
                            "primitive_symbol": symbol,
                            "outputs": output,
                            "held_part": held,
                            "custody_known": custody_known,
                            "command_started_at_ns": command_started_at_ns,
                            "created_at_ns": time.time_ns(),
                            "command_elapsed_sec": previous_step_finished - command_started,
                            "motion_elapsed_sec": motion_elapsed,
                            "completion_feedback_elapsed_sec": completion_feedback_elapsed,
                        },
                    )
                    completed = index
                check_stop()
                status, message = (
                    "completed",
                    "Primitive program completed with acknowledged gripper, detach and board-attachment commands. Physical fit was not evaluated."
                    if placement_parent is not None else
                    "All commands acknowledged. Shaft fitting was not rechecked during execution; post-release seating remains unobserved."
                    if is_observed_scope(program.report["scope"]) and "goal" in evidence else
                    "Pick-and-place completed. All motion, gripper and attach/detach commands acknowledged."
                    if is_pick_place_scope(program.report["scope"]) else
                    "Execution completed. All commands acknowledged; assembly success has not been established.",
                )
        except asyncio.CancelledError:
            stop.set()
            status, message = (
                "interrupted",
                "Execution was interrupted; no automatic resumption is permitted.",
            )
            custody_known = False if command_dispatched else custody_known
        except (ImportError, OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
            status = "stopped" if stop.is_set() else "failed" if command_dispatched else "blocked"
            message = (f"Step {index} ({symbol}): " if command_dispatched else "") + f"{type(exc).__name__}: {exc}"
            if isinstance(exc, TimeoutError) and command_dispatched:
                custody_known = False
                status = "unknown"
            if isinstance(exc, GripperCommandError):
                custody_known = exc.outcome_known
                if not custody_known:
                    status = "unknown"
                await record(f"step_{index:04d}_gripper_result.json", {
                    "record_type": "PrimitiveExecutionGripperResult",
                    "request_ref": gripper_request, **exc.diagnostics,
                })
        await emit(status, message, completed_steps=completed)
        result = {
            "record_type": "PrimitiveExecutionResult",
            "request_ref": request_ref,
            "status": status,
            "message": message,
            "completed_steps": completed,
            "command_dispatched": command_dispatched,
            "motion_dispatched": motion_dispatched,
            "held_part": held,
            "gripper_state": gripper_state,
            "custody_known": custody_known,
            "last_event_ref": previous_event,
            "record_refs": refs,
            "observation_ref": None,
            "assembly_success": None,
            "created_at_ns": time.time_ns(),
            "elapsed_sec": time.monotonic() - started,
            "preparation_timings_sec": preparation_timings,
        }
        append_record(root, directory, "result.json", result)
        return result
