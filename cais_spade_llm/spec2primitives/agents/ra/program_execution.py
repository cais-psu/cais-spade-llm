from __future__ import annotations

"""Run the exact validated program through an owned simulation-only transport."""

import asyncio
import fcntl
import hashlib
import logging
import threading
import time
import uuid
import time
from contextlib import AsyncExitStack
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .execution_state import (
    ACTIVE,
    assert_execution_available,
    execution_busy,
    execution_custody,
    read_primitive_execution_diagnostic,
)
from .primitive_composition import (
    _assert_inputs_unchanged,
    _evidence_value,
    _load_inputs,
    _read_record,
    _recorded_request_inputs,
    read_primitive_composition_diagnostic,
)
from .program_validation import resolve_selected_values, validate_program
from .validation_scope import is_pick_place_scope, read_validation_scope, supported_primitive_symbols
from .refinement import _ACTIVE as ACTIVE_COMPOSITIONS
from .refinement import _robot_changed
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
    fixture_instances,
    load_execution_profile,
    match_instance,
    prepare_trajectory,
)
from ...adapters.target_calculation import calculate_target
from ...adapters.robot_validation_context import validation_capture

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
    view = read_primitive_composition_diagnostic(root)
    if view["status"] != "validated_for_declared_scope":
        raise ValueError(
            "Run in Gazebo requires the current program to be validated_for_declared_scope."
        )
    result = view["refinement"]["result"]
    request_ref = result["request_ref"]
    request = verify_record(root, request_ref)
    scope = read_validation_scope(request["profile"])
    result_ref = pin(root, owned_path(root, request_ref["ref"]).parent / "result.json")
    if (
        verify_record(root, result_ref) != result
        or request["base_context_refs"] != inputs.context_refs
    ):
        raise ValueError("The validated run has changed authority.")
    candidate_ref, validation_ref = result["candidate_refs"][-1], result["validation_refs"][-1]
    candidate = _read_record(owned_path(root, candidate_ref["ref"]))
    if read_pin(root, candidate_ref) != candidate or candidate != view["candidate"]:
        raise ValueError("The displayed candidate differs from the validated program.")
    saved_request = _read_record(owned_path(root, candidate["request_ref"]))
    read_pin(root, {"ref": candidate["request_ref"], "sha256": candidate["request_sha256"]})
    if read_validation_scope(saved_request) != scope:
        raise ValueError("The authored program and refinement run have different validation scopes.")
    inputs = _recorded_request_inputs(inputs, saved_request)
    if inputs.includes_execution_identifiers:
        raise ValueError(
            "Compose a new program with the current composition interface before execution."
        )
    report = verify_record(root, validation_ref)
    if report != view.get("validation"):
        raise ValueError("The selected validation report differs from the displayed report.")
    binding_ref = result.get("binding_refs", [None])[-1] if result.get("binding_refs") else None
    if report.get("binding_ref") != binding_ref or view.get("binding_ref") != binding_ref:
        raise ValueError("The displayed binding differs from the validated program.")
    if binding_ref is not None:
        from .program_binding import read_program_binding

        binding, inputs = read_program_binding(_load_inputs(root), binding_ref)
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
    ) -> None:
        """Inject owned robot authority, transport, and final RGB-D capture."""
        self.robot_runtime, self.capture_runtime = robot_runtime, capture_runtime
        self.profile = dict(profile) if profile is not None else load_execution_profile()
        self.session_factory, self.validator, self.share = session_factory, validator, share

    async def run(
        self,
        root: Path,
        *,
        candidate_ref: str | None = None,
        binding_ref: str | None = None,
        progress: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Join duplicate clicks; permit only one owned execution across interactions."""
        root = root.resolve()
        session = ACTIVE.get(root)
        if session is not None and not session.task.done():
            return await asyncio.shield(session.task)
        if execution_busy() or any(not task.done() for task in tuple(ACTIVE_COMPOSITIONS.values())):
            raise ValueError("Another composition or Gazebo execution is active.")
        stop = threading.Event()
        task = asyncio.create_task(self._run(root, stop, progress, candidate_ref, binding_ref))
        ACTIVE[root] = _Session(task, stop)
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
        await asyncio.to_thread(assert_execution_available, root.parent)
        program = await asyncio.to_thread(load_validated_program, root)
        if binding_ref is not None and (program.binding_ref or {}).get("ref") != binding_ref:
            raise ValueError("The displayed parameter binding changed. Refresh before Run in Gazebo.")
        if candidate_ref is not None and candidate_ref != program.candidate_ref["ref"]:
            raise ValueError(
                "The displayed program changed. Refresh the program before Run in Gazebo."
            )
        assignment = program.inputs.assignment
        if assignment.selected_execution_mode != "simulation":
            raise ValueError("Run in Gazebo supports simulation only.")
        custody = await asyncio.to_thread(
            execution_custody, root.parent, assignment.selected_resource_jid
        )
        previous_view = await asyncio.to_thread(read_primitive_execution_diagnostic, root)
        if previous_view.get("candidate_ref") == program.candidate_ref and (
            previous_view.get("result") or {}
        ).get("command_dispatched", True):
            raise ValueError(
                "This program already has an execution attempt. Capture current context and compose a fresh program; saved commands are never replayed."
            )
        authority = await self.robot_runtime.execution_configuration(assignment)
        configuration = authority["configuration"]
        transport_profile = {
            **self.profile,
            "joint_states_topic": program.profile["resources"][assignment.selected_resource_jid][
                "joint_states_topic"
            ],
            "state_max_age_sec": program.profile["state_max_age_sec"],
            "fk_orientation_tolerance_rad": program.profile["fk_orientation_tolerance_rad"],
        }
        recorded_catalog = read_pin(root, program.inputs.context_refs["primitive_catalog"])[
            "primitive_catalog"
        ]
        if authority["primitive_catalog"] != recorded_catalog:
            raise ValueError("The live selected RobotAgent catalog changed after composition.")
        directory = root / "execution" / ("run_" + str(time.time_ns()) + "_" + uuid.uuid4().hex[:8])
        refs = []
        previous_event = None
        event_count = 0
        completed = 0
        motion_dispatched = False
        command_dispatched = False
        custody_known = True
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

        async def check_authority() -> None:
            check_stop()
            await asyncio.to_thread(_assert_inputs_unchanged, program.inputs)
            await asyncio.to_thread(read_pin, root, program.candidate_ref)
            if program.binding_ref is not None:
                await asyncio.to_thread(read_pin, root, program.binding_ref)
            await asyncio.to_thread(verify_record, root, program.validation_ref)
            for reference in program.report["evidence_refs"].values():
                await asyncio.to_thread(verify_evidence_tree, root, reference)
            current = await self.robot_runtime.execution_configuration(assignment)
            if current != authority:
                raise ValueError(
                    "The selected robot configuration or catalog changed during execution."
                )

        status, message = "blocked", "Execution preparation did not complete."
        try:
            await emit(
                "preparing", "Checking the saved program, current robot state and Gazebo binding."
            )
            check_stop()
            async with AsyncExitStack() as capture_stack:
                fresh = dict(
                    await capture_stack.enter_async_context(validation_capture(
                        self.robot_runtime, assignment, profile=program.profile, custody={"held_part": held},
                    ))
                )
                validation_entry_ns = time.time_ns()
                fresh_ref = await record("robot_initial.json", fresh)
                if fingerprint(configuration) != fresh["configuration_sha256"] or _robot_changed(
                    program.robot, fresh, program.profile
                ):
                    raise ValueError(
                        "Robot state or configuration changed after validation. Capture current context and compose again."
                    )
                fresh_report = await self.validator(
                    inputs=program.inputs,
                    steps=program.steps,
                    robot=fresh,
                    evidence=program.report["evidence_refs"],
                    directory=directory / "validation",
                    profile=program.profile,
                    cache={},
                    _validation_started_at_ns=validation_entry_ns,
                )
            validation_ref = await record("validation.json", fresh_report)
            if fresh_report["status"] != "passed" or fresh_report[
                "candidate_fingerprint"
            ] != fingerprint(program.steps) or fresh_report.get("scope") != program.report["scope"] or (
                fresh_report.get("binding_ref") != program.binding_ref
            ):
                raise ValueError("Fresh validation did not pass for the unchanged program.")
            final = dict(
                await self.robot_runtime.capture_execution_context(
                    assignment, profile=program.profile, custody={"held_part": held}
                )
            )
            await record("robot_ready.json", final)
            if _robot_changed(fresh, final, program.profile):
                raise ValueError("Robot state changed during execution preparation.")
            evidence = {
                role: await asyncio.to_thread(verify_evidence_tree, root, reference)
                for role, reference in program.report["evidence_refs"].items()
            }
            for value in evidence.values():
                if value.get("authority") == "explicit_experiment_specification":
                    specification = owned_path(
                        Path(__file__).resolve().parents[2], value["source_path"]
                    )
                    if (
                        hashlib.sha256(specification.read_bytes()).hexdigest()
                        != value["source_sha256"]
                    ):
                        raise ValueError(
                            "The explicit experiment specification changed after validation."
                        )
            self._check_scene_age(evidence, final["measured_at_ros_ns"], program.profile)
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
            checked = {item["step_index"]: item for item in fresh_report["checked_steps"]}
            trajectories = {
                index: prepare_trajectory(
                    checked[index]["trajectory"],
                    fresh,
                    checked[index]["resolved_params"].get(
                        "speed", fresh["policy"]["trajectory_time_scale"]
                    ),
                )
                for index, step in enumerate(program.steps, 1)
                if step["primitive_symbol"] == "move_cartesian"
            }
            await record(
                "prepared.json",
                {
                    "record_type": "PrimitiveExecutionPreparation",
                    "validation_ref": validation_ref,
                    "robot_context_ref": fresh_ref,
                    "trajectories": {str(index): value for index, value in trajectories.items()},
                },
            )
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
                binding_ref = await record(
                    "binding.json",
                    {
                        "record_type": "GazeboInstanceBinding",
                        "part_ref": program.report["evidence_refs"]["part"],
                        **binding,
                    },
                )
                pose = deepcopy(fresh["ee_pose"])
                joints = deepcopy(fresh["joint_state"])
                gripper_joint = configuration["gripper"]["joint"]
                if gripper_joint not in joints["names"]:
                    raise ValueError("The configured gripper has no measured joint feedback.")
                gripper_position = joints["positions"][joints["names"].index(gripper_joint)]
                results: dict[int, Mapping[str, Any]] = {}
                for index, step in enumerate(program.steps, 1):
                    await check_authority()
                    self._verify_binding_sources(binding)
                    feedback = await transport.feedback(
                        fresh, joints, pose, program.profile["state_change_joint_tolerance_rad"]
                    )
                    self._check_scene_age(evidence, feedback["measured_at_ros_ns"], program.profile)
                    params = resolve_selected_values(
                        step["params"],
                        read_evidence=lambda ref, pointer: _evidence_value(
                            program.inputs, ref, pointer
                        ),
                        results=results,
                    )
                    symbol = step["primitive_symbol"]
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
                        },
                    )
                    check_stop()
                    if symbol in {"compute_pick_targets", "compute_place_targets"}:
                        output = calculate_target(symbol, params, fresh, pose, validation_scope=program.report["scope"])
                        calculation = verify_record(root, checked[index]["calculation_ref"])
                        if output != calculation["result"]:
                            raise ValueError(
                                "Helper outputs differ from the freshly validated program."
                            )
                    elif symbol == "move_cartesian":
                        command_dispatched = motion_dispatched = True
                        output = await transport.move(trajectories[index])
                        pose = {
                            **pose,
                            **{
                                key: params[key]
                                for key in ("x", "y", "z", "qx", "qy", "qz", "qw")
                                if key in params
                            },
                        }
                        # Mimic joints follow the gripper controller. Retain its
                        # measured primary joint alongside the checked arm joints.
                        names = trajectories[index]["joint_names"]
                        joints = {
                            "names": names + [gripper_joint],
                            "positions": trajectories[index]["positions"][-1] + [gripper_position],
                        }
                        output["feedback"] = await transport.feedback(
                            fresh, joints, pose, program.profile["state_change_joint_tolerance_rad"]
                        )
                    elif symbol == "grasp_part":
                        states = await transport.entity_states([binding["model_name"]])
                        match_instance(
                            [binding],
                            states,
                            part,
                            cad_scale=0.001 if cad["units"] == "mm" else 1.0,
                            profile=self.profile,
                            interaction_root=root, validation_scope=program.report["scope"],
                        )
                        if held is not None:
                            raise ValueError("grasp_part requires held_part to be null.")
                        custody_known = False
                        command_dispatched = True
                        position = params.get("position", configuration["gripper"]["close"])
                        grip = await transport.gripper_command(position)
                        gripper_position = position
                        joints["positions"][joints["names"].index(gripper_joint)] = position
                        gripper_state = "closed"
                        check_stop()
                        output = await transport.attachment(binding, True)
                        if output.get("success") is not True or output.get("attached") is not True:
                            raise RuntimeError("Gazebo attachment was not acknowledged.")
                        output["gripper_feedback"] = grip
                        held = params["part_name"]
                        custody_known = True
                    else:
                        if held is None or params.get("part_name", held) != held:
                            raise ValueError("release_part does not match acknowledged held_part.")
                        custody_known = False
                        command_dispatched = True
                        grip = await transport.gripper_command(configuration["gripper"]["open"])
                        gripper_position = configuration["gripper"]["open"]
                        joints["positions"][joints["names"].index(gripper_joint)] = gripper_position
                        gripper_state = "open"
                        check_stop()
                        output = await transport.attachment(binding, False)
                        if output.get("success") is not True or output.get("attached") is not False:
                            raise RuntimeError("Gazebo detachment was not acknowledged.")
                        output["gripper_feedback"] = grip
                        held, custody_known = None, True
                    results[index] = deepcopy(output)
                    await record(
                        f"step_{index:04d}_result.json",
                        {
                            "record_type": "PrimitiveExecutionStepResult",
                            "step_index": index,
                            "primitive_symbol": symbol,
                            "outputs": output,
                            "held_part": held,
                            "custody_known": custody_known,
                        },
                    )
                    completed = index
                check_stop()
                status, message = (
                    "completed",
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
            message = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, TimeoutError) and command_dispatched:
                custody_known = False
                status = "unknown"
        observation = None
        if status == "completed" and self.capture_runtime is not None:
            await emit("capturing", "Commands completed; capturing final RGB-D for inspection.")
            try:
                path = await asyncio.to_thread(
                    self.capture_runtime.capture,
                    directory / "observations",
                    "execution_" + directory.name,
                    timeout_sec=self.profile["capture_timeout_sec"],
                )
                observation = str(path.relative_to(root))
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                message += f" Final RGB-D capture unavailable: {exc}"
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
            "observation_ref": observation,
            "assembly_success": None,
            "created_at_ns": time.time_ns(),
        }
        append_record(root, directory, "result.json", result)
        return result

    @staticmethod
    def _check_scene_age(evidence: Mapping[str, Any], now: int, profile: Mapping[str, Any]) -> None:
        for value in evidence.values():
            stamp = value.get("observation_timestamp_ns")
            if stamp is not None and not 0 <= now - stamp <= profile["scene_max_age_sec"] * 1e9:
                raise ValueError(
                    "Observed scene evidence is stale; no subsequent primitive can execute."
                )

    @staticmethod
    def _verify_binding_sources(binding: Mapping[str, Any]) -> None:
        for source in binding["sources"]:
            if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("Gazebo fixture or mesh changed after instance binding.")
