from __future__ import annotations

"""Bound evidence acquisition, audited calculation, validation and RA revision."""

import asyncio
import hashlib
import json
import time
from contextlib import AsyncExitStack, asynccontextmanager
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .primitive_composition import (
    _assert_inputs_unchanged,
    _evidence_value,
    _load_inputs,
    _result_schema,
    _with_refinement,
    _with_scope,
    author_primitive_program_candidate,
)
from .program_dependencies import assess_program_dependencies
from .program_binding import apply_primitive_bindings
from .program_validation import validate_program
from ...adapters.robot_validation_context import validation_capture
from .validation_scope import is_pick_place_scope, read_validation_scope, required_validation_roles
from .refinement_records import (
    append_record,
    fingerprint,
    owned_path,
    pin,
    read_pin,
    verify_evidence_tree,
    verify_record,
)

_RUNS = Path("composition/refinement_runs")
_ACTIVE: dict[Path, asyncio.Task[dict[str, Any]]] = {}
_PROFILE = Path(__file__).resolve().parents[2] / "config/phase5_validation.json"


@asynccontextmanager
async def _deadline(seconds: float):
    """Bound this task on the project's Python 3.10 runtime."""
    task = asyncio.current_task()
    expired = False

    def expire() -> None:
        nonlocal expired
        expired = True
        task.cancel()

    timer = asyncio.get_running_loop().call_later(seconds, expire)
    try:
        yield
    except asyncio.CancelledError:
        if expired:
            raise TimeoutError("Refinement deadline reached.") from None
        raise
    finally:
        timer.cancel()


def load_refinement_profile() -> dict[str, Any]:
    """Read finite owned budgets and sensor/planner policies, never goal coordinates."""
    profile = json.loads(_PROFILE.read_bytes())
    read_validation_scope(profile)
    for field in ("max_candidates", "max_pa_batches", "max_pa_operations"):
        if type(profile.get(field)) is not int or not 1 <= profile[field] <= 32:
            raise ValueError(f"Invalid refinement budget: {field}.")
    for field in (
        "deadline_sec",
        "service_timeout_sec",
        "planning_timeout_sec",
        "state_max_age_sec",
        "max_capture_skew_sec",
        "scene_max_age_sec",
        "cartesian_max_step_m",
    ):
        if type(profile.get(field)) not in (int, float) or not 0 < profile[field] <= 3600:
            raise ValueError(f"Invalid refinement time/resolution policy: {field}.")
    return profile


def _robot_changed(
    previous: Mapping[str, Any], current: Mapping[str, Any], profile: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Describe differences under the existing robot-context comparison policy."""
    differences = []
    for key in (
        "configuration_sha256",
        "model_parameters_sha256",
        "frame_id",
        "ee_link",
        "tcp_link",
        "held_part",
    ):
        if previous.get(key) != current.get(key):
            differences.append(
                {"field": key, "previous": previous.get(key), "current": current.get(key)}
            )
    first_transform = np.asarray(previous["ee_from_tcp"], dtype=float)
    second_transform = np.asarray(current["ee_from_tcp"], dtype=float)
    for row, column in np.argwhere(
        ~np.isclose(first_transform, second_transform, rtol=0, atol=1e-6)
    ):
        first_value = float(first_transform[row, column])
        second_value = float(second_transform[row, column])
        differences.append(
            {
                "field": f"ee_from_tcp[{row}][{column}]",
                "previous": first_value,
                "current": second_value,
                "difference": abs(first_value - second_value),
                "tolerance": 1e-6,
            }
        )
    first = dict(
        zip(previous["joint_state"]["names"], previous["joint_state"]["positions"], strict=True)
    )
    second = dict(
        zip(current["joint_state"]["names"], current["joint_state"]["positions"], strict=True)
    )
    if first.keys() != second.keys():
        differences.append(
            {
                "field": "joint_state.names",
                "previous": previous["joint_state"]["names"],
                "current": current["joint_state"]["names"],
                "added": [key for key in second if key not in first],
                "missing": [key for key in first if key not in second],
            }
        )
    for key in first:
        if key in second:
            difference = abs(first[key] - second[key])
            if difference > profile["state_change_joint_tolerance_rad"]:
                differences.append(
                    {
                        "field": f"joint_state.positions[{key!r}]",
                        "previous": first[key],
                        "current": second[key],
                        "difference": difference,
                        "tolerance": profile["state_change_joint_tolerance_rad"],
                    }
                )
    return differences


def _robot_change_message(differences: list[dict[str, Any]]) -> str:
    descriptions = []
    for item in differences[:2]:
        field = item["field"]
        if "difference" in item:
            descriptions.append(
                f"{field}: {item['previous']:.9g} → {item['current']:.9g} "
                f"(difference {item['difference']:.3g}, threshold {item['tolerance']:.3g})"
            )
        elif field == "joint_state.names":
            descriptions.append(f"{field}: added {item['added']!r}, missing {item['missing']!r}")
        elif field.endswith("_sha256"):
            descriptions.append(f"{field} differs")
        else:
            descriptions.append(f"{field}: {item['previous']!r} → {item['current']!r}")
    return (
        "Robot context comparison detected differences: "
        + "; ".join(descriptions)
        + ". Open the program records for all differences and compared captures."
    )


def _experiment_specification(
    root: Path, directory: Path, profile: Mapping[str, Any]
) -> tuple[dict[str, str] | None, Path | None, str | None]:
    if is_pick_place_scope(read_validation_scope(profile)):
        return None, None, None
    configured = profile.get("validation_specification_path")
    if configured is None:
        return None, None, None
    path = (Path(__file__).resolve().parents[2] / configured).resolve()
    cases = Path(__file__).resolve().parents[2] / "cases"
    if not path.is_relative_to(cases) or path.suffix != ".json" or path.is_symlink():
        raise ValueError(
            "Explicit experiment specifications must be JSON under Spec2Primitives/cases."
        )
    source = json.loads(path.read_bytes())
    if (
        source.get("record_type") != "AssemblyValidationSpecification"
        or source.get("status") != "accepted"
    ):
        raise ValueError("The configured experiment specification is invalid.")
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    reference = append_record(
        root,
        directory,
        "experiment_specification.json",
        {
            **source,
            "authority": "explicit_experiment_specification",
            "source_path": configured,
            "source_sha256": source_hash,
        },
    )
    return reference, path, source_hash


class PrimitiveRefinementRuntime:
    """Coordinate the owned PA/RA adapters while RA retains program authorship."""

    def __init__(
        self,
        *,
        program_runtime: Any,
        robot_runtime: Any,
        product_runtime: Any = None,
        profile: Mapping[str, Any] | None = None,
        validator: Callable[..., Awaitable[dict[str, Any]]] = validate_program,
    ) -> None:
        """Inject runtime authorities and an optional controlled validation boundary."""
        self.program_runtime, self.robot_runtime, self.product_runtime = (
            program_runtime,
            robot_runtime,
            product_runtime,
        )
        self.profile = deepcopy(dict(profile)) if profile is not None else load_refinement_profile()
        self.validator = validator

    async def compose(
        self, root: Path, *, progress: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
        deadline_sec: float | None = None,
    ) -> dict[str, Any]:
        """Start or join a run with an optional deadline applying only to a new run.

        Args:
            root: Saved interaction to compose against.
            progress: Optional persisted-event consumer.
            deadline_sec: Finite per-run deadline; omitted uses the configured profile.

        Returns:
            The recorded result of the new or already active run.
        """
        if deadline_sec is not None and (
            type(deadline_sec) not in (int, float) or not 0 < deadline_sec <= 3600
        ):
            raise ValueError("The composition deadline must be finite and between 0 and 3600 seconds.")
        root = root.resolve()
        task = _ACTIVE.get(root)
        if task is None or task.done():
            profile = deepcopy(self.profile)
            if deadline_sec is not None:
                profile["deadline_sec"] = deadline_sec
            task = asyncio.create_task(self._run(root, progress=progress, profile=profile))
            _ACTIVE[root] = task
            task.add_done_callback(
                lambda done: _ACTIVE.pop(root, None) if _ACTIVE.get(root) is done else None
            )
        return await asyncio.shield(task)

    async def _run(
        self, root: Path, *, progress: Callable[[Mapping[str, Any]], Awaitable[None]] | None,
        profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        inputs = await asyncio.to_thread(_load_inputs, root)
        scope = read_validation_scope(profile)
        inputs = _with_scope(inputs, scope)
        parent = root / _RUNS
        parent.mkdir(parents=True, exist_ok=True)
        directory = parent / f"run_{len(list(parent.glob('run_*'))) + 1:04d}"
        directory.mkdir(exist_ok=False)
        request_ref = append_record(
            root,
            directory,
            "request.json",
            {
                "record_type": "PrimitiveRefinementRequest",
                "base_context_refs": inputs.context_refs,
                "profile": profile,
                "created_at_ns": time.time_ns(),
            },
        )
        events: list[dict[str, str]] = []
        candidate_refs: list[dict[str, str]] = []
        decisions: list[dict[str, str]] = []
        reports: list[dict[str, str]] = []
        binding_refs: list[dict[str, str]] = []
        source_pins: dict[str, dict[str, str]] = {}
        validation_refs: dict[str, dict[str, str]] = {}
        pa_findings: list[dict[str, Any]] = []
        pa_answer_refs: list[dict[str, str]] = []
        robot_ref = None
        robot = None
        cache: dict[str, dict[str, Any]] = {}
        pa_batches = pa_operations = 0
        stop_reason = "The refinement budget was exhausted."
        status = "budget_exhausted"
        started = time.monotonic()
        event_lock = asyncio.Lock()
        current_stage = "composing"

        async def emit(stage: str, message: str, **details: Any) -> None:
            nonlocal current_stage
            current_stage = stage
            # Concurrent PA measurements publish into one append-only event chain.
            async with event_lock:
                record = {
                    "record_type": "PrimitiveRefinementEvent",
                    "stage": stage,
                    "message": message,
                    "elapsed_sec": time.monotonic() - started,
                    "created_at_ns": time.time_ns(),
                    **deepcopy(details),
                }
                publication = asyncio.create_task(asyncio.to_thread(
                    append_record, root, directory, f"event_{len(events) + 1:04d}.json", record
                ))
                cancelled = False
                try:
                    reference = await asyncio.shield(publication)
                except asyncio.CancelledError:
                    cancelled = True
                    reference = await publication
                events.append(reference)
                if cancelled:
                    raise asyncio.CancelledError
                if progress is not None:
                    await progress(record)

        async def composition_progress(message: str) -> None:
            await emit("composing", message)

        async def validation_progress(event: Mapping[str, Any]) -> None:
            details: dict[str, Any] = {"step_index": event["step_index"]}
            if "calculation_ref" in event:
                from .program_binding import display_primitive_steps

                completed_calculations.append(event["calculation_ref"])
                displayed = await asyncio.to_thread(display_primitive_steps, extended, steps,
                                                    calculation_refs=completed_calculations)
                details.update(calculation_ref=event["calculation_ref"], candidate_ref=candidate_ref,
                               binding_ref=binding_ref, resolved_primitive_steps=displayed)
            await emit(event["stage"], event["message"], **details)

        try:
            async with _deadline(float(profile["deadline_sec"])):
                await emit(
                    "composing", f"Composition started with a {profile['deadline_sec']:g}-second deadline.",
                    deadline_sec=profile["deadline_sec"],
                )
                specification_ref, specification_path, specification_hash = await asyncio.to_thread(
                    _experiment_specification, root, directory, profile
                )
                if specification_ref:
                    validation_refs["specification"] = specification_ref
                    source_pins[specification_ref["ref"]] = specification_ref
                refinement_ref = None
                seen: set[str] = set()
                for _ in range(profile["max_candidates"]):
                    await asyncio.to_thread(_assert_inputs_unchanged, inputs)
                    await emit("composing", "RA is authoring the initial proposal." if not candidate_refs
                               else "RA is revising primitive or motion decisions using validation findings.")
                    candidate = await author_primitive_program_candidate(
                        self.program_runtime, root, refinement_ref=refinement_ref,
                        progress=composition_progress, validation_scope=scope,
                    )
                    candidate_ref = pin(root, candidate.path)
                    decisions.append(candidate_ref)
                    if candidate.record["status"] != "proposed":
                        status, stop_reason = candidate.record["status"], candidate.record["reason"]
                        break
                    candidate_refs.append(candidate_ref)
                    await emit("proposal", "RA's program proposal is recorded; deterministic input binding is pending.",
                               candidate_ref=candidate_ref, candidate=candidate.record, candidate_count=len(candidate_refs))
                    candidate_inputs = (await asyncio.to_thread(_with_refinement, inputs, refinement_ref)
                                        if refinement_ref else inputs)
                    extended, steps = candidate_inputs, deepcopy(candidate.record["primitive_steps"])
                    submission = read_pin(root, candidate.record["exchange_refs"][-1])["response"]["action"]
                    explicit_needs = deepcopy(submission.get("context_requests", []))
                    answers_by_batch: dict[int, dict[str, str]] = {}
                    binding_ref = None
                    completed_calculations = []
                    bound_answers = None
                    pa_findings = []

                    async def save_binding() -> None:
                        nonlocal steps, extended, binding_ref, bound_answers
                        from .program_binding import display_primitive_steps

                        answer_refs = list(answers_by_batch.values())
                        if binding_ref is not None and answer_refs == bound_answers:
                            return
                        steps, extended = await asyncio.to_thread(
                            apply_primitive_bindings, candidate_inputs, candidate_ref, answer_refs,
                        )
                        payload = {"record_type": "PrimitiveProgramBinding", "run_request_ref": request_ref,
                                   "candidate_ref": candidate_ref, "pa_answer_refs": deepcopy(answer_refs),
                                   "primitive_steps": steps, "created_at_ns": time.time_ns()}
                        binding_ref = await asyncio.to_thread(
                            append_record, root, directory, f"binding_{len(binding_refs) + 1:04d}.json", payload,
                        )
                        binding_refs.append(binding_ref)
                        bound_answers = deepcopy(answer_refs)
                        extended = replace(extended, binding_ref=binding_ref)
                        displayed = await asyncio.to_thread(display_primitive_steps, extended, steps)
                        await emit("binding", "Checked PA answers have been bound deterministically; calculations remain pending.",
                                   binding_ref=binding_ref, binding=payload, resolved_primitive_steps=displayed)

                    def input_needs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
                        dependencies = assess_program_dependencies(
                            steps, extended.catalog, extended.composition_input["robot_state"],
                            read_evidence=lambda ref, pointer: _evidence_value(extended, ref, pointer),
                            result_schema=lambda ref: _result_schema(steps, ref, extended),
                        )
                        needs = [need for need in dependencies["context_requests"] if need["authority"] == "PA"]
                        for role in required_validation_roles(scope):
                            evidence = (verify_evidence_tree(root, validation_refs[role])
                                        if role in validation_refs else {})
                            reason = None
                            if not evidence:
                                reason = f"Required {role} evidence is missing."
                            elif evidence.get("status") != "accepted":
                                reason = f"Required {role} evidence is {evidence.get('status', 'unavailable')}."
                            elif role == "scene" and (evidence.get("coverage") != "all_observed_candidates"
                                                       or evidence.get("unresolved_candidates")):
                                reason = "Required scene evidence does not cover all declared observed candidates."
                            if reason:
                                needs.append({"step_index": None, "quantity": role, "authority": "PA",
                                              "reason": reason})
                        return dependencies, needs

                    while True:
                        dependencies, product_needs = await asyncio.to_thread(input_needs)
                        unique = {(need["step_index"], need.get("parameter_path", need["quantity"])): need
                                  for need in product_needs}
                        for need in explicit_needs:
                            if need["authority"] == "PA":
                                unique.setdefault((need["step_index"], need["quantity"]), need)
                        product_needs = list(unique.values())
                        if (not product_needs or self.product_runtime is None
                                or pa_batches >= profile["max_pa_batches"] or pa_operations >= profile["max_pa_operations"]):
                            break
                        pa_batches += 1
                        batch_number = pa_batches
                        allowance = min(6, profile["max_pa_operations"] - pa_operations)
                        operations_before_batch = pa_operations
                        available = {ref: sha for ref, sha in inputs.record_hashes.items() if not ref.startswith("resources/")}
                        available.update({ref: source["sha256"] for ref, source in source_pins.items()
                                          if read_pin(root, source).get("record_type") not in {
                                              "RobotValidationContext", "PrimitiveCalculationRecord"}})
                        from ..pa.primitive_context import _answer_result, _number_needs, _read_answer_checkpoint

                        product_needs = _number_needs(product_needs)
                        batch_request = {
                            "record_type": "PrimitiveContextRequest", "run_request_ref": request_ref,
                            "candidate_ref": candidate_ref, "assignment_fingerprint": inputs.assignment.fingerprint,
                            "validation_scope": scope, "target_feature": deepcopy(inputs.composition_input["target_feature"]),
                            "needs": product_needs, "primitive_steps": deepcopy(candidate.record["primitive_steps"]),
                            "validation_refs": deepcopy(validation_refs),
                            "evidence_refs": [{"ref": ref, "sha256": sha} for ref, sha in available.items()],
                            "base_context_refs": inputs.context_refs,
                        }
                        batch_dir = directory / f"pa_{batch_number:04d}"
                        await asyncio.to_thread(append_record, root, batch_dir, "request.json", batch_request)
                        await emit("evidence", "RA is sending PA a pinned SPADE input request.", pa_batch=batch_number)

                        async def pa_progress(event: Mapping[str, Any]) -> None:
                            nonlocal pa_operations
                            used = event["operations_used"]
                            if type(used) is not int or not 0 <= used <= allowance:
                                raise ValueError("PA exceeded its deterministic operation budget.")
                            pa_operations = operations_before_batch + used
                            await emit("evidence", f"PA batch {batch_number}: {event['message']}",
                                       pa_batch=batch_number, pa_operations=pa_operations,
                                       **{key: event[key] for key in ("pa_answers_ref", "pa_operation_ref", "input_resolution_ref") if key in event})
                            if event.get("pa_answers_ref"):
                                answers_by_batch[batch_number] = deepcopy(event["pa_answers_ref"])
                                await save_binding()

                        outcome = await self.product_runtime.request_primitive_context(
                            robot_runtime=self.program_runtime, assignment=inputs.assignment, interaction_root=root,
                            directory=batch_dir, request=batch_request, max_operations=allowance,
                            deadline=started + float(profile["deadline_sec"]), progress=pa_progress,
                        )
                        used = outcome["operations_used"]
                        if type(used) is not int or not 0 <= used <= allowance or outcome.get("model_responses") != 0:
                            raise ValueError("PA response violates the deterministic operation contract.")
                        pa_operations = operations_before_batch + used
                        if outcome.get("answers_ref"):
                            checkpoint = _read_answer_checkpoint(root, outcome["answers_ref"])
                            if Path(outcome["answers_ref"]["ref"]).parent != batch_dir.relative_to(root):
                                raise ValueError("PA answers belong to another batch.")
                            checked_outcome = _answer_result(product_needs, {
                                answer["need_id"]: answer for answer in checkpoint["answers"]
                            }, used)
                            if any(outcome.get(key) != value for key, value in checked_outcome.items()):
                                raise ValueError("PA response evidence differs from its checked answers.")
                            answers_by_batch[batch_number] = deepcopy(outcome["answers_ref"])
                            pa_answer_refs.append(deepcopy(outcome["answers_ref"]))
                        elif outcome.get("evidence_refs") or outcome.get("validation_refs"):
                            raise ValueError("PA evidence requires a checked answer checkpoint.")
                        pa_response_ref = outcome["response_ref"]
                        saved_response = verify_record(root, pa_response_ref)
                        if saved_response != {key: value for key, value in outcome.items() if key != "response_ref"}:
                            raise ValueError("PA reply differs from its saved response.")
                        await emit("evidence", "PA's SPADE reply contains checked measurements and input findings.",
                                   pa_batch=batch_number, pa_response_ref=pa_response_ref)
                        pa_findings = [{"step_index": None, "authority": "PA", "status": "unknown",
                                        "message": message, "pa_response_ref": pa_response_ref}
                                       for message in outcome.get("unresolved", [])]
                        for reference in outcome.get("evidence_refs", []):
                            await asyncio.to_thread(verify_evidence_tree, root, reference)
                            source_pins[reference["ref"]] = deepcopy(reference)
                        roles = outcome.get("validation_refs", {})
                        if set(roles) - set(required_validation_roles(scope)):
                            raise ValueError("PA returned a validation role outside this scope.")
                        validation_refs.update(deepcopy(roles))
                        explicit_needs = []
                        await save_binding()
                        if used < allowance:
                            break
                    await save_binding()
                    dependencies, unresolved_inputs = await asyncio.to_thread(input_needs)
                    if unresolved_inputs or pa_findings:
                        from .program_validation import _report

                        findings = [{"step_index": need["step_index"], "check": need["quantity"],
                                     "status": "unknown", "authority": "PA", "message": need["reason"]}
                                    for need in unresolved_inputs] + pa_findings
                        report = _report(steps, findings, [], [], None, scope=scope)
                        report.update(candidate_ref=candidate_ref, binding_ref=binding_ref,
                                      robot_context_ref=None, evidence_refs=deepcopy(validation_refs))
                        report_ref = await asyncio.to_thread(append_record, root, directory,
                                                            f"validation_{len(decisions):04d}.json", report)
                        reports.append(report_ref)
                        await emit("validation_result", "Input validation found unresolved measurement prerequisites.",
                                   validation_ref=report_ref, validation=report)
                        status, stop_reason = "needs_context", "Required measurements remain unresolved; inspect the recorded input findings."
                        break
                    await emit(
                        "robot_context", "Capturing measured robot state and EE/TCP context."
                    )
                    # Persist and render progress before capture so UI work cannot
                    # consume the snapshot's two-second validation-entry allowance.
                    await emit(
                        "validating",
                        "Preparing program checks and capturing fresh robot context without motion.",
                    )
                    async with AsyncExitStack() as capture_stack:
                        robot_failure = None
                        validation_entry_ns = None
                        try:
                            captured = dict(
                                await capture_stack.enter_async_context(validation_capture(
                                    self.robot_runtime, inputs.assignment, profile=profile,
                                ))
                            )
                            validation_entry_ns = time.time_ns()
                            captured_ref = await asyncio.to_thread(
                                append_record,
                                root,
                                directory,
                                f"robot_context_{len(decisions):04d}.json",
                                captured,
                            )
                            differences = _robot_changed(robot, captured, profile) if robot else []
                            if differences:
                                status, stop_reason = (
                                    "stale",
                                    _robot_change_message(differences),
                                )
                                # Keep the rejected capture as diagnostics, without replacing
                                # the context against which the previous candidate was checked.
                                await emit(
                                    "robot_context",
                                    stop_reason,
                                    previous_robot_context_ref=robot_ref,
                                    robot_context_ref=captured_ref,
                                    differences=differences,
                                )
                                break
                            robot = captured
                            robot_ref = captured_ref
                            source_pins[robot_ref["ref"]] = robot_ref
                        except (
                            ImportError,
                            OSError,
                            RuntimeError,
                            KeyError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            robot = None
                            robot_ref = None
                            robot_failure = str(exc)
                        report = await self.validator(
                            inputs=extended,
                            steps=steps,
                            robot=robot,
                            evidence=validation_refs,
                            directory=directory / f"validation_{len(decisions):04d}",
                            profile=profile,
                            cache=cache,
                            _validation_started_at_ns=validation_entry_ns,
                            progress=validation_progress,
                        )
                    if report.get("scope") != scope:
                        raise ValueError("The validation report scope differs from its refinement run.")
                    await asyncio.to_thread(_assert_inputs_unchanged, extended)
                    if robot_failure:
                        report["robot_context_failure"] = robot_failure
                    if (
                        specification_path
                        and hashlib.sha256(specification_path.read_bytes()).hexdigest()
                        != specification_hash
                    ):
                        status, stop_reason = (
                            "stale",
                            "The explicit experiment specification changed during refinement.",
                        )
                        break
                    if report["status"] == "passed":
                        try:
                            final_robot = dict(
                                await self.robot_runtime.capture_validation_context(
                                    inputs.assignment, profile=profile
                                )
                            )
                            report["final_robot_context_ref"] = await asyncio.to_thread(
                                append_record,
                                root,
                                directory,
                                f"robot_final_{len(decisions):04d}.json",
                                final_robot,
                            )
                            differences = _robot_changed(robot, final_robot, profile)
                            if differences:
                                message = _robot_change_message(differences)
                                await emit(
                                    "robot_context",
                                    message,
                                    previous_robot_context_ref=robot_ref,
                                    robot_context_ref=report["final_robot_context_ref"],
                                    differences=differences,
                                )
                                raise ValueError(message)
                            for reference in validation_refs.values():
                                accepted = await asyncio.to_thread(
                                    verify_evidence_tree, root, reference
                                )
                                stamp = accepted.get("observation_timestamp_ns")
                                if (
                                    stamp is not None
                                    and not 0
                                    <= final_robot["measured_at_ros_ns"] - stamp
                                    <= profile["scene_max_age_sec"] * 1e9
                                ):
                                    raise ValueError(
                                        "Observed scene geometry became stale during validation."
                                    )
                        except (
                            ImportError,
                            OSError,
                            RuntimeError,
                            KeyError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            report["status"] = "unknown"
                            report["findings"].append(
                                {
                                    "step_index": None,
                                    "check": "final_freshness",
                                    "status": "unknown",
                                    "message": str(exc),
                                    "authority": "RA",
                                }
                            )
                    for reference in report.get("calculation_refs", []):
                        source_pins[reference["ref"]] = reference
                    report["candidate_ref"] = candidate_refs[-1]
                    report["binding_ref"] = binding_ref
                    report["robot_context_ref"] = robot_ref
                    report["evidence_refs"] = deepcopy(validation_refs)
                    report_ref = await asyncio.to_thread(
                        append_record,
                        root,
                        directory,
                        f"validation_{len(decisions):04d}.json",
                        report,
                    )
                    reports.append(report_ref)
                    await emit(
                        "validation_result",
                        "Validation completed for the recorded scope.",
                        validation_ref=report_ref,
                        validation=report,
                    )
                    if report["status"] == "passed":
                        status, stop_reason = (
                            "validated_for_declared_scope",
                            "Validated for Gazebo pick-and-place. No motion was executed."
                            if is_pick_place_scope(scope) else
                            "Program validated against the pinned rigid vertical geometry and direct-motion model; no motion was executed.",
                        )
                        break
                    state_key = fingerprint({"steps": steps, "findings": report["findings"]})
                    if state_key in seen:
                        status, stop_reason = "no_progress", "RA repeated an unchanged program and validation findings."
                        break
                    seen.add(state_key)
                    if any(finding.get("authority") == "PA" or finding.get("check") in {
                        "robot_context", "robot_freshness", "freshness", "final_freshness",
                    } for finding in report["findings"]):
                        status, stop_reason = "needs_context", "Measured robot or product context remains unresolved; inspect the recorded findings."
                        break
                    context = {
                        "record_type": "PrimitiveRefinementContext", "run_request_ref": request_ref,
                        "base_context_refs": inputs.context_refs, "previous_candidate_ref": candidate_ref,
                        "binding_ref": binding_ref, "evidence_refs": list(source_pins.values()),
                        "pa_answer_refs": deepcopy(pa_answer_refs), "robot_context_ref": robot_ref,
                        "findings": report["findings"], "validation_ref": report_ref,
                        "created_at_ns": time.time_ns(),
                    }
                    refinement_ref = await asyncio.to_thread(append_record, root, directory,
                                                            f"context_{len(decisions):04d}.json", context)
        except TimeoutError:
            status, stop_reason = (
                "budget_exhausted",
                f"The configured refinement deadline ({profile['deadline_sec']} seconds) was reached during {current_stage}.",
            )
        except asyncio.CancelledError:
            status, stop_reason = "cancelled", "The operator cancelled refinement."
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
            status, stop_reason = "failed", f"{type(exc).__name__}: {exc}"
        result = {
            "record_type": "PrimitiveRefinementResult",
            "request_ref": request_ref,
            "status": status,
            "stop_reason": stop_reason,
            "candidate_refs": candidate_refs,
            "decision_refs": decisions,
            "validation_refs": reports,
            "binding_refs": binding_refs,
            "event_refs": events,
            "pa_batches": pa_batches,
            "pa_operations": pa_operations,
            "elapsed_sec": time.monotonic() - started,
            "motion_executed": False,
            "created_at_ns": time.time_ns(),
        }
        await asyncio.to_thread(append_record, root, directory, "result.json", result)
        await emit("finished", stop_reason, status=status)
        return result


def cancel_primitive_refinement(root: Path) -> bool:
    """Cancel one active no-motion run, leaving its recorded proposal intact."""
    task = _ACTIVE.get(root.resolve())
    if task is None or task.done():
        return False
    task.cancel()
    return True


def enrich_composition_diagnostic(
    root: Path, view: dict[str, Any], context_refs: Mapping[str, Any]
) -> dict[str, Any]:
    """Reopen the latest run without using its program as a new composition input."""
    runs = sorted((root / _RUNS).glob("run_*"))
    if not runs:
        return view
    directory = runs[-1]
    request = verify_record(root, pin(root, directory / "request.json"))
    scope = read_validation_scope(request["profile"])
    if request["base_context_refs"] != context_refs:
        return view
    view["validation_scope"] = scope
    events = [
        verify_record(root, pin(root, path)) for path in sorted(directory.glob("event_*.json"))
    ]
    result_path = directory / "result.json"
    if result_path.exists():
        result = verify_record(root, pin(root, result_path))
        for reference in [
            *result["candidate_refs"],
            *result["decision_refs"],
            *result["validation_refs"],
            *result.get("binding_refs", []),
            *result["event_refs"],
        ]:
            read_pin(root, reference)
        view["status"], view["message"] = result["status"], result["stop_reason"]
    else:
        active = root.resolve() in _ACTIVE and not _ACTIVE[root.resolve()].done()
        view["status"] = events[-1]["stage"] if active and events else "interrupted"
        view["message"] = (
            events[-1]["message"]
            if active and events
            else "The previous refinement run was interrupted; its recorded proposal is preserved."
        )
        result = None
    proposed = [event for event in events if event["stage"] == "proposal"]
    # Match the live proposal count; context requests are separate RA decisions.
    view["attempt_count"] = len(proposed)
    if proposed:
        latest = proposed[-1]
        candidate = read_pin(root, latest["candidate_ref"])
        view["candidate"] = candidate
        view["latest_candidate_ref"] = latest["candidate_ref"]["ref"]
    view.pop("binding", None)
    view.pop("binding_ref", None)
    view.pop("resolved_primitive_steps", None)
    bound_inputs = None
    for event in reversed(events):
        if "binding_ref" not in event or event["stage"] != "binding" or not proposed:
            continue
        from .program_binding import read_program_binding

        binding, bound_inputs = read_program_binding(_load_inputs(root), event["binding_ref"])
        if binding["candidate_ref"] == proposed[-1]["candidate_ref"]:
            view["binding"], view["binding_ref"] = binding, event["binding_ref"]
            view["composition_input"] = deepcopy(bound_inputs.composition_input)
            break
        bound_inputs = None
    validations = [event for event in events if event["stage"] == "validation_result"]
    view.pop("validation", None)
    if validations:
        validation = read_pin(root, validations[-1]["validation_ref"])
        if validation.get("scope") != scope:
            raise ValueError("The saved validation report scope differs from its refinement run.")
        # A revision can be saved just before cancellation or the deadline. Its
        # predecessor's report stays in the trace and cannot describe this proposal.
        if (proposed and validation.get("candidate_ref") == proposed[-1]["candidate_ref"]
                and validation.get("binding_ref") == view.get("binding_ref")):
            effective_steps = view.get("binding", view["candidate"])["primitive_steps"]
            if validation.get("candidate_fingerprint") != fingerprint(effective_steps):
                raise ValueError("Validation does not match the displayed primitive binding.")
            view["validation"] = validation
    if bound_inputs is not None:
        from .program_binding import display_primitive_steps

        steps = view["binding"]["primitive_steps"]
        calculations = [event["calculation_ref"] for event in events if "calculation_ref" in event
                        and event.get("binding_ref") == view["binding_ref"]]
        view["resolved_primitive_steps"] = display_primitive_steps(
            bound_inputs, steps, view.get("validation"), calculation_refs=calculations,
        )
        view["binding_issues"] = assess_program_dependencies(
            steps, bound_inputs.catalog, bound_inputs.composition_input["robot_state"],
            read_evidence=lambda ref, pointer: _evidence_value(bound_inputs, ref, pointer),
            result_schema=lambda ref: _result_schema(steps, ref, bound_inputs),
        )["issues"]
    pa_responses = []
    pa_trace = []
    for event in events:
        if "previous_robot_context_ref" in event:
            for field in ("previous_robot_context_ref", "robot_context_ref"):
                reference = event[field]
                if owned_path(root, reference["ref"]).parent != directory:
                    raise ValueError("Compared robot context does not belong to this refinement run.")
                context = verify_record(root, reference)
                if context.get("record_type") != "RobotValidationContext":
                    raise ValueError("Robot comparison must reference a RobotValidationContext.")
        for field, record_type in (("pa_reply_ref", "PrimitiveContextReply"),
                                   ("input_resolution_ref", "PrimitiveInputResolution"),
                                   ("pa_answers_ref", "PrimitiveContextAnswers"),
                                   ("pa_operation_ref", "PrimitiveContextOperationResult")):
            if field not in event:
                continue
            reference = event[field]
            batch_dir = directory / f"pa_{event['pa_batch']:04d}"
            if not owned_path(root, reference["ref"]).is_relative_to(batch_dir):
                raise ValueError("PA progress record belongs to another investigation.")
            if field == "pa_answers_ref":
                from ..pa.primitive_context import _read_answer_checkpoint

                record = _read_answer_checkpoint(root, reference)
            else:
                record = verify_record(root, reference)
            if record.get("record_type") != record_type:
                raise ValueError("PA progress references an unexpected record type.")
            pa_trace.append({"reference": reference, "record": record})
        if "pa_response_ref" in event:
            response = verify_record(root, event["pa_response_ref"])
            expected = directory / f"pa_{event['pa_batch']:04d}" / "response.json"
            if (
                event["pa_response_ref"]["ref"] != expected.relative_to(root).as_posix()
                or response.get("record_type") != "PrimitiveContextResponse"
            ):
                raise ValueError("PA feedback does not belong to this investigation.")
            pa_responses.append(response)
    view["refinement"] = {
        "request": request,
        "result": result,
        "events": events,
        "pa_responses": pa_responses,
        "pa_trace": pa_trace,
    }
    return view
