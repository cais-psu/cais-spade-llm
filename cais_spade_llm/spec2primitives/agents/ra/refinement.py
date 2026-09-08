from __future__ import annotations

"""Bound evidence acquisition, audited calculation, validation and RA revision."""

import asyncio
import hashlib
import json
import time
from contextlib import asynccontextmanager
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .primitive_composition import (
    _assert_inputs_unchanged,
    _evidence_value,
    _load_inputs,
    _read_record,
    _result_schema,
    _with_refinement,
    author_primitive_program_candidate,
)
from .program_dependencies import assess_program_dependencies
from .program_validation import validate_program
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
        self, root: Path, *, progress: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None
    ) -> dict[str, Any]:
        """Start one run or join the existing run; disconnects do not duplicate work."""
        root = root.resolve()
        task = _ACTIVE.get(root)
        if task is None or task.done():
            task = asyncio.create_task(self._run(root, progress=progress))
            _ACTIVE[root] = task
            task.add_done_callback(
                lambda done: _ACTIVE.pop(root, None) if _ACTIVE.get(root) is done else None
            )
        return await asyncio.shield(task)

    async def _run(
        self, root: Path, *, progress: Callable[[Mapping[str, Any]], Awaitable[None]] | None
    ) -> dict[str, Any]:
        inputs = await asyncio.to_thread(_load_inputs, root)
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
                "profile": self.profile,
                "created_at_ns": time.time_ns(),
            },
        )
        events: list[dict[str, str]] = []
        candidate_refs: list[dict[str, str]] = []
        decisions: list[dict[str, str]] = []
        reports: list[dict[str, str]] = []
        source_pins: dict[str, dict[str, str]] = {}
        validation_refs: dict[str, dict[str, str]] = {}
        pa_findings: list[dict[str, Any]] = []
        robot_ref = None
        robot = None
        cache: dict[str, dict[str, Any]] = {}
        pa_batches = pa_operations = 0
        stop_reason = "The refinement budget was exhausted."
        status = "budget_exhausted"
        started = time.monotonic()

        async def emit(stage: str, message: str, **details: Any) -> None:
            record = {
                "record_type": "PrimitiveRefinementEvent",
                "stage": stage,
                "message": message,
                "elapsed_sec": time.monotonic() - started,
                "created_at_ns": time.time_ns(),
                **deepcopy(details),
            }
            reference = await asyncio.to_thread(
                append_record, root, directory, f"event_{len(events) + 1:04d}.json", record
            )
            events.append(reference)
            if progress is not None:
                await progress(record)

        try:
            async with _deadline(float(self.profile["deadline_sec"])):
                specification_ref, specification_path, specification_hash = await asyncio.to_thread(
                    _experiment_specification, root, directory, self.profile
                )
                if specification_ref:
                    validation_refs["specification"] = specification_ref
                    source_pins[specification_ref["ref"]] = specification_ref
                refinement_ref = None
                seen: set[str] = set()
                attempted_requests: set[tuple[str, str]] = set()
                pending_requests: list[dict[str, Any]] = []
                while (
                    len(candidate_refs) < self.profile["max_candidates"]
                    and len(decisions)
                    < self.profile["max_candidates"] + self.profile["max_pa_batches"]
                ):
                    await asyncio.to_thread(_assert_inputs_unchanged, inputs)
                    await emit(
                        "composing",
                        "RA is authoring the initial proposal."
                        if not candidate_refs
                        else "RA is revising its program using the recorded findings.",
                    )
                    candidate = await author_primitive_program_candidate(
                        self.program_runtime, root, refinement_ref=refinement_ref
                    )
                    candidate_ref = pin(root, candidate.path)
                    decisions.append(candidate_ref)
                    if candidate.record["status"] == "needs_context" and candidate_refs:
                        latest = await asyncio.to_thread(
                            _read_record,
                            candidate.path.parent
                            / f"exchange_{len(candidate.record['exchange_refs']):04d}.json",
                        )
                        pending_requests = deepcopy(latest["response"]["action"]["requests"])
                        previous = read_pin(root, candidate_refs[-1])
                        if any(
                            need["step_index"] > len(previous["primitive_steps"])
                            for need in pending_requests
                        ):
                            raise ValueError(
                                "RA context request refers to a step absent from its preceding candidate."
                            )
                    elif candidate.record["status"] != "proposed":
                        status, stop_reason = (
                            candidate.record["status"],
                            candidate.record.get("reason")
                            or "RA did not submit a valid candidate.",
                        )
                        break
                    else:
                        candidate_refs.append(candidate_ref)
                        previous = candidate.record
                        await emit(
                            "proposal",
                            "RA's program proposal is recorded; validation is still pending.",
                            candidate_ref=candidate_ref,
                            candidate=previous,
                            candidate_count=len(candidate_refs),
                        )
                    extended = (
                        await asyncio.to_thread(_with_refinement, inputs, refinement_ref)
                        if refinement_ref
                        else inputs
                    )
                    steps = deepcopy(previous["primitive_steps"])
                    dependencies = await asyncio.to_thread(
                        assess_program_dependencies,
                        steps,
                        extended.catalog,
                        extended.composition_input["robot_state"],
                        read_evidence=lambda ref, pointer: _evidence_value(extended, ref, pointer),
                        result_schema=lambda ref: _result_schema(steps, ref, extended),
                    )
                    await emit(
                        "robot_context", "Capturing measured robot state and EE/TCP context."
                    )
                    robot_failure = None
                    try:
                        captured = dict(
                            await self.robot_runtime.capture_validation_context(
                                inputs.assignment, profile=self.profile
                            )
                        )
                        captured_ref = await asyncio.to_thread(
                            append_record,
                            root,
                            directory,
                            f"robot_context_{len(decisions):04d}.json",
                            captured,
                        )
                        differences = _robot_changed(robot, captured, self.profile) if robot else []
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
                    await emit(
                        "validating",
                        "Calculating selected helpers and checking the program without motion.",
                    )
                    report = await self.validator(
                        inputs=extended,
                        steps=steps,
                        robot=robot,
                        evidence=validation_refs,
                        directory=directory / f"validation_{len(decisions):04d}",
                        profile=self.profile,
                        cache=cache,
                    )
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
                                    inputs.assignment, profile=self.profile
                                )
                            )
                            report["final_robot_context_ref"] = await asyncio.to_thread(
                                append_record,
                                root,
                                directory,
                                f"robot_final_{len(decisions):04d}.json",
                                final_robot,
                            )
                            differences = _robot_changed(robot, final_robot, self.profile)
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
                                    <= self.profile["scene_max_age_sec"] * 1e9
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
                            "Program validated against the pinned rigid vertical geometry and direct-motion model; no motion was executed.",
                        )
                        break
                    product_pins = [
                        reference
                        for reference in source_pins.values()
                        if read_pin(root, reference).get("record_type")
                        not in {"RobotValidationContext", "PrimitiveCalculationRecord"}
                    ]
                    state_key = fingerprint(
                        {
                            "steps": steps,
                            "findings": report["findings"],
                            "evidence": product_pins,
                            "requests": pending_requests,
                            "robot_state": {
                                key: robot[key]
                                for key in ("configuration_sha256", "ee_pose", "ee_from_tcp")
                            }
                            if robot
                            else None,
                        }
                    )
                    if state_key in seen:
                        status, stop_reason = (
                            "no_progress",
                            "RA repeated an unchanged candidate and unresolved findings.",
                        )
                        break
                    seen.add(state_key)
                    if len(candidate_refs) >= self.profile["max_candidates"]:
                        break
                    available = [
                        {"ref": ref, "sha256": sha}
                        for ref, sha in inputs.record_hashes.items()
                        if not ref.startswith("resources/")
                    ]
                    available.extend(product_pins)
                    available_hashes = {item["ref"]: item["sha256"] for item in available}
                    evidence_key = fingerprint(available_hashes)
                    product_needs = []
                    request_keys = set()
                    # Explicit RA requests keep their wording and take precedence over
                    # the same missing field derived from its selected primitive.
                    for need in [*pending_requests, *dependencies["context_requests"]]:
                        if need["authority"] != "PA":
                            continue
                        request_key = fingerprint(
                            {
                                "step_index": need["step_index"],
                                "primitive_symbol": steps[need["step_index"] - 1]["primitive_symbol"],
                                "params": steps[need["step_index"] - 1]["params"],
                                "quantity": need.get("parameter_path", need["quantity"]),
                            }
                        )
                        if (
                            request_key in request_keys
                            or (request_key, evidence_key) in attempted_requests
                        ):
                            continue
                        request_keys.add(request_key)
                        product_needs.append(deepcopy(need))
                    if (
                        product_needs
                        and self.product_runtime is not None
                        and pa_batches < self.profile["max_pa_batches"]
                        and pa_operations < self.profile["max_pa_operations"]
                    ):
                        pa_batches += 1
                        await emit(
                            "evidence",
                            "PA is investigating missing product and scene facts.",
                            pa_batch=pa_batches,
                        )
                        attempted_requests.update((key, evidence_key) for key in request_keys)
                        batch_request = {
                            "record_type": "PrimitiveContextRequest",
                            "target_feature": deepcopy(inputs.composition_input["target_feature"]),
                            "needs": deepcopy(product_needs),
                            "evidence_refs": available,
                            "base_context_refs": inputs.context_refs,
                        }
                        batch_dir = directory / f"pa_{pa_batches:04d}"
                        await asyncio.to_thread(
                            append_record, root, batch_dir, "request.json", batch_request
                        )
                        remaining = self.profile["max_pa_operations"] - pa_operations
                        outcome = dict(
                            await self.product_runtime.investigate(
                                interaction_root=root,
                                directory=batch_dir,
                                request=batch_request,
                                max_operations=min(6, remaining),
                            )
                        )
                        used = outcome["operations_used"]
                        if type(used) is not int or not 0 <= used <= min(6, remaining):
                            raise ValueError("PA exceeded its evidence budget.")
                        pa_operations += used
                        pa_response_ref = await asyncio.to_thread(
                            append_record,
                            root,
                            batch_dir,
                            "response.json",
                            {"record_type": "PrimitiveContextResponse", **outcome},
                        )
                        await emit(
                            "evidence",
                            "PA evidence investigation completed.",
                            pa_batch=pa_batches,
                            pa_response_ref=pa_response_ref,
                        )
                        # PA explanations are revision feedback, never geometry evidence.
                        pa_findings = [
                            {
                                "step_index": None,
                                "authority": "PA",
                                "status": "unknown",
                                "message": message,
                                "pa_response_ref": pa_response_ref,
                            }
                            for message in outcome.get("unresolved", [])
                        ]
                        if outcome["status"] == "authority_conflict":
                            status, stop_reason = "authority_conflict", outcome["reason"]
                            break
                        for reference in outcome.get("evidence_refs", []):
                            await asyncio.to_thread(read_pin, root, reference)
                            source_pins[reference["ref"]] = deepcopy(reference)
                            available_hashes[reference["ref"]] = reference["sha256"]
                        # Returning partial evidence does not itself warrant repeating
                        # this investigation before RA has reviewed those records.
                        returned_evidence_key = fingerprint(available_hashes)
                        attempted_requests.update(
                            (key, returned_evidence_key) for key in request_keys
                        )
                        validation_refs.update(deepcopy(outcome.get("validation_refs", {})))
                        if (
                            not outcome.get("evidence_refs")
                            and not robot
                            and candidate.record["status"] == "needs_context"
                        ):
                            status, stop_reason = (
                                "needs_context",
                                "Required product evidence and measured robot context remain unavailable.",
                            )
                            break
                    pending_requests = []
                    findings = [*dependencies["issues"], *report["findings"], *pa_findings]
                    context = {
                        "record_type": "PrimitiveRefinementContext",
                        "run_request_ref": request_ref,
                        "base_context_refs": inputs.context_refs,
                        "previous_candidate_ref": candidate_refs[-1],
                        "evidence_refs": list(source_pins.values()),
                        "robot_context_ref": robot_ref,
                        "findings": findings,
                        "validation_ref": report_ref,
                        "created_at_ns": time.time_ns(),
                    }
                    refinement_ref = await asyncio.to_thread(
                        append_record,
                        root,
                        directory,
                        f"context_{len(decisions):04d}.json",
                        context,
                    )
        except TimeoutError:
            status, stop_reason = (
                "budget_exhausted",
                f"The configured refinement deadline ({self.profile['deadline_sec']} seconds) was reached.",
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
    if request["base_context_refs"] != context_refs:
        return view
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
    validations = [event for event in events if event["stage"] == "validation_result"]
    if validations:
        view["validation"] = read_pin(root, validations[-1]["validation_ref"])
    pa_responses = []
    for event in events:
        if "previous_robot_context_ref" in event:
            for field in ("previous_robot_context_ref", "robot_context_ref"):
                reference = event[field]
                if owned_path(root, reference["ref"]).parent != directory:
                    raise ValueError("Compared robot context does not belong to this refinement run.")
                context = verify_record(root, reference)
                if context.get("record_type") != "RobotValidationContext":
                    raise ValueError("Robot comparison must reference a RobotValidationContext.")
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
    }
    return view
