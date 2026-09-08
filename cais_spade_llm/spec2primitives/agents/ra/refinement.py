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
) -> bool:
    for key in (
        "configuration_sha256",
        "model_parameters_sha256",
        "frame_id",
        "ee_link",
        "tcp_link",
        "held_part",
    ):
        if previous.get(key) != current.get(key):
            return True
    if not np.allclose(previous["ee_from_tcp"], current["ee_from_tcp"], rtol=0, atol=1e-6):
        return True
    first = dict(
        zip(previous["joint_state"]["names"], previous["joint_state"]["positions"], strict=True)
    )
    second = dict(
        zip(current["joint_state"]["names"], current["joint_state"]["positions"], strict=True)
    )
    return first.keys() != second.keys() or any(
        abs(first[key] - second[key]) > profile["state_change_joint_tolerance_rad"] for key in first
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
                        if robot and _robot_changed(robot, captured, self.profile):
                            status, stop_reason = (
                                "stale",
                                "Robot state, model or tool configuration changed during refinement. Capture fresh context.",
                            )
                            break
                        robot = captured
                        robot_ref = await asyncio.to_thread(
                            append_record,
                            root,
                            directory,
                            f"robot_context_{len(decisions):04d}.json",
                            robot,
                        )
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
                            if _robot_changed(robot, final_robot, self.profile):
                                raise ValueError(
                                    "Robot state or tool configuration changed during validation."
                                )
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
                    needs = [*pending_requests, *dependencies["context_requests"]]
                    needs.extend(
                        {
                            "step_index": item.get("step_index"),
                            "authority": item["authority"],
                            "quantity": item["check"],
                            "reason": item["message"],
                            "evidence_refs": [],
                        }
                        for item in report["findings"]
                        if item.get("authority") and item["status"] == "unknown"
                    )
                    # Measurement requests already addressed by the selected RA
                    # never go to PA. PA receives no primitive sequence to repair.
                    product_needs = [need for need in needs if need["authority"] == "PA"]
                    unique_needs = {fingerprint(need): need for need in product_needs}
                    if (
                        unique_needs
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
                        available = [
                            {"ref": ref, "sha256": sha}
                            for ref, sha in inputs.record_hashes.items()
                            if not ref.startswith("resources/")
                        ]
                        available.extend(
                            value
                            for value in source_pins.values()
                            if read_pin(root, value).get("record_type")
                            not in {"RobotValidationContext", "PrimitiveCalculationRecord"}
                        )
                        batch_request = {
                            "record_type": "PrimitiveContextRequest",
                            "target_feature": deepcopy(inputs.composition_input["target_feature"]),
                            "needs": list(unique_needs.values()),
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
                        await asyncio.to_thread(
                            append_record,
                            root,
                            batch_dir,
                            "response.json",
                            {"record_type": "PrimitiveContextResponse", **outcome},
                        )
                        if outcome["status"] == "authority_conflict":
                            status, stop_reason = "authority_conflict", outcome["reason"]
                            break
                        for reference in outcome.get("evidence_refs", []):
                            await asyncio.to_thread(read_pin, root, reference)
                            source_pins[reference["ref"]] = deepcopy(reference)
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
                    findings = [*dependencies["issues"], *report["findings"]]
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
    if proposed:
        latest = proposed[-1]
        candidate = read_pin(root, latest["candidate_ref"])
        view["candidate"] = candidate
        view["latest_candidate_ref"] = latest["candidate_ref"]["ref"]
    validations = [event for event in events if event["stage"] == "validation_result"]
    if validations:
        view["validation"] = read_pin(root, validations[-1]["validation_ref"])
    view["refinement"] = {"request": request, "result": result, "events": events}
    return view
