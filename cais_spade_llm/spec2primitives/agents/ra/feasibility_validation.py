"""Persist one exact RobotAgent plan-only feasibility verdict."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    ReachabilityCheckRecord,
)
from cais_spade_llm.spec2primitives.ontology.workcell import (
    PredefinedWorkcellSnapshot,
)

_VALIDATION_STATUSES = frozenset({"accepted", "rejected", "needs_context"})
_RESPONSE_KEYS = {"status", "current_state", "desired_state", "feedback"}
_ENDPOINT_KEYS = {"status", "message", "error_code"}
_VALIDATION_SCOPE = "endpoint_motion"
_CHECKED_CONSTRAINTS = (
    "positional_ik",
    "collision_aware_endpoints",
    "path_between_endpoints",
)
_UNVALIDATED_CONSTRAINTS_WITHOUT_PROCESS_TOLERANCE = (
    "grasping",
    "end_effector_orientation",
    "attached_object_geometry",
    "force_contact",
    "insertion_constraints",
)


class RobotAgentFeasibilityError(ValueError):
    """Raised when exact-resource plan-only validation is inconsistent."""


class RobotAgentFeasibilityRuntime(Protocol):
    """Expose plan-only validation through the exact selected RobotAgent."""

    async def validate_plan_only_allocation(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Return only endpoint results, feedback, and one closed verdict."""
        ...


@dataclass(frozen=True)
class PlanOnlyFeasibilityValidation:
    """Hold one persisted hash-pinned no-motion RobotAgent validation."""

    record_path: Path = field(repr=False, compare=False)
    record_ref: str
    validation_number: int
    process_symbol: str
    process_iri: str
    feature_iri: str
    current_state_iri: str
    desired_state_iri: str
    resource_symbol: str
    resource_iri: str
    resource_jid: str
    execution_mode: str
    moveit_group: str
    end_effector_link: str
    target_frame: str
    validation_scope: str
    checked_constraints: tuple[str, ...]
    unvalidated_constraints: tuple[str, ...]
    status: str
    feedback: str | None
    request_fingerprint: str
    fingerprint: str

    def to_record(self) -> Mapping[str, object]:
        """Reload and return the exact persisted record."""
        try:
            value = json.loads(self.record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RobotAgentFeasibilityError(
                "Plan-only feasibility record could not be read."
            ) from exc
        if not isinstance(value, Mapping):
            raise RobotAgentFeasibilityError("Plan-only feasibility record must be an object.")
        return value


async def validate_provisional_allocation(
    runtime: RobotAgentFeasibilityRuntime,
    *,
    interaction_root: Path,
    workcell: PredefinedWorkcellSnapshot,
    reachability: ReachabilityCheckRecord,
    validation_number: int = 1,
) -> PlanOnlyFeasibilityValidation:
    """Ask only the PA-chosen RobotAgent for a plan-only two-endpoint check."""
    if isinstance(validation_number, bool) or not isinstance(validation_number, int):
        raise RobotAgentFeasibilityError("validation_number must be a positive integer.")
    if validation_number < 1:
        raise RobotAgentFeasibilityError("validation_number must be a positive integer.")
    workcell.assert_unchanged()
    reachability.assert_unchanged()
    moveit_group, end_effector_link = _moveit_profile(workcell, reachability)
    unvalidated_constraints = _unvalidated_constraints(reachability.process_symbol)
    request = {
        "process_symbol": reachability.process_symbol,
        "process_iri": reachability.process_iri,
        "feature_iri": reachability.feature_iri,
        "resource_symbol": reachability.resource_symbol,
        "resource_iri": reachability.resource_iri,
        "resource_jid": reachability.resource_jid,
        "execution_mode": reachability.execution_mode,
        "moveit_group": moveit_group,
        "end_effector_link": end_effector_link,
        "target_frame": reachability.target_frame,
        "validation_scope": _VALIDATION_SCOPE,
        "checked_constraints": list(_CHECKED_CONSTRAINTS),
        "unvalidated_constraints": list(unvalidated_constraints),
        "current_state": {
            "state_iri": reachability.current_state.state_iri,
            "evidence_handle": reachability.current_state.evidence_handle,
            "translation_m": list(reachability.current_state.translation_m),
            "location_record_ref": (reachability.current_state.location_record_ref),
            "location_record_sha256": (reachability.current_state.location_record_sha256),
        },
        "desired_state": {
            "state_iri": reachability.desired_state.state_iri,
            "evidence_handle": reachability.desired_state.evidence_handle,
            "translation_m": list(reachability.desired_state.translation_m),
            "location_record_ref": (reachability.desired_state.location_record_ref),
            "location_record_sha256": (reachability.desired_state.location_record_sha256),
        },
        "mode": "plan_only",
        "motion_executed": False,
        "request_fingerprint": reachability.fingerprint,
    }
    try:
        response = await runtime.validate_plan_only_allocation(request)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        response = {
            "status": "needs_context",
            "current_state": {
                "status": "needs_context",
                "message": "Exact RobotAgent validation is unavailable.",
                "error_code": None,
            },
            "desired_state": {
                "status": "needs_context",
                "message": "Exact RobotAgent validation is unavailable.",
                "error_code": None,
            },
            "feedback": f"{type(exc).__name__}: {exc}",
        }
    validated_response = _validated_response(response)
    payload: dict[str, object] = {
        "schema_version": 2,
        "record_type": "PlanOnlyFeasibilityValidationRecord",
        "validation_number": validation_number,
        "validator_authority": reachability.resource_jid,
        "process_symbol": reachability.process_symbol,
        "process_iri": reachability.process_iri,
        "feature_iri": reachability.feature_iri,
        "current_state_iri": reachability.current_state.state_iri,
        "desired_state_iri": reachability.desired_state.state_iri,
        "resource_symbol": reachability.resource_symbol,
        "resource_iri": reachability.resource_iri,
        "resource_jid": reachability.resource_jid,
        "execution_mode": reachability.execution_mode,
        "moveit_group": moveit_group,
        "end_effector_link": end_effector_link,
        "target_frame": reachability.target_frame,
        "validation_scope": _VALIDATION_SCOPE,
        "checked_constraints": list(_CHECKED_CONSTRAINTS),
        "unvalidated_constraints": list(unvalidated_constraints),
        "current_state": validated_response["current_state"],
        "desired_state": validated_response["desired_state"],
        "mode": "plan_only",
        "motion_executed": False,
        "status": validated_response["status"],
        "feedback": validated_response["feedback"],
        "validated_at_ns": time.time_ns(),
        "request_fingerprint": reachability.fingerprint,
    }
    payload["fingerprint"] = _fingerprint(payload)
    root = Path(interaction_root).resolve()
    destination = (
        root
        / "resources"
        / reachability.resource_jid
        / "validation"
        / f"plan_only_validation_{validation_number:04d}"
    )
    record_path = destination / "plan_only_feasibility_validation_record.json"
    _persist_record(destination, record_path.name, payload)
    return PlanOnlyFeasibilityValidation(
        record_path=record_path,
        record_ref=record_path.relative_to(root).as_posix(),
        validation_number=validation_number,
        process_symbol=reachability.process_symbol,
        process_iri=reachability.process_iri,
        feature_iri=reachability.feature_iri,
        current_state_iri=reachability.current_state.state_iri,
        desired_state_iri=reachability.desired_state.state_iri,
        resource_symbol=reachability.resource_symbol,
        resource_iri=reachability.resource_iri,
        resource_jid=reachability.resource_jid,
        execution_mode=reachability.execution_mode,
        moveit_group=moveit_group,
        end_effector_link=end_effector_link,
        target_frame=reachability.target_frame,
        validation_scope=_VALIDATION_SCOPE,
        checked_constraints=_CHECKED_CONSTRAINTS,
        unvalidated_constraints=unvalidated_constraints,
        status=str(validated_response["status"]),
        feedback=(
            str(validated_response["feedback"])
            if validated_response["feedback"] is not None
            else None
        ),
        request_fingerprint=reachability.fingerprint,
        fingerprint=str(payload["fingerprint"]),
    )


def _unvalidated_constraints(process_symbol: str) -> tuple[str, ...]:
    """Name the process-specific tolerance that endpoint motion does not validate."""
    if not isinstance(process_symbol, str) or not process_symbol:
        raise RobotAgentFeasibilityError("Selected process symbol is unavailable.")
    return (
        *_UNVALIDATED_CONSTRAINTS_WITHOUT_PROCESS_TOLERANCE[:3],
        f"{process_symbol}_tolerance",
        *_UNVALIDATED_CONSTRAINTS_WITHOUT_PROCESS_TOLERANCE[3:],
    )


def _moveit_profile(
    workcell: PredefinedWorkcellSnapshot,
    reachability: ReachabilityCheckRecord,
) -> tuple[str, str]:
    matches = [
        resource
        for resource in workcell._profile.resources
        if resource.symbol == reachability.resource_symbol
    ]
    if len(matches) != 1:
        raise RobotAgentFeasibilityError("PA-chosen resource manifest is unavailable.")
    manifest_path = matches[0].manifest_path.resolve()
    try:
        source = manifest_path.read_bytes()
        payload = json.loads(source.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RobotAgentFeasibilityError("PA-chosen resource manifest could not be read.") from exc
    if hashlib.sha256(source).hexdigest() != reachability.manifest_sha256:
        raise RobotAgentFeasibilityError(
            "PA-chosen resource manifest changed before RobotAgent validation."
        )
    resource = payload.get(reachability.resource_symbol) if isinstance(payload, Mapping) else None
    environment_name = "gazebo" if reachability.execution_mode == "simulation" else "real"
    environment = resource.get(environment_name) if isinstance(resource, Mapping) else None
    controller = environment.get("controller") if isinstance(environment, Mapping) else None
    move_group = controller.get("move_group") if isinstance(controller, Mapping) else None
    group_name = move_group.get("group_name") if isinstance(move_group, Mapping) else None
    end_effector_link = (
        move_group.get("tcp_link") or move_group.get("ee_link")
        if isinstance(move_group, Mapping)
        else None
    )
    frame_id = move_group.get("frame_id") if isinstance(move_group, Mapping) else None
    if (
        not isinstance(group_name, str)
        or not group_name
        or not isinstance(end_effector_link, str)
        or not end_effector_link
        or frame_id != reachability.target_frame
    ):
        raise RobotAgentFeasibilityError(
            "PA-chosen resource has no matching MoveIt validation profile."
        )
    return group_name, end_effector_link


def _validated_response(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _RESPONSE_KEYS:
        raise RobotAgentFeasibilityError(
            "RobotAgent plan-only validation response fields are invalid."
        )
    status = value["status"]
    if status not in _VALIDATION_STATUSES:
        raise RobotAgentFeasibilityError("RobotAgent plan-only validation status is invalid.")
    endpoints: dict[str, Mapping[str, object]] = {}
    for state_name in ("current_state", "desired_state"):
        endpoint = value[state_name]
        if not isinstance(endpoint, Mapping) or set(endpoint) != _ENDPOINT_KEYS:
            raise RobotAgentFeasibilityError(f"RobotAgent {state_name} validation is invalid.")
        endpoint_status = endpoint["status"]
        message = endpoint["message"]
        error_code = endpoint["error_code"]
        if (
            endpoint_status not in _VALIDATION_STATUSES
            or not isinstance(message, str)
            or not message.strip()
            or (
                error_code is not None
                and (isinstance(error_code, bool) or not isinstance(error_code, int))
            )
        ):
            raise RobotAgentFeasibilityError(f"RobotAgent {state_name} validation is invalid.")
        endpoints[state_name] = dict(endpoint)
    expected_status = (
        "rejected"
        if any(item["status"] == "rejected" for item in endpoints.values())
        else (
            "needs_context"
            if any(item["status"] == "needs_context" for item in endpoints.values())
            else "accepted"
        )
    )
    feedback = value["feedback"]
    if status != expected_status or (
        feedback is not None and (not isinstance(feedback, str) or not feedback.strip())
    ):
        raise RobotAgentFeasibilityError("RobotAgent plan-only validation verdict is inconsistent.")
    return {
        "status": status,
        "current_state": endpoints["current_state"],
        "desired_state": endpoints["desired_state"],
        "feedback": feedback,
    }


def _persist_record(
    destination: Path,
    record_name: str,
    payload: Mapping[str, object],
) -> None:
    if destination.exists():
        raise RobotAgentFeasibilityError(
            f"Plan-only validation already exists: {destination.name}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        with (temporary / record_name).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.rename(destination)
    except (OSError, TypeError, ValueError) as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise RobotAgentFeasibilityError("Plan-only validation persistence failed.") from exc


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PlanOnlyFeasibilityValidation",
    "RobotAgentFeasibilityError",
    "RobotAgentFeasibilityRuntime",
    "validate_provisional_allocation",
]
