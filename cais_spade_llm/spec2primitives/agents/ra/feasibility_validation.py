"""Persist one exact RobotAgent plan-only feasibility verdict."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    CartesianReachabilityRequest,
    CartesianStateEvidence,
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
_CARTESIAN_RESPONSE_KEYS = {
    "status",
    "live_start_pose",
    "ee_to_tcp_transform",
    "waypoints",
    "phases",
    "feedback",
    "motion_executed",
}
_CARTESIAN_CHECKED_CONSTRAINTS = (
    "live_tf",
    "collision_aware_cartesian_pick_path",
    "collision_aware_cartesian_transfer_place_path",
    "complete_path_fraction",
)
_CARTESIAN_PARAMETERS = {
    "max_step_m": 0.01,
    "jump_threshold": 0.0,
    "avoid_collisions": True,
    "minimum_fraction": 0.999,
}


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
    schema_version: int = 2
    motion_mode: str | None = None
    tcp_link: str | None = None
    cartesian_path_service: str | None = None

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
    reachability: ReachabilityCheckRecord | CartesianReachabilityRequest,
    validation_number: int = 1,
) -> PlanOnlyFeasibilityValidation:
    """Ask only the PA-chosen RobotAgent for the applicable no-motion check."""
    if isinstance(validation_number, bool) or not isinstance(validation_number, int):
        raise RobotAgentFeasibilityError("validation_number must be a positive integer.")
    if validation_number < 1:
        raise RobotAgentFeasibilityError("validation_number must be a positive integer.")
    if isinstance(reachability, CartesianReachabilityRequest):
        return await _validate_cartesian_allocation(
            runtime,
            interaction_root=interaction_root,
            workcell=workcell,
            prepared=reachability,
            validation_number=validation_number,
        )
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


async def _validate_cartesian_allocation(  # noqa: PLR0913
    runtime: RobotAgentFeasibilityRuntime,
    *,
    interaction_root: Path,
    workcell: PredefinedWorkcellSnapshot,
    prepared: CartesianReachabilityRequest,
    validation_number: int,
) -> PlanOnlyFeasibilityValidation:
    """Persist one strict live Cartesian pick-and-place feasibility result."""
    root = Path(interaction_root).resolve()
    workcell.assert_unchanged()
    prepared.assert_unchanged(root)
    if prepared.workcell_fingerprint != workcell.fingerprint:
        raise RobotAgentFeasibilityError("Cartesian request and Workcell fingerprints differ.")
    controller = prepared.controller
    unvalidated_constraints = _cartesian_unvalidated_constraints(prepared.process_symbol)
    request = {
        "process_symbol": prepared.process_symbol,
        "process_iri": prepared.process_iri,
        "feature_iri": prepared.feature_iri,
        "resource_symbol": prepared.resource_symbol,
        "resource_iri": prepared.resource_iri,
        "resource_jid": prepared.resource_jid,
        "execution_mode": prepared.execution_mode,
        "motion_mode": "cartesian_pick_place",
        "moveit_group": controller.moveit_group,
        "end_effector_link": controller.end_effector_link,
        "tcp_link": controller.tcp_link,
        "target_frame": controller.target_frame,
        "cartesian_path_service": controller.cartesian_path_service,
        "validation_scope": "cartesian_pick_place",
        "checked_constraints": list(_CARTESIAN_CHECKED_CONSTRAINTS),
        "unvalidated_constraints": list(unvalidated_constraints),
        "cartesian_parameters": dict(_CARTESIAN_PARAMETERS),
        "current_state": _cartesian_request_state(prepared.current_state),
        "desired_state": _cartesian_request_state(prepared.desired_state),
        "grounded_targets": dict(prepared.cartesian_targets),
        "mode": "plan_only",
        "motion_executed": False,
        "request_fingerprint": prepared.request_fingerprint,
    }
    try:
        response = await runtime.validate_plan_only_allocation(request)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        response = _cartesian_unavailable_response(
            f"Exact live Cartesian validation is unavailable: {type(exc).__name__}: {exc}"
        )
    validated_response = _validated_cartesian_response(response)
    payload: dict[str, object] = {
        "schema_version": 3,
        "record_type": "PlanOnlyFeasibilityValidationRecord",
        "validation_number": validation_number,
        "validator_authority": prepared.resource_jid,
        "process_symbol": prepared.process_symbol,
        "process_iri": prepared.process_iri,
        "feature_iri": prepared.feature_iri,
        "current_state_iri": prepared.current_state.state_iri,
        "desired_state_iri": prepared.desired_state.state_iri,
        "resource_symbol": prepared.resource_symbol,
        "resource_iri": prepared.resource_iri,
        "resource_jid": prepared.resource_jid,
        "execution_mode": prepared.execution_mode,
        "motion_mode": "cartesian_pick_place",
        "moveit_group": controller.moveit_group,
        "end_effector_link": controller.end_effector_link,
        "tcp_link": controller.tcp_link,
        "target_frame": controller.target_frame,
        "cartesian_path_service": controller.cartesian_path_service,
        "validation_scope": "cartesian_pick_place",
        "checked_constraints": list(_CARTESIAN_CHECKED_CONSTRAINTS),
        "unvalidated_constraints": list(unvalidated_constraints),
        "cartesian_parameters": dict(_CARTESIAN_PARAMETERS),
        "live_start_pose": validated_response["live_start_pose"],
        "ee_to_tcp_transform": validated_response["ee_to_tcp_transform"],
        "waypoints": validated_response["waypoints"],
        "phases": validated_response["phases"],
        "mode": "plan_only",
        "motion_executed": False,
        "status": validated_response["status"],
        "feedback": validated_response["feedback"],
        "validated_at_ns": time.time_ns(),
        "request_fingerprint": prepared.request_fingerprint,
    }
    payload["fingerprint"] = _fingerprint(payload)
    destination = (
        root
        / "resources"
        / prepared.resource_jid
        / "validation"
        / f"plan_only_validation_{validation_number:04d}"
    )
    record_path = destination / "plan_only_feasibility_validation_record.json"
    _persist_record(destination, record_path.name, payload)
    return PlanOnlyFeasibilityValidation(
        record_path=record_path,
        record_ref=record_path.relative_to(root).as_posix(),
        validation_number=validation_number,
        process_symbol=prepared.process_symbol,
        process_iri=prepared.process_iri,
        feature_iri=prepared.feature_iri,
        current_state_iri=prepared.current_state.state_iri,
        desired_state_iri=prepared.desired_state.state_iri,
        resource_symbol=prepared.resource_symbol,
        resource_iri=prepared.resource_iri,
        resource_jid=prepared.resource_jid,
        execution_mode=prepared.execution_mode,
        moveit_group=controller.moveit_group,
        end_effector_link=controller.end_effector_link,
        target_frame=controller.target_frame,
        validation_scope="cartesian_pick_place",
        checked_constraints=_CARTESIAN_CHECKED_CONSTRAINTS,
        unvalidated_constraints=unvalidated_constraints,
        status=str(validated_response["status"]),
        feedback=(
            str(validated_response["feedback"])
            if validated_response["feedback"] is not None
            else None
        ),
        request_fingerprint=prepared.request_fingerprint,
        fingerprint=str(payload["fingerprint"]),
        schema_version=3,
        motion_mode="cartesian_pick_place",
        tcp_link=controller.tcp_link,
        cartesian_path_service=controller.cartesian_path_service,
    )


def _cartesian_request_state(state: CartesianStateEvidence) -> dict[str, object]:
    return {
        "state_iri": state.state_iri,
        "evidence_handle": state.evidence_handle,
        "translation_m": list(state.translation_m),
        "location_record_ref": state.location_record_ref,
        "location_record_sha256": state.location_record_sha256,
    }


def _cartesian_unvalidated_constraints(process_symbol: str) -> tuple[str, ...]:
    if not isinstance(process_symbol, str) or not process_symbol:
        raise RobotAgentFeasibilityError("Selected process symbol is unavailable.")
    return (
        "grasp_contact",
        "gripper_actuation",
        "attached_part_collision_geometry",
        f"{process_symbol}_tolerance",
        "force_control",
        "final_constrained_insertion_stroke",
    )


def _validated_cartesian_response(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _CARTESIAN_RESPONSE_KEYS:
        raise RobotAgentFeasibilityError(
            "RobotAgent Cartesian validation response fields are invalid."
        )
    if value.get("motion_executed") is not False:
        raise RobotAgentFeasibilityError("RobotAgent Cartesian validation may not execute motion.")
    status = value.get("status")
    if status not in _VALIDATION_STATUSES:
        raise RobotAgentFeasibilityError("RobotAgent Cartesian validation status is invalid.")
    phases = value.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != {"pick", "place"}:
        raise RobotAgentFeasibilityError("RobotAgent Cartesian phase results are invalid.")
    phase_statuses: list[str] = []
    expected_roles = {
        "pick": ["pick_approach", "grasp", "pick_retreat"],
        "place": ["transfer", "place_approach", "placement", "place_retreat"],
    }
    validated_phases: dict[str, dict[str, object]] = {}
    for phase in ("pick", "place"):
        result = phases.get(phase)
        if not isinstance(result, Mapping) or set(result) != {
            "phase",
            "status",
            "waypoint_roles",
            "fraction",
            "moveit_error_code",
            "terminal_state_available",
            "message",
        }:
            raise RobotAgentFeasibilityError(f"RobotAgent Cartesian {phase} result is invalid.")
        phase_status = result.get("status")
        fraction = result.get("fraction")
        error_code = result.get("moveit_error_code")
        if (
            result.get("phase") != phase
            or result.get("waypoint_roles") != expected_roles[phase]
            or phase_status not in _VALIDATION_STATUSES
            or not isinstance(result.get("message"), str)
            or not str(result.get("message")).strip()
            or not isinstance(result.get("terminal_state_available"), bool)
            or (
                fraction is not None
                and (
                    isinstance(fraction, bool)
                    or not isinstance(fraction, (int, float))
                    or not math.isfinite(float(fraction))
                    or not 0.0 <= float(fraction) <= 1.0
                )
            )
            or (
                error_code is not None
                and (isinstance(error_code, bool) or not isinstance(error_code, int))
            )
        ):
            raise RobotAgentFeasibilityError(f"RobotAgent Cartesian {phase} result is invalid.")
        if phase_status == "accepted" and (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or float(fraction) < 0.999
            or error_code != 1
            or result.get("terminal_state_available") is not True
        ):
            raise RobotAgentFeasibilityError(
                f"RobotAgent accepted an incomplete Cartesian {phase} path."
            )
        phase_statuses.append(str(phase_status))
        validated_phases[phase] = dict(result)
    if phase_statuses[0] != "accepted" and phase_statuses[1] != "needs_context":
        raise RobotAgentFeasibilityError(
            "RobotAgent Cartesian place phase was not chained from an accepted pick."
        )
    expected_status = (
        "rejected"
        if "rejected" in phase_statuses
        else "needs_context"
        if "needs_context" in phase_statuses
        else "accepted"
    )
    feedback = value.get("feedback")
    if (
        status != expected_status
        or (status == "accepted" and feedback is not None)
        or (status != "accepted" and (not isinstance(feedback, str) or not feedback.strip()))
    ):
        raise RobotAgentFeasibilityError("RobotAgent Cartesian aggregate verdict is inconsistent.")
    live_start_pose = value.get("live_start_pose")
    ee_to_tcp = value.get("ee_to_tcp_transform")
    waypoints = value.get("waypoints")
    if isinstance(waypoints, list) and waypoints:
        _validated_live_pose(live_start_pose)
        _validated_ee_to_tcp(ee_to_tcp)
        _validated_waypoints(waypoints)
    elif (
        not isinstance(waypoints, list)
        or status == "accepted"
        or live_start_pose is not None
        or ee_to_tcp is not None
    ):
        raise RobotAgentFeasibilityError("RobotAgent Cartesian waypoint evidence is invalid.")
    return {
        "status": status,
        "live_start_pose": (
            dict(live_start_pose) if isinstance(live_start_pose, Mapping) else None
        ),
        "ee_to_tcp_transform": (dict(ee_to_tcp) if isinstance(ee_to_tcp, Mapping) else None),
        "waypoints": [dict(item) for item in waypoints],
        "phases": validated_phases,
        "feedback": feedback,
    }


def _validated_live_pose(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "frame_id",
        "link_name",
        "position_m",
        "orientation_xyzw",
    }:
        raise RobotAgentFeasibilityError("Live start pose is invalid.")
    _finite_sequence(value.get("position_m"), 3, "live start position")
    _normalized_quaternion(value.get("orientation_xyzw"), "live start orientation")


def _validated_ee_to_tcp(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "parent_link",
        "child_link",
        "translation_m",
        "rotation_xyzw",
    }:
        raise RobotAgentFeasibilityError("Live EE-to-TCP transform is invalid.")
    _finite_sequence(value.get("translation_m"), 3, "EE-to-TCP translation")
    _normalized_quaternion(value.get("rotation_xyzw"), "EE-to-TCP rotation")


def _validated_waypoints(value: object) -> None:
    if not isinstance(value, list) or len(value) != 7:
        raise RobotAgentFeasibilityError("Cartesian waypoint list is invalid.")
    expected = [
        ("pick", "pick_approach"),
        ("pick", "grasp"),
        ("pick", "pick_retreat"),
        ("place", "transfer"),
        ("place", "place_approach"),
        ("place", "placement"),
        ("place", "place_retreat"),
    ]
    for item, (phase, role) in zip(value, expected, strict=True):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"phase", "role", "pose"}
            or item.get("phase") != phase
            or item.get("role") != role
        ):
            raise RobotAgentFeasibilityError("Cartesian waypoint order is invalid.")
        pose = item.get("pose")
        if not isinstance(pose, Mapping) or set(pose) != {
            "position_m",
            "orientation_xyzw",
        }:
            raise RobotAgentFeasibilityError("Cartesian waypoint pose is invalid.")
        _finite_sequence(pose.get("position_m"), 3, "waypoint position")
        _normalized_quaternion(pose.get("orientation_xyzw"), "waypoint orientation")


def _finite_sequence(value: object, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise RobotAgentFeasibilityError(f"{label} is invalid.")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RobotAgentFeasibilityError(f"{label} is invalid.") from exc
    if not all(math.isfinite(item) for item in result):
        raise RobotAgentFeasibilityError(f"{label} is invalid.")
    return result


def _normalized_quaternion(value: object, label: str) -> tuple[float, ...]:
    quaternion = _finite_sequence(value, 4, label)
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm <= 1e-12 or abs(norm - 1.0) > 1e-5:
        raise RobotAgentFeasibilityError(f"{label} is invalid.")
    return quaternion


def _cartesian_unavailable_response(message: str) -> Mapping[str, object]:
    def phase(name: str, roles: list[str]) -> dict[str, object]:
        return {
            "phase": name,
            "status": "needs_context",
            "waypoint_roles": roles,
            "fraction": None,
            "moveit_error_code": None,
            "terminal_state_available": False,
            "message": message,
        }

    return {
        "status": "needs_context",
        "live_start_pose": None,
        "ee_to_tcp_transform": None,
        "waypoints": [],
        "phases": {
            "pick": phase("pick", ["pick_approach", "grasp", "pick_retreat"]),
            "place": phase(
                "place",
                ["transfer", "place_approach", "placement", "place_retreat"],
            ),
        },
        "feedback": message,
        "motion_executed": False,
    }


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
