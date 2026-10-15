from __future__ import annotations

"""Explicit, non-executing ROS preview acquisition for recovery outline evidence.

Importing this module does not import ROS or contact any live interface. Offline
validation may use the explicit file backend; live entrypoints remain separate.
"""

import asyncio
import copy
import hashlib
import json
import math
import struct
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SceneObjectBindingError(ValueError):
    """Identify the recorded sources needed to resolve a scene binding."""

    def __init__(self, message: str, context_requests: list[str]) -> None:
        """Keep the original validation error and exact recorded references.

        Args:
            message: Failure reported by the recorded scene checks.
            context_requests: Source references needed to resolve the failure.
        """
        super().__init__(message)
        self.context_requests = context_requests


_FINGERPRINTS = (
    "resource_id", "capture_fingerprint", "proposal_fingerprint",
    "scene_fingerprint", "model_fingerprint",
)
_SETTINGS = (
    "max_translation_step", "max_rotation_step", "max_joint_step",
    "velocity_scaling", "acceleration_scaling", "timeout_seconds",
)
_REQUEST_FIELDS = frozenset((
    "planning_scene", "start_state", "group_name", "link_name", "frame_id",
    "mode", "waypoints", "joint_target", "recorded_root_transforms",
    "support_departure_json", "support_departure_fingerprint", "route_search_json",
    "mating_contact_json", "mating_contact_fingerprint", *_FINGERPRINTS, *_SETTINGS,
    "scene_endpoint_json", "scene_endpoint_fingerprint",
    "base_trajectory", "base_link",
))

_ROUTE_ALGORITHM = "Cartesian RRT extension with greedy goal connections"
_ROUTE_RECORDS = Path(__file__).resolve().parents[1] / "monitor/environment_runs/recovery/route_calculations"


def _route_search_record(request: Mapping[str, Any]) -> dict | None:
    text = request.get("route_search_json", "")
    if not isinstance(text, str):
        raise ValueError("route_search_json must be a string")
    if not text:
        return None
    config = json.loads(text)
    keys = {"version", "algorithm", "waypoint_index", "maximum_edges", "maximum_seconds",
            "random_seed", "gripper_reach", "static_capabilities_fingerprint"}
    if (not isinstance(config, dict) or set(config) != keys or config["version"] != 1
            or config["algorithm"] != _ROUTE_ALGORITHM or config["waypoint_index"] != 0
            or request.get("mode") != "cartesian" or request.get("support_departure_json")
            or request.get("mating_contact_json")):
        raise ValueError("Unsupported Cartesian route calculation contract")
    if (type(config["maximum_edges"]) is not int or not 1 <= config["maximum_edges"] <= 128
            or type(config["random_seed"]) is not int or not 0 <= config["random_seed"] <= 2**32 - 1
            or type(config["maximum_seconds"]) not in (float, int)
            or not math.isfinite(config["maximum_seconds"]) or not 0 < config["maximum_seconds"] <= 120
            or not isinstance(config["gripper_reach"], dict)
            or not isinstance(config["static_capabilities_fingerprint"], str)
            or not config["static_capabilities_fingerprint"]):
        raise ValueError("Cartesian route calculation requires bounded recorded inputs")
    return config


def prepare_cartesian_route_search(request: Mapping[str, Any], static_capabilities: dict) -> dict:
    """Prepare bounded model calculation without changing any motion setting."""
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    result = copy.deepcopy(dict(request))
    result["route_search_json"] = json.dumps({"version": 1, "algorithm": _ROUTE_ALGORITHM,
        "waypoint_index": 0, "maximum_edges": 128, "maximum_seconds": min(120., request["timeout_seconds"]),
        "random_seed": 42, "gripper_reach": static_capabilities.get("gripper_reach"),
        "static_capabilities_fingerprint": recovery_validation_fingerprint(static_capabilities)},
        sort_keys=True, separators=(",", ":"), allow_nan=False)
    return validate_preview_request(result)


def _route_candidate_record(request: Mapping[str, Any], result: Mapping[str, Any]) -> dict | None:
    config = _route_search_record(request)
    text = result.get("route_candidate_json", "")
    if config is None:
        if text or result.get("status") == "ROUTE_CANDIDATE":
            raise ValueError("Unexpected calculated route without a recorded search request")
        return None
    if (result.get("path_complete") is not False or result.get("status") == "PREVIEW_COMPLETE"
            or result.get("trajectory", {}).get("joint_trajectory", {}).get("points")):
        raise ValueError("A route calculation cannot provide approved motion evidence")
    if not text:
        if result.get("status") == "ROUTE_CANDIDATE":
            raise ValueError("Calculated route omitted its source evidence")
        return None
    candidate = json.loads(text)
    if (not isinstance(candidate, dict) or candidate.get("version") != 1
            or candidate.get("request_binding") != {field: request[field] for field in _FINGERPRINTS}
            or candidate.get("route_search") != config
            or candidate.get("original_waypoints") != request["waypoints"]
            or candidate.get("command_sent") is not False or candidate.get("execution_authorized") is not False):
        raise ValueError("Calculated route has changed source inputs")
    if not candidate.get("solved"):
        if result.get("status") == "ROUTE_CANDIDATE":
            raise ValueError("Unresolved route cannot be a calculated candidate")
        return None
    if result.get("status") != "ROUTE_CANDIDATE":
        raise ValueError("Solved route has an incompatible result status")
    source = candidate.get("source_state", {}).get("joint_state", {})
    names, positions = source.get("name", []), source.get("position", [])
    start = request["start_state"]["joint_state"]
    if (len(names) != len(positions) or len(set(names)) != len(names)
            or dict(zip(names, positions)) != dict(zip(start["name"], start["position"]))):
        raise ValueError("Calculated route starts from different recorded joints")
    waypoints, indices = candidate.get("derived_waypoints"), candidate.get("mandatory_waypoint_indices")
    if (not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 1024
            or not isinstance(indices, list) or len(indices) != len(request["waypoints"])
            or any(type(i) is not int or not 0 <= i < len(waypoints) for i in indices)
            or sorted(set(indices)) != indices or indices[-1] != len(waypoints) - 1
            or indices != list(range(len(waypoints) - len(request["waypoints"]), len(waypoints)))
            or any(waypoints[index] != original for index, original in zip(indices, request["waypoints"], strict=True))):
        raise ValueError("Calculated route changes mandatory Cartesian endpoints or their order")
    orientation = request["waypoints"][0]["orientation"]
    for waypoint in [*request["waypoints"], *waypoints]:
        if (not isinstance(waypoint, dict) or waypoint.get("orientation") != orientation
                or set(waypoint.get("position", {})) != {"x", "y", "z"}
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in waypoint["position"].values())):
            raise ValueError("Calculated route changes the recorded orientation or has an invalid target")
    from cais_spade_llm.resources.robot.recovery_capability import check_outline_motion_capabilities

    checked = check_outline_motion_capabilities(motion_requests=[{"mode": "cartesian", "waypoints": waypoints}],
        static_capabilities={"gripper_reach": config["gripper_reach"]}, frame=request["frame_id"],
        resource_jid=request["resource_id"])
    if checked["feasibility_status"] != "FEASIBLE":
        raise ValueError("Calculated route exceeds its captured capability: " + str(checked.get("reason")))
    return {key: copy.deepcopy(candidate[key]) for key in ("version", "request_binding", "route_search",
        "original_waypoints", "derived_waypoints", "mandatory_waypoint_indices", "source_state",
        "solved", "command_sent", "execution_authorized")}


def archive_cartesian_route(request: dict, result: dict, *, previous_request: dict | None = None,
                            previous_result: dict | None = None) -> dict:
    """Retain raw calculation once; return a compact, independently readable proof."""
    from cais_spade_llm.recovery_framework.diagnostics import write_record
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    checked = preview_result_record(request, result)
    directory = _ROUTE_RECORDS / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "calculation.json"
    write_record(path, {"request": request, "result": result,
        "previous_request": previous_request, "previous_result": previous_result})
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"version": 1, "source": "resource-owned Cartesian route calculation",
        "request_fingerprint": recovery_validation_fingerprint(request),
        "response_fingerprint": recovery_validation_fingerprint(result),
        "candidate": checked.get("route_candidate"), "status": checked.get("status"), "reason": checked.get("reason"),
        "artifact": {"path": str(path), "sha256": digest.hexdigest()},
        "requires_composition_validation": True, "command_sent": False, "execution_authorized": False}


def read_cartesian_route(proof: dict) -> dict:
    """Verify saved calculation provenance using files only, without ROS or controllers."""
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    if (proof.get("version") != 1 or proof.get("source") != "resource-owned Cartesian route calculation"
            or proof.get("command_sent") is not False or proof.get("execution_authorized") is not False
            or proof.get("requires_composition_validation") is not True):
        raise ValueError("Invalid saved Cartesian route proof")
    artifact = proof.get("artifact") or {}
    path = Path(artifact.get("path", "")).resolve()
    if not path.is_relative_to(_ROUTE_RECORDS.resolve()) or not path.is_file():
        raise ValueError("Saved Cartesian route artifact is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != artifact.get("sha256"):
        raise ValueError("Saved Cartesian route artifact changed")
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    current_digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            current_digest.update(chunk)
    if current_digest.hexdigest() != artifact["sha256"]:
        raise ValueError("Saved Cartesian route artifact changed during validation")
    request, response = validate_preview_request(raw["request"]), raw["result"]
    if (recovery_validation_fingerprint(request) != proof.get("request_fingerprint")
            or recovery_validation_fingerprint(response) != proof.get("response_fingerprint")
            or preview_result_record(request, response).get("route_candidate") != proof.get("candidate")):
        raise ValueError("Saved Cartesian route differs from its captured calculation")
    return request


def bind_support_departure(request: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Bind recorded support contact to one complete private preview request."""
    result, evidence = copy.deepcopy(dict(request)), copy.deepcopy(dict(record))
    evidence["preview_binding"] = {field: result[field] for field in _FINGERPRINTS}
    result["support_departure_json"] = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["support_departure_fingerprint"] = hashlib.sha256(result["support_departure_json"].encode()).hexdigest()
    return validate_preview_request(result)


def _support_departure_record(request: Mapping[str, Any]) -> dict[str, Any] | None:
    text = request.get("support_departure_json", "")
    fingerprint = request.get("support_departure_fingerprint", "")
    if not isinstance(text, str) or not isinstance(fingerprint, str):
        raise ValueError("Support departure record and fingerprint must be strings")
    if not text:
        if fingerprint:
            raise ValueError("Support departure fingerprint has no record")
        return None
    if len(text.encode()) > 16 * 1024 * 1024 or hashlib.sha256(text.encode()).hexdigest() != fingerprint:
        raise ValueError("Support departure record fingerprint differs")
    evidence = json.loads(text)
    if (not isinstance(evidence, dict) or evidence.get("version") != 1 or evidence.get("status") != "prepared"
            or evidence.get("command_sent") is not False or evidence.get("execution_authorized") is not False):
        raise ValueError("Support departure requires prepared non-executing evidence")
    if evidence.get("resource_id") != request["resource_id"] or evidence.get("preview_binding") != {
            field: request[field] for field in _FINGERPRINTS}:
        raise ValueError("Support departure has stale request bindings")
    pairs = evidence.get("pairs")
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 32:
        raise ValueError("Support departure requires between one and 32 exact collision pairs")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("Support departure collision pair must be a record")
        for field in ("part_object_id", "support_object_id", "part_collision_id", "support_collision_id",
                      "source_contact_fingerprint", "geometry_fingerprint"):
            if not isinstance(pair.get(field), str) or not pair[field]:
                raise ValueError("Support departure source identity is missing: " + field)
        identity = (pair["part_object_id"], pair["support_object_id"])
        if identity in seen or identity[0] == identity[1]:
            raise ValueError("Support departure collision pair is invalid or duplicated")
        seen.add(identity)
        for field in ("plane_point", "plane_normal"):
            values = pair.get(field)
            if not isinstance(values, list) or len(values) != 3 or any(
                    type(value) not in (int, float) or not math.isfinite(value) for value in values):
                raise ValueError("Support departure geometry must contain three finite coordinates")
        if not math.isclose(sum(value * value for value in pair["plane_normal"]), 1., rel_tol=0., abs_tol=1e-12):
            raise ValueError("Support departure plane normal must be a unit vector")
        value = pair.get("initial_penetration_bound")
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Support departure initial penetration must be finite and nonnegative")
    return evidence


def bind_mating_contact(request: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a captured owner mating declaration to one private insertion stage."""
    result, evidence = copy.deepcopy(dict(request)), copy.deepcopy(dict(record))
    evidence["preview_binding"] = {field: result[field] for field in _FINGERPRINTS}
    result["mating_contact_json"] = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["mating_contact_fingerprint"] = hashlib.sha256(result["mating_contact_json"].encode()).hexdigest()
    return validate_preview_request(result)


def _mating_contact_record(request: Mapping[str, Any]) -> dict | None:
    from cais_spade_llm.resources.primitive_observations_3d import _unit_pose

    text, fingerprint = request.get("mating_contact_json", ""), request.get("mating_contact_fingerprint", "")
    if not isinstance(text, str) or not isinstance(fingerprint, str):
        raise ValueError("Mating contact record and fingerprint must be strings")
    if not text:
        if fingerprint:
            raise ValueError("Mating contact fingerprint has no record")
        return None
    if (request.get("mode") != "cartesian" or request.get("support_departure_json")
            or request.get("route_search_json")):
        raise ValueError("Mating contact requires one ordinary aligned Cartesian stage")
    if len(text.encode()) > 16 * 1024 * 1024 or hashlib.sha256(text.encode()).hexdigest() != fingerprint:
        raise ValueError("Mating contact record fingerprint differs")
    evidence = json.loads(text)
    if (not isinstance(evidence, dict) or evidence.get("version") != 1 or evidence.get("status") != "prepared"
            or evidence.get("command_sent") is not False or evidence.get("execution_authorized") is not False):
        raise ValueError("Mating contact requires prepared non-executing evidence")
    if (evidence.get("preview_binding") != {field: request[field] for field in _FINGERPRINTS}
            or evidence.get("resource_id") != request["resource_id"]
            or evidence.get("reference_link") != request["link_name"] or evidence.get("frame_id") != request["frame_id"]):
        raise ValueError("Mating contact has stale request bindings")
    for field in ("model_name", "part_name", "part_object_id"):
        if not isinstance(evidence.get(field), str) or not evidence[field]:
            raise ValueError("Mating contact has missing identity: " + field)
    if evidence["part_object_id"] != evidence["model_name"] + "/link/collision":
        raise ValueError("Mating part object differs from the owner declaration")
    binding = evidence.get("binding") or {}
    for field in ("state_fingerprint", "model_fingerprint", "owner_evidence_fingerprint", "physics_fingerprint",
                  "scene_object_bindings_fingerprint", "peer_fingerprint"):
        if not isinstance(binding.get(field), str) or not binding[field]:
            raise ValueError("Mating contact source fingerprint is missing: " + field)
    policy = evidence.get("simulation_mating_contact") or {}
    for field in ("axis_tolerance_m", "max_contact_depth_m", "target_yaw_rad", "yaw_tolerance_rad"):
        value = policy.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or (field != "target_yaw_rad" and value < 0):
            raise ValueError("Mating contact requires recorded finite policy limits")
    target = policy.get("target_collision_object")
    ignored = policy.get("ignore_tooth_contact_with", [])
    if (not isinstance(target, str) or not target or target == evidence["part_object_id"]
            or not isinstance(ignored, list) or any(not isinstance(value, str) or not value for value in ignored)
            or len(set(ignored)) != len(ignored) or {target, evidence["part_object_id"]}.intersection(ignored)):
        raise ValueError("Mating contact collision identities are invalid")
    for field in ("start_part_pose", "start_tool_pose", "target_tool_pose", "tool_to_part_pose"):
        _unit_pose(evidence.get(field), "mating " + field)
    target_pose = evidence.get("target_origin_pose") or {}
    if any(type(target_pose.get(axis)) not in (int, float) or not math.isfinite(target_pose[axis]) for axis in "xyz"):
        raise ValueError("Mating contact requires an exact target origin")
    if type(evidence.get("start_part_z")) not in (int, float) or evidence["start_part_z"] != evidence["start_part_pose"][2]:
        raise ValueError("Mating contact start height differs")
    return evidence


def _mating_contact_result(request: Mapping[str, Any], result: Mapping[str, Any], trajectory: dict) -> dict | None:
    from cais_spade_llm.resources.recovery_mating_contact import validate_mating_contact_samples

    prepared = _mating_contact_record(request)
    text = result.get("mating_contact_evidence_json", "")
    if prepared is None:
        if text or result.get("mating_contact_fingerprint"):
            raise ValueError("Unexpected mating contact evidence without its captured request")
        return None
    if result.get("mating_contact_fingerprint") != request["mating_contact_fingerprint"]:
        raise ValueError("Stale preview result: mating_contact_fingerprint differs")
    audit = json.loads(text) if text else None
    if audit is not None:
        binding = {field: request[field] for field in _FINGERPRINTS}
        binding.update(support_departure_fingerprint="", mating_contact_fingerprint=request["mating_contact_fingerprint"])
        if (not isinstance(audit, dict) or audit.get("version") != 1 or audit.get("source") != "PreviewRecoveryMotion"
                or audit.get("request_binding") != binding or audit.get("pairs_complete") is not True
                or audit.get("continuous_collision_verified") is not False):
            raise ValueError("Mating contact samples have incomplete or stale provenance")
    if result.get("status") == "PREVIEW_COMPLETE":
        validate_mating_contact_samples({"mating_contact_json": request["mating_contact_json"],
            "mating_contact_fingerprint": request["mating_contact_fingerprint"],
            "native_mating_contact_evidence": audit, "joint_trajectory": trajectory})
        start = request["start_state"]["joint_state"]
        peer_names = set(start["name"]) - set(trajectory["joint_names"])
        original = dict(zip(start["name"], start["position"], strict=True))
        for sample in audit["samples"]:
            joints = sample["joint_state"]
            if (sample.get("phase") != "trajectory" or set(joints["name"]) != set(start["name"])
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in joints["position"])):
                raise ValueError("Mating contact samples omitted complete checked joints")
            values = dict(zip(joints["name"], joints["position"], strict=True))
            if any(values[name] != original[name] for name in peer_names):
                raise ValueError("Mating contact samples changed a peer joint")
    return audit


def bind_scene_endpoint(request: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a non-moving deposited-object check to its exact private scene."""
    result, evidence = copy.deepcopy(dict(request)), copy.deepcopy(dict(record))
    result.update(mode="scene_endpoint", waypoints=[], joint_target={})
    for key in ("support_departure_json", "support_departure_fingerprint", "mating_contact_json",
                "mating_contact_fingerprint", "route_search_json"):
        result[key] = ""
    evidence["preview_binding"] = {field: result[field] for field in _FINGERPRINTS}
    result["scene_endpoint_json"] = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["scene_endpoint_fingerprint"] = hashlib.sha256(result["scene_endpoint_json"].encode()).hexdigest()
    return validate_preview_request(result)


def _scene_endpoint_record(request: Mapping[str, Any]) -> dict | None:
    text, fingerprint = request.get("scene_endpoint_json", ""), request.get("scene_endpoint_fingerprint", "")
    if not text:
        if fingerprint or request.get("mode") == "scene_endpoint":
            raise ValueError("Endpoint checking requires its bound record")
        return None
    if (not isinstance(text, str) or len(text.encode()) > 16 * 1024 * 1024
            or hashlib.sha256(text.encode()).hexdigest() != fingerprint):
        raise ValueError("Endpoint record fingerprint differs")
    evidence = json.loads(text)
    if (request.get("mode") != "scene_endpoint" or request.get("waypoints")
            or request.get("joint_target", {}).get("name")
            or any(request.get(key) for key in ("support_departure_json", "support_departure_fingerprint",
                                                "mating_contact_json", "mating_contact_fingerprint", "route_search_json"))):
        raise ValueError("Endpoint checking cannot include motion or contact policies")
    if (not isinstance(evidence, dict) or evidence.get("version") != 1 or evidence.get("status") != "prepared"
            or evidence.get("command_sent") is not False or evidence.get("execution_authorized") is not False
            or evidence.get("resource_id") != request["resource_id"]
            or evidence.get("preview_binding") != {field: request[field] for field in _FINGERPRINTS}
            or not isinstance(evidence.get("source"), dict) or not evidence["source"]):
        raise ValueError("Endpoint evidence is incomplete or has stale request bindings")
    changed = evidence.get("changed_object_ids")
    world = request["planning_scene"].get("world", {}).get("collision_objects", [])
    world_ids = [row.get("id") for row in world]
    attached = {row.get("object", {}).get("id") for row in request["start_state"].get("attached_collision_objects", [])}
    if (not isinstance(changed, list) or not 1 <= len(changed) <= 32
            or any(not isinstance(name, str) or not name for name in changed)
            or len(changed) != len(set(changed)) or len(world_ids) != len(set(world_ids))
            or any(name not in world_ids or name in attached for name in changed)):
        raise ValueError("Endpoint changed object is missing, duplicated, or attached")
    pairs = evidence.get("support_pairs")
    if not isinstance(pairs, list) or len(pairs) > 32:
        raise ValueError("Endpoint support pairs are invalid")
    identities = set()
    for pair in pairs:
        identity = pair.get("part_object_id"), pair.get("support_object_id")
        if (identity in identities or identity[0] not in changed or identity[1] not in world_ids
                or identity[1] in changed):
            raise ValueError("Endpoint support identity is missing or conflicting")
        identities.add(identity)
        for field in ("plane_point", "plane_normal"):
            values = pair.get(field)
            if (not isinstance(values, list) or len(values) != 3
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
                raise ValueError("Endpoint support geometry requires finite coordinates")
        if abs(math.sqrt(sum(value * value for value in pair["plane_normal"])) - 1.0) > 1e-12:
            raise ValueError("Endpoint support normal is not unit length")
    return evidence


def _scene_endpoint_result(request: Mapping[str, Any], result: Mapping[str, Any]) -> dict | None:
    prepared = _scene_endpoint_record(request)
    text = result.get("scene_endpoint_evidence_json", "")
    if prepared is None:
        if text or result.get("scene_endpoint_fingerprint"):
            raise ValueError("Unexpected endpoint result")
        return None
    if result.get("scene_endpoint_fingerprint") != request["scene_endpoint_fingerprint"]:
        raise ValueError("Stale preview result: scene_endpoint_fingerprint differs")
    checked = result.get("status") in {"SCENE_ENDPOINT_COMPLETE", "SCENE_ENDPOINT_REJECTED"}
    if not checked:
        if text:
            raise ValueError("Incomplete endpoint result cannot carry a completed audit")
        return None
    if (result.get("path_complete") is not False or result.get("fraction") != 0.0
            or result.get("sampled_collision_rejected") is not False
            or result.get("trajectory", {}).get("joint_trajectory", {}).get("points")
            or result.get("trajectory", {}).get("multi_dof_joint_trajectory", {}).get("points")):
        raise ValueError("Endpoint evidence cannot claim a motion result")
    audit = json.loads(text) if text else None
    if (not isinstance(audit, dict) or audit.get("version") != 1 or audit.get("source") != "PreviewRecoveryMotion"
            or audit.get("endpoint_only") is not True or audit.get("pairs_complete") is not True
            or audit.get("continuous_collision_verified") is not False
            or audit.get("request_binding") != {field: request[field] for field in _FINGERPRINTS}
            or audit.get("scene_endpoint_fingerprint") != request["scene_endpoint_fingerprint"]
            or audit.get("changed_object_ids") != prepared["changed_object_ids"]
            or audit.get("robot_collision_checked") is not True):
        raise ValueError("Endpoint audit has missing or stale evidence")
    world = {row["id"]: row for row in request["planning_scene"]["world"]["collision_objects"]}
    if audit.get("world_object_ids") != sorted(world):
        raise ValueError("Endpoint audit omitted world objects")
    supports = {(row["part_object_id"], row["support_object_id"]): row for row in prepared["support_pairs"]}
    actual_supports = audit.get("support_pairs", [])
    if len(actual_supports) != len(supports):
        raise ValueError("Endpoint audit omitted support evidence")
    support_seen = set()
    clear = audit.get("robot_collision_free") is True
    contacts = audit.get("robot_contacts")
    if not isinstance(contacts, list) or len(contacts) >= 4096 or clear != (not contacts):
        raise ValueError("Endpoint robot contacts are inconsistent")
    checked_contacts = len(contacts)
    for row in actual_supports:
        identity = row.get("part_object_id"), row.get("support_object_id")
        distance = row.get("minimum_signed_distance")
        if (identity in support_seen or identity not in supports
                or {key: value for key, value in row.items() if key != "minimum_signed_distance"} != supports[identity]
                or type(distance) not in (float, int) or not math.isfinite(distance)):
            raise ValueError("Endpoint support evidence differs from its declaration")
        support_seen.add(identity)
        clear = clear and distance >= 0.0
    expected_pairs = {tuple(sorted((part, other))) for part in prepared["changed_object_ids"] for other in world if part != other}
    seen = set()
    pairs = audit.get("checked_pairs")
    if not isinstance(pairs, list):
        raise ValueError("Endpoint audit omitted world pairs")
    for row in pairs:
        first, second = row.get("body_1"), row.get("body_2")
        if not isinstance(first, str) or not isinstance(second, str):
            raise ValueError("Endpoint audit has invalid world pair identities")
        identity = tuple(sorted((first, second)))
        if identity not in expected_pairs or identity in seen:
            raise ValueError("Endpoint audit has conflicting world pairs")
        seen.add(identity)
        shape_counts = [sum(len(world[name].get(key, [])) for key in ("primitives", "meshes", "planes")) for name in (first, second)]
        if not all(shape_counts) or row.get("shape_pairs") != shape_counts[0] * shape_counts[1]:
            raise ValueError("Endpoint audit omitted collision shapes")
        allowed = (first, second) in supports
        if row.get("allowed_support_tangency") is not allowed or not isinstance(row.get("contacts"), list):
            raise ValueError("Endpoint contact policy differs")
        for contact in row["contacts"]:
            depth = contact.get("depth")
            if (type(depth) not in (float, int) or not math.isfinite(depth)
                    or any(type(contact.get(key)) is not int or not 0 <= contact[key] < count
                           for key, count in zip(("shape_1", "shape_2"), shape_counts, strict=True))):
                raise ValueError("Endpoint contact evidence is invalid")
            for field in ("position", "normal"):
                values = contact.get(field)
                if (not isinstance(values, list) or len(values) != 3
                        or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
                    raise ValueError("Endpoint contact geometry is invalid")
            clear = clear and allowed and depth == 0.0
        checked_contacts += len(row["contacts"])
    if checked_contacts >= 4096 or audit.get("checked_contacts") != checked_contacts:
        raise ValueError("Endpoint contact inventory is incomplete")
    if seen != expected_pairs or audit.get("collision_free") is not clear:
        raise ValueError("Endpoint audit is incomplete or contradicts its contacts")
    if (result["status"] == "SCENE_ENDPOINT_COMPLETE") is not clear:
        raise ValueError("Endpoint status contradicts its audit")
    return audit


def validate_preview_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Check declared inputs without creating missing positions or configuration.

    Returns a detached copy. ROS and model-specific validation happen again at
    the server boundary against the supplied complete planning scene.
    """
    unknown = set(request) - _REQUEST_FIELDS
    if unknown:
        raise ValueError(f"Unsupported preview fields: {sorted(unknown)}")
    result = copy.deepcopy(dict(request))
    for name in (*_FINGERPRINTS, "group_name", "link_name", "frame_id"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ValueError(f"Missing {name}")
    scene, state = result.get("planning_scene"), result.get("start_state")
    if not isinstance(scene, dict) or not isinstance(state, dict):
        raise ValueError("A complete planning_scene and start_state are required")
    if scene.get("is_diff") is not False or state.get("is_diff") is not False:
        raise ValueError("planning_scene and start_state must explicitly have is_diff=false")
    if scene.get("robot_state") != state:
        raise ValueError("start_state differs from planning_scene.robot_state")
    roots = result.get("recorded_root_transforms", [])
    if not isinstance(roots, list):
        raise ValueError("recorded_root_transforms must be a list")
    seen = set()
    for root in roots:
        from cais_spade_llm.resources.primitive_observations_3d import _unit_pose

        child = root.get("child_frame_id")
        if not isinstance(child, str) or not child or child in seen or root.get("header", {}).get("frame_id") != result["frame_id"]:
            raise ValueError("Recorded roots require unique model links in the exact world frame")
        seen.add(child)
        transform = root.get("transform", {})
        _unit_pose([transform.get("translation", {}).get(axis) for axis in ("x", "y", "z")]
                   + [transform.get("rotation", {}).get(axis) for axis in ("x", "y", "z", "w")], "recorded root")
        recorded_root_model_fingerprint(result["model_fingerprint"], [root])
    for name in _SETTINGS:
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing finite {name}")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("velocity_scaling", "acceleration_scaling"):
        if result[name] > 1:
            raise ValueError(f"{name} must be at most 1")
    if result["timeout_seconds"] > 600:
        raise ValueError("timeout_seconds must be at most 600")
    mode = result.get("mode")
    if mode == "cartesian":
        if not result.get("waypoints") or result.get("joint_target", {}).get("name"):
            raise ValueError("Cartesian preview requires waypoints only")
    elif mode == "joint_target":
        target = result.get("joint_target", {})
        if result.get("waypoints") or not target.get("name"):
            raise ValueError("Joint preview requires joint_target only")
        if len(target["name"]) != len(target.get("position", [])):
            raise ValueError("joint_target names and positions differ in size")
    elif mode == "base_trajectory":
        _base_trajectory_record(result)
    elif mode == "scene_endpoint":
        if result.get("waypoints") or result.get("joint_target", {}).get("name"):
            raise ValueError("Endpoint checking cannot request motion")
    else:
        raise ValueError("Unsupported preview mode")
    if mode != 'base_trajectory' and (result.get('base_link') or result.get('base_trajectory', {}).get('joint_names')
                                       or result.get('base_trajectory', {}).get('points')):
        raise ValueError('Base motion evidence requires base_trajectory mode')
    _support_departure_record(result)
    _mating_contact_record(result)
    _route_search_record(result)
    _scene_endpoint_record(result)
    return result


def _base_trajectory_record(request: Mapping[str, Any]) -> dict:
    """Check cubic owner motion without inventing base velocities or timing."""
    from cais_spade_llm.resources.resource_safety_preparation import validate_joint_trajectory

    if (request.get('waypoints') or request.get('joint_target', {}).get('name')
            or any(request.get(field) for field in ('support_departure_json', 'mating_contact_json',
                                                   'route_search_json', 'scene_endpoint_json'))):
        raise ValueError('Base motion requires its own trajectory and unchanged custody')
    if not isinstance(request.get('base_link'), str) or not request['base_link']:
        raise ValueError('Base motion requires its declared base_link')
    trajectory = request.get('base_trajectory')
    if not isinstance(trajectory, dict):
        raise ValueError('Base motion requires an owner trajectory')
    header = trajectory.get('header', {})
    if (header.get('stamp', {'sec': 0, 'nanosec': 0}) != {'sec': 0, 'nanosec': 0}
            or header.get('frame_id', '') not in ('', request['frame_id'])):
        raise ValueError('Base motion must use relative time in the captured frame')
    names, points = trajectory.get('joint_names', []), trajectory.get('points', [])
    if not 1 <= len(names) <= 6 or len(points) < 2 or len(points) > 1024:
        raise ValueError('Base motion has missing or excessive joints or points')
    last = points[-1]['time_from_start']
    validate_joint_trajectory({**trajectory, 'duration_ns': last['sec'] * 1_000_000_000 + last['nanosec']}, names)
    start = request['start_state']['joint_state']
    observed = dict(zip(start['name'], start['position'], strict=True))
    if len(observed) != len(start['name']) or any(name not in observed for name in names):
        raise ValueError('Base motion joints are absent or ambiguous in the captured scene')
    if points[0]['positions'] != [observed[name] for name in names]:
        raise ValueError('Base motion starts from a different captured configuration')
    for point in points:
        if (len(point.get('velocities', [])) != len(names) or point.get('accelerations') or point.get('effort')
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in point['velocities'])):
            raise ValueError('Base motion requires finite cubic velocity data without other derivatives')
    first = points[0]['time_from_start']
    if first != {'sec': 0, 'nanosec': 0}:
        raise ValueError('Base motion must begin at zero time')
    recorded = copy.deepcopy(trajectory)
    # ROS serialization fills these optional empty arrays. Preserve every
    # supplied position, velocity and timestamp when comparing the reply.
    for point in recorded['points']:
        point.setdefault('accelerations', [])
        point.setdefault('effort', [])
    return recorded


def preview_result_record(request: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a provider result to its exact request without claiming CCA approval."""
    # Revalidation receives enriched records containing a complete private scene.
    # Keep the original key order without copying fields replaced below.
    record = copy.deepcopy({
        name: None if name in {
            "request", "joint_trajectory", "source", "coverage",
            "execution_authorized", "command_sent",
        } else value
        for name, value in dict(result).items()
        if name not in {
            "cartesian_diagnostics", "cartesian_failure_evidence", "route_candidate",
            "mating_contact_evidence", "scene_endpoint_evidence",
        }
    })
    for name in _FINGERPRINTS:
        if result.get(name) != request.get(name):
            raise ValueError(f"Stale preview result: {name} differs")
    if request.get("recorded_root_transforms") and result.get("status") in {"PREVIEW_COMPLETE", "ROUTE_CANDIDATE", "SCENE_ENDPOINT_COMPLETE", "SCENE_ENDPOINT_REJECTED"}:
        expected = recorded_root_model_fingerprint(request["model_fingerprint"], request["recorded_root_transforms"])
        if result.get("derived_model_fingerprint") != expected:
            raise ValueError("Stale preview result: derived_model_fingerprint differs")
    if result.get("continuous_collision_verified") is not False:
        raise ValueError("This preview server cannot certify continuous collision clearance")
    candidate = _route_candidate_record(request, result)
    if candidate is not None:
        record["route_candidate"] = candidate
    endpoint = _scene_endpoint_result(request, result)
    if endpoint is not None:
        record["scene_endpoint_evidence"] = endpoint
    complete = result.get("status") == "PREVIEW_COMPLETE"
    if complete and (result.get("path_complete") is not True or result.get("fraction") != 1.0):
        raise ValueError("Partial preview cannot provide complete path evidence")
    trajectory = result.get("trajectory", {})
    joint_trajectory = trajectory.get("joint_trajectory", {})
    if complete and (not joint_trajectory.get("joint_names") or not joint_trajectory.get("points")):
        raise ValueError("Completed preview omitted its trajectory")
    if complete and request.get('mode') == 'base_trajectory':
        expected = _base_trajectory_record(request)
        actual = _base_trajectory_record({**request, 'base_trajectory': joint_trajectory})
        if any(actual.get(field) != expected.get(field) for field in ('joint_names', 'points')):
            raise ValueError('Base preview changed the owner trajectory or its timing')
    support = _support_departure_record(request)
    if support is not None:
        if result.get("support_departure_fingerprint") != request["support_departure_fingerprint"]:
            raise ValueError("Stale preview result: support_departure_fingerprint differs")
        captured = result.get("support_departure_evidence_json", "")
        sampled = json.loads(captured) if captured else None
        if sampled is not None and (not isinstance(sampled, dict) or sampled.get("continuous_collision_verified") is not False):
            raise ValueError("Support departure samples cannot certify continuous collision clearance")
        if complete and (sampled is None or sampled.get("pairs_complete") is not True or not sampled.get("samples")):
            raise ValueError("Completed support departure omitted complete sampled evidence")
        if complete:
            expected = {(row["part_object_id"], row["support_object_id"]) for row in support["pairs"]}
            actual = {(row["part_object_id"], row["support_object_id"]) for row in sampled.get("pairs", [])
                      if row.get("endpoint_separated") is True}
            if actual != expected or len(sampled["pairs"]) != len(expected):
                raise ValueError("Completed support departure has missing or conflicting separation evidence")
        record["support_departure_evidence"] = copy.deepcopy(sampled)
    elif result.get("support_departure_fingerprint") or result.get("support_departure_evidence_json"):
        raise ValueError("Unexpected support departure result without recorded request evidence")
    mating_audit = _mating_contact_result(request, result, joint_trajectory)
    if mating_audit is not None:
        record["mating_contact_evidence"] = mating_audit
    rejected = _collision_rejection_record(request, result, support)
    if rejected is not None:
        record["collision_rejection_evidence"] = rejected
    diagnostic_text = result.get("cartesian_diagnostics_json", "")
    if diagnostic_text:
        diagnostic = json.loads(diagnostic_text)
        if (not isinstance(diagnostic, dict) or diagnostic.get("version") != 1
                or diagnostic.get("source") != "PreviewRecoveryMotion" or diagnostic.get("diagnostic_only") is not True
                or diagnostic.get("request_binding") != {key: request.get(key) for key in _FINGERPRINTS}
                or request.get("mode") != "cartesian"):
            raise ValueError("Cartesian diagnostics have stale or unsupported request bindings")
        for field in ("group_name", "link_name", "frame_id", "max_translation_step", "max_rotation_step", "max_joint_step"):
            if diagnostic.get(field) != request.get(field):
                raise ValueError(f"Cartesian diagnostics differ from request: {field}")
        record["cartesian_diagnostics"] = copy.deepcopy(diagnostic)
        failure = diagnostic.get("failure_kind")
        if failure:
            if complete or failure not in {"ik_search_failed", "joint_jump", "solver_unavailable", "canceled_or_deadline",
                                           "collision_evidence_incomplete", "unclassified"}:
                raise ValueError("Cartesian failure diagnostics conflict with the preview result")
            waypoints, queries = diagnostic.get("waypoints"), diagnostic.get("ik_queries")
            if (not isinstance(waypoints, list) or not waypoints or not isinstance(queries, list)
                    or diagnostic.get("fraction") != result.get("fraction")):
                raise ValueError("Cartesian failure diagnostics omitted their partial path record")
            last = waypoints[-1]
            if (not isinstance(last, dict) or last.get("index") != len(waypoints) - 1
                    or last.get("query_end") != len(queries)):
                raise ValueError("Cartesian failure diagnostics have inconsistent query indices")
            if failure == "ik_search_failed" and (not queries or queries[-1].get("success") is not False):
                raise ValueError("Cartesian IK failure omitted the actual failed solver query")
            if failure == "joint_jump" and last.get("jump_check_truncated") is not True:
                raise ValueError("Cartesian jump failure omitted its truncation record")
            record["cartesian_failure_evidence"] = copy.deepcopy(diagnostic)
    record.update({
        "source": "PreviewRecoveryMotion",
        "request": copy.deepcopy(dict(request)),
        "joint_trajectory": copy.deepcopy(joint_trajectory),
        "execution_authorized": False,
        "command_sent": False,
        "coverage": {
            "path_complete": complete,
            "continuous_collision_verified": False,
            "cca_validated": False,
        },
    })
    return record


def _collision_rejection_record(request: Mapping[str, Any], result: Mapping[str, Any], support: dict | None) -> dict | None:
    """Retain a complete failed sample, separately from continuous validation."""
    text = result.get("collision_rejection_evidence_json", "")
    if result.get("sampled_collision_rejected") is not True:
        if text:
            raise ValueError("Unexpected collision rejection evidence without a rejected sample")
        return None
    if result.get("status") == "PREVIEW_COMPLETE" or not isinstance(text, str) or not text:
        raise ValueError("Rejected collision sample lacks its exact evidence")
    evidence = json.loads(text)
    binding = {field: request[field] for field in _FINGERPRINTS}
    binding["support_departure_fingerprint"] = request.get("support_departure_fingerprint", "")
    mating = _mating_contact_record(request)
    if mating is not None:
        binding["mating_contact_fingerprint"] = request["mating_contact_fingerprint"]
    if (not isinstance(evidence, dict) or evidence.get("version") != 1
            or evidence.get("source") != "PreviewRecoveryMotion" or evidence.get("request_binding") != binding
            or evidence.get("contacts_complete") is not True
            or evidence.get("continuous_collision_verified") is not False):
        raise ValueError("Rejected collision sample has incomplete or stale provenance")
    if (evidence.get("phase") not in {"initial", "cartesian", "joint_target", "trajectory", "base_trajectory"}
            or type(evidence.get("state_index")) is not int or evidence["state_index"] < 0
            or not isinstance(evidence.get("reason"), str) or not evidence["reason"]):
        raise ValueError("Rejected collision sample omits its phase, index, or reason")
    joints = evidence.get("joint_state", {})
    names, positions = joints.get("name", []), joints.get("position", [])
    if (not names or len(names) != len(set(names)) or len(names) != len(positions)
            or set(names) != set(request["start_state"]["joint_state"]["name"])
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in positions)):
        raise ValueError("Rejected collision sample omits its checked joint configuration")
    contacts, count = evidence.get("contacts"), evidence.get("checked_contacts")
    if not isinstance(contacts, list) or type(count) is not int or not 0 <= count < 4096:
        raise ValueError("Rejected collision sample contact inventory is incomplete")
    pairs = {}
    for row in contacts:
        pair, details = (row.get("body_1"), row.get("body_2")), row.get("contacts")
        if (any(not isinstance(value, str) or not value for value in pair) or frozenset(pair) in pairs
                or not isinstance(details, list) or not 0 < len(details) < 4096):
            raise ValueError("Rejected collision sample contact pair is incomplete")
        for contact in details:
            depth = contact.get("depth")
            if type(depth) not in (int, float) or not math.isfinite(depth) or depth < 0:
                raise ValueError("Rejected collision sample contact depth is unresolved")
            for field in ("position", "normal"):
                values = contact.get(field)
                if (not isinstance(values, list) or len(values) != 3
                        or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
                    raise ValueError("Rejected collision sample contact geometry is unresolved")
        pairs[frozenset(pair)] = details
    if sum(map(len, pairs.values())) != count:
        raise ValueError("Rejected collision sample contact count differs")
    expected = {frozenset((row["part_object_id"], row["support_object_id"])): row
                for row in (support or {}).get("pairs", [])}
    supported = {}
    for row in evidence.get("support_pairs", []):
        pair = frozenset((row.get("part_object_id"), row.get("support_object_id")))
        distance = row.get("minimum_signed_distance")
        if (pair not in expected or pair in supported
                or any(row.get(field) != expected[pair][field] for field in ("part_object_id", "support_object_id"))
                or row.get("initial_penetration_bound") != expected[pair]["initial_penetration_bound"]
                or type(distance) not in (int, float) or not math.isfinite(distance)
                or type(row.get("previously_separated")) is not bool):
            raise ValueError("Rejected collision sample has conflicting support evidence")
        supported[pair] = row
    if set(supported) != set(expected):
        raise ValueError("Rejected collision sample omits its support evidence")
    mating_pairs = set()
    mating_depth = mating_pose = False
    if mating is not None:
        from cais_spade_llm.recovery_framework.part_collision import mating_pose_valid
        from cais_spade_llm.resources.primitive_observations_3d import _unit_pose

        policy = mating["simulation_mating_contact"]
        target = frozenset((mating["part_object_id"], policy["target_collision_object"]))
        mating_pairs = {target, *(frozenset((mating["part_object_id"], other))
            for other in policy.get("ignore_tooth_contact_with", []))}
        mating_depth = any(contact["depth"] > policy["max_contact_depth_m"] for contact in pairs.get(target, []))
        pose = _unit_pose(evidence.get("mating_part_pose"), "rejected mating part pose")
        mating_pose = not mating_pose_valid({**policy, "target_origin_pose": mating["target_origin_pose"],
            "start_part_z": mating["start_part_z"]}, pose)
    kind = evidence.get("kind")
    proven = (
        (kind == "mating_depth" and mating_depth) or (kind == "mating_pose" and mating_pose)
        or
        (kind == "unrelated_collision" and bool(set(pairs) - set(expected) - mating_pairs))
        or (kind == "support_depth" and any(contact["depth"] > expected[pair]["initial_penetration_bound"]
            for pair, details in pairs.items() if pair in expected for contact in details))
        or (kind == "support_distance" and any(row["minimum_signed_distance"] < -row["initial_penetration_bound"]
            for row in supported.values()))
        or (kind == "support_reentry" and any(row["previously_separated"]
            and (pair in pairs or row["minimum_signed_distance"] <= 0) for pair, row in supported.items()))
    )
    if not proven:
        raise ValueError("Rejected collision sample does not support its recorded rejection")
    return copy.deepcopy(evidence)


def recorded_root_model_fingerprint(source_model: str, roots: list[dict[str, Any]]) -> str:
    """Bind the configured model to exact TransformStamped field values."""
    def text(value):
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    lineage = source_model
    for root in roots:
        header = root["header"]
        stamp = header.get("stamp", {"sec": 0, "nanosec": 0})
        transform = root["transform"]
        values = [transform["translation"][axis] for axis in ("x", "y", "z")]
        values += [transform["rotation"][axis] for axis in ("x", "y", "z", "w")]
        data = struct.pack("<iI", stamp["sec"], stamp["nanosec"])
        data += text(header["frame_id"]) + text(root["child_frame_id"]) + struct.pack("<7d", *values)
        lineage += "\n" + hashlib.sha256(data).hexdigest()
    return hashlib.sha256(lineage.encode()).hexdigest()


def prepare_recorded_root_transforms(captured_scene: Mapping[str, Any], physical_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Retain captured kinematic roots for a private native planning model.

    Fixed mounting transforms are separate from the configured source model.
    Native validation also requires movable roots to match the complete joint
    state; these cannot be substituted into the URDF as fixed mounts.
    """
    from cais_spade_llm.resources.primitive_observations_3d import _unit_pose
    from cais_spade_llm.recovery_framework.geometry import quaternion
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint as fingerprint

    models = physical_evidence.get("resource_models", {})
    if not any("continuous_motion_configuration" in row for row in physical_evidence.get("resources", {}).values()):
        return {"recorded_root_transforms": [], "evidence": None}
    description = captured_scene["robot_description"]
    root = ET.fromstring(description)
    links = {item.get("name") for item in root.findall("link")}
    frame = physical_evidence["snapshot"]["frame"]
    transforms, sources, seen = [], {}, set()
    for rid, model in models.items():
        configuration = model.get("configuration")
        if configuration is None:
            continue
        child = configuration["root"]
        if child not in links or child == frame or child in seen:
            raise ValueError("Recorded resource root has an unresolved or duplicate model identity: " + str(child))
        seen.add(child)
        rotation = configuration.get("root_quaternion")
        if rotation is None:
            rotation = quaternion(configuration["root_rpy"])
        values = [*configuration["root_xyz"], *rotation]
        _unit_pose(values, "recorded resource root")
        physical = physical_evidence["resources"][rid]
        recorded = physical.get("continuous_motion_configuration")
        if not isinstance(recorded, dict) or any(configuration.get(key) != recorded.get(key)
                for key in ("root", "root_xyz", "root_quaternion", "root_rpy", "description_sha256")):
            raise ValueError("Resource root differs from its captured physical model: " + rid)
        transforms.append({"header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": frame},
            "child_frame_id": child, "transform": {
                "translation": dict(zip(("x", "y", "z"), values[:3], strict=True)),
                "rotation": dict(zip(("x", "y", "z", "w"), values[3:], strict=True))}})
        sources[rid] = {"configuration_fingerprint": fingerprint(configuration),
            "source": f"physical_evidence/resources/{rid}/continuous_motion_configuration",
            "native_physics_source": copy.deepcopy(physical.get("native_physics_source")),
            "root": child}
    return {"recorded_root_transforms": transforms, "evidence": {
        "source_model_fingerprint": captured_scene["model_fingerprint"],
        "derived_model_fingerprint": recorded_root_model_fingerprint(captured_scene["model_fingerprint"], transforms),
        "robot_description_sha256": hashlib.sha256(description.encode()).hexdigest(),
        "resources": sources, "recorded_root_transforms": copy.deepcopy(transforms),
        "command_sent": False, "execution_authorized": False}}


def named_pose_joint_target(semantic_description: str, *, group_name: str, named_pose: str) -> dict[str, Any]:
    """Read exact declared joint values from an SRDF group state, without defaults."""
    root = ET.fromstring(semantic_description)
    matches = [entry for entry in root.findall("group_state")
               if entry.get("group") == group_name and entry.get("name") == named_pose]
    if len(matches) != 1:
        raise ValueError("The requested named pose does not have one exact SRDF declaration")
    names, positions = [], []
    for joint in matches[0].findall("joint"):
        name = joint.get("name")
        values = joint.get("value", "").split()
        if not name or name in names or len(values) != 1:
            raise ValueError("Named pose contains missing, duplicate, or unsupported joint values")
        position = float(values[0])
        if not math.isfinite(position):
            raise ValueError("Named pose contains a non-finite joint value")
        names.append(name)
        positions.append(position)
    if not names:
        raise ValueError("Named pose has no declared joint values")
    return {"name": names, "position": positions}


async def _wait_future(node: Any, future: Any, deadline: float, cancel_event: Any = None) -> Any:
    from rclpy.executors import SingleThreadedExecutor

    executor = SingleThreadedExecutor(context=node.context)
    executor.add_node(node)
    try:
        while not future.done():
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError()
            if time.monotonic() >= deadline:
                raise TimeoutError("Recovery preview request exceeded its deadline")
            executor.spin_once(timeout_sec=0.01)
            await asyncio.sleep(0.01)
        return future.result()
    finally:
        executor.remove_node(node)
        executor.shutdown()


async def _read_service(node: Any, client: Any, request: Any, deadline: float) -> tuple[Any, int]:
    for attempt in range(1, 4):
        future = client.call_async(request)
        try:
            response = await _wait_future(node, future, min(deadline, time.monotonic() + 5.0))
            return response, attempt
        except TimeoutError:
            client.remove_pending_request(future)
            if attempt == 3 or time.monotonic() >= deadline:
                raise
    raise RuntimeError("Read-only preview request exhausted its attempts")


def _new_node() -> tuple[Any, Any]:
    import rclpy
    from rclpy.context import Context

    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node(f"recovery_preview_{uuid.uuid4().hex}", context=context)
    return node, context


async def preview_recovery_motion(
    request: Mapping[str, Any], *, execution_mode: str,
    action_name: str = "/preview_recovery_motion", timeout_seconds: float = 60.0,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Acquire a private-scene trajectory after an explicit simulation UI action.

    Args:
        request: Exact captured scene, owner targets/settings, and fingerprints.
        execution_mode: Must be ``simulation``; hardware use is rejected.
        action_name: Configured non-executing preview action endpoint.
        timeout_seconds: Client wait deadline, including connection setup.
        cancel_event: Optional asyncio-compatible event checked while waiting.

    Returns:
        An immutable-copy evidence record; never execution permission.
    """
    if execution_mode != "simulation":
        raise ValueError("Recovery motion previews are restricted to simulation")
    prepared = validate_preview_request(request)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    from cais_lab_robotics.action import PreviewRecoveryMotion
    from rclpy.action import ActionClient
    from rosidl_runtime_py.convert import message_to_ordereddict

    goal = PreviewRecoveryMotion.Goal()
    _set_recorded_message_fields(goal, prepared)
    scene_hash = _scene_message_fingerprint(goal.planning_scene)
    if scene_hash != prepared["scene_fingerprint"]:
        raise ValueError("Planning scene fingerprint differs from the captured request")
    node, context = _new_node()
    client = ActionClient(node, PreviewRecoveryMotion, action_name)
    handle = None
    goal_future = None
    result_future = None
    deadline = time.monotonic() + timeout_seconds
    try:
        while not client.server_is_ready():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Preview server unavailable: {action_name}")
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError()
            await asyncio.sleep(0.05)
        goal_future = client.send_goal_async(goal)
        handle = await _wait_future(node, goal_future, deadline, cancel_event)
        if not handle.accepted:
            raise RuntimeError("Preview server rejected the request or is already busy")
        result_future = handle.get_result_async()
        response = await _wait_future(node, result_future, deadline, cancel_event)
        record = message_to_ordereddict(response.result)
        if response.status != 4 and record.get("status") in {"PREVIEW_COMPLETE", "SCENE_ENDPOINT_COMPLETE"}:
            raise ValueError("An unsuccessful action cannot establish a complete preview")
        return preview_result_record(prepared, record)
    finally:
        # Even a cancellation during goal acceptance must settle before disposal.
        try:
            if handle is None and goal_future is not None:
                handle = await _wait_future(node, goal_future, time.monotonic() + prepared["timeout_seconds"] + 5.0)
            if handle is not None and handle.accepted and (result_future is None or not result_future.done()):
                await _wait_future(node, handle.cancel_goal_async(), time.monotonic() + 10.0)
                if result_future is None:
                    result_future = handle.get_result_async()
                await _wait_future(node, result_future, time.monotonic() + prepared["timeout_seconds"] + 5.0)
        finally:
            client.destroy()
            node.destroy_node()
            context.shutdown()


async def capture_preview_scene(
    *, execution_mode: str, planning_scene_service: str = "/get_planning_scene",
    preview_parameter_service: str = "/preview_recovery_motion/get_parameters",
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Read a complete scene and preview model identity without changing either.

    This is a live observation and must be saved as supplementary/new evidence;
    it cannot be relabelled as the outcome of an unexecuted proposed event.
    """
    if execution_mode != "simulation":
        raise ValueError("Recovery preview capture is restricted to simulation")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    from moveit_msgs.msg import PlanningSceneComponents
    from moveit_msgs.srv import GetPlanningScene
    from rcl_interfaces.srv import GetParameters, ListParameters
    from rosidl_runtime_py.convert import message_to_ordereddict

    node, context = _new_node()
    scene_client = node.create_client(GetPlanningScene, planning_scene_service)
    parameter_client = node.create_client(GetParameters, preview_parameter_service)
    list_client = node.create_client(ListParameters, preview_parameter_service.rsplit("/", 1)[0] + "/list_parameters")
    deadline = time.monotonic() + timeout_seconds
    try:
        for client in (scene_client, parameter_client, list_client):
            while not client.service_is_ready():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Planning scene or preview model service is unavailable")
                await asyncio.sleep(0.05)
        scene_request = GetPlanningScene.Request()
        for name in (
            "SCENE_SETTINGS", "ROBOT_STATE", "ROBOT_STATE_ATTACHED_OBJECTS",
            "WORLD_OBJECT_NAMES", "WORLD_OBJECT_GEOMETRY", "OCTOMAP", "TRANSFORMS",
            "ALLOWED_COLLISION_MATRIX", "LINK_PADDING_AND_SCALING", "OBJECT_COLORS",
        ):
            scene_request.components.components |= getattr(PlanningSceneComponents, name)
        scene_response, scene_attempts = await _read_service(node, scene_client, scene_request, deadline)
        parameter_request = GetParameters.Request(names=[
            "model_fingerprint", "robot_description", "robot_description_semantic",
        ])
        parameter_response, parameter_attempts = await _read_service(node, parameter_client, parameter_request, deadline)
        if len(parameter_response.values) != 3 or any(not value.string_value for value in parameter_response.values):
            raise ValueError("Preview server did not provide a model fingerprint")
        listed, list_attempts = await _read_service(node, list_client, ListParameters.Request(
            prefixes=["robot_description_kinematics", "robot_description_planning"], depth=100), deadline)
        names = sorted(listed.result.names)
        if len(names) != len(set(names)) or any(not any(name == prefix or name.startswith(prefix + ".")
                for prefix in ("robot_description_kinematics", "robot_description_planning")) for name in names):
            raise ValueError("Preview model parameter inventory is invalid")
        names = ["robot_description", "robot_description_semantic", *names]
        typed, typed_attempts = await _read_service(node, parameter_client, GetParameters.Request(names=names), deadline)
        if len(typed.values) != len(names):
            raise ValueError("Preview model parameter capture is incomplete")
        if any(message_to_ordereddict(typed.values[index]) != message_to_ordereddict(parameter_response.values[index + 1])
               for index in range(2)):
            raise ValueError("Preview model changed during capture")
        typed_parameters = {name: message_to_ordereddict(value) for name, value in zip(names, typed.values, strict=True)}
        listed_again, _ = await _read_service(node, list_client, ListParameters.Request(
            prefixes=["robot_description_kinematics", "robot_description_planning"], depth=100), deadline)
        repeated, _ = await _read_service(node, parameter_client,
            GetParameters.Request(names=["model_fingerprint", *names]), deadline)
        if (sorted(listed_again.result.names) != names[2:] or len(repeated.values) != len(names) + 1
                or repeated.values[0].string_value != parameter_response.values[0].string_value
                or any(message_to_ordereddict(value) != typed_parameters[name]
                       for name, value in zip(names, repeated.values[1:], strict=True))):
            raise ValueError("Preview model parameters changed during capture")
        scene = message_to_ordereddict(scene_response.scene)
        if scene["is_diff"] or scene["robot_state"]["is_diff"]:
            raise ValueError("GetPlanningScene returned an incomplete scene")
        return {
            "planning_scene": scene,
            "start_state": copy.deepcopy(scene["robot_state"]),
            "scene_fingerprint": _scene_message_fingerprint(scene_response.scene),
            "model_fingerprint": parameter_response.values[0].string_value,
            "robot_description": parameter_response.values[1].string_value,
            "robot_description_semantic": parameter_response.values[2].string_value,
            "native_model_parameters": typed_parameters,
            "source": planning_scene_service,
            "captured_at_unix": time.time(),
            "capture_attempts": {"planning_scene": scene_attempts, "model_parameters": parameter_attempts,
                                 "parameter_inventory": list_attempts, "typed_parameters": typed_attempts},
            "execution_authorized": False,
        }
    finally:
        node.destroy_node()
        context.shutdown()


def project_preview_scene(captured_scene: Mapping[str, Any], physical_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Copy supported projected joints and part geometry into a private ROS scene.

    ``scene_object_bindings`` must explicitly associate each part with its exact
    collision object identities and part-local geometry. Attachment declarations
    provide the attachment link, its transform to the resource reference link,
    and touch links. None of those identities/transforms are inferred from names.
    This function neither imports ROS nor mutates the captured scene.
    """
    from cais_spade_llm.resources.primitive_observations_3d import _compose, _unit_pose

    def object_pose(item):
        pose = item.get("pose")
        if pose is None:
            return (0., 0., 0., 0., 0., 0., 1.)
        values = [pose["position"][axis] for axis in ("x", "y", "z")]
        values += [pose["orientation"][axis] for axis in ("x", "y", "z", "w")]
        return _unit_pose(values, "collision object pose")

    scene = copy.deepcopy(dict(captured_scene.get("planning_scene", captured_scene)))
    snapshot = physical_evidence.get("snapshot", {})
    models = physical_evidence.get("resource_models", {})
    frame = snapshot.get("frame")
    if not frame or scene.get("is_diff") is not False or scene.get("robot_state", {}).get("is_diff") is not False:
        raise ValueError("Projected scene needs a complete captured scene and explicit snapshot frame")
    joint_state = scene["robot_state"].get("joint_state", {})
    names, positions = joint_state.get("name", []), joint_state.get("position", [])
    if len(names) != len(positions) or len(names) != len(set(names)):
        raise ValueError("Captured scene joint names/positions are incomplete or conflicting")
    assignments: dict[str, float] = {}
    resources = snapshot.get("resources", {})
    for rid, model in models.items():
        configuration = model.get("configuration")
        if configuration is None:
            continue
        row = resources.get(rid, {})
        recorded_names, values = row.get("joint_names"), row.get("joint_positions")
        if not isinstance(recorded_names, list) or not isinstance(values, list):
            raise ValueError(f"Projected joint configuration is missing for {rid}")
        if recorded_names != configuration.get("joint_names") or len(recorded_names) != len(values):
            raise ValueError(f"Projected joint identities differ from the owner model for {rid}")
        pairs = list(zip(recorded_names, values, strict=True))
        for joint in configuration.get("joints", []):
            if "frozen_position" in joint:
                pairs.append((joint["name"], joint["frozen_position"]))
            elif "frozen_interval" in joint:
                low, high = joint["frozen_interval"]
                if low != high:
                    raise ValueError(f"Projected joint {joint['name']} has no exact configuration")
                pairs.append((joint["name"], low))
        for name, value in pairs:
            if name not in names or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Projected joint is missing or invalid: {name}")
            if name in assignments and assignments[name] != value:
                raise ValueError(f"Conflicting projected joint evidence: {name}")
            assignments[name] = value
    for name, value in assignments.items():
        positions[names.index(name)] = value
    world = scene.get("world", {}).get("collision_objects", [])
    attachments = scene["robot_state"].get("attached_collision_objects", [])
    objects: dict[str, dict[str, Any]] = {}
    for item in [*world, *(entry["object"] for entry in attachments)]:
        object_id = item.get("id")
        if not object_id or object_id in objects:
            raise ValueError("Captured collision object identities are missing or conflicting")
        objects[object_id] = item
    bindings = physical_evidence.get("scene_object_bindings", {})
    replaced: set[str] = set()
    projected_world, projected_attachments = [], []
    for part, part_state in snapshot.get("parts", {}).items():
        binding = bindings.get(part)
        if not isinstance(binding, dict) or not binding.get("source") or not binding.get("objects"):
            raise ValueError(f"Missing scene_object_bindings for {part}")
        holders = [rid for rid, row in resources.items() if row.get("held_part") == part]
        if len(holders) > 1 or (holders and part_state.get("contained_by") is not None):
            raise ValueError(f"Conflicting projected custody for {part}")
        attachment = None
        if holders:
            rid = holders[0]
            attachment = binding.get("attachments", {}).get(rid)
            if not isinstance(attachment, dict) or not attachment.get("source"):
                raise ValueError(f"Missing attachment declaration for {rid} and {part}")
            if not attachment.get("link_name") or not isinstance(attachment.get("touch_links"), list):
                raise ValueError(f"Missing attachment link or touch links for {part}")
            carried_pose = _compose(
                _unit_pose(resources[rid].get("current_pose"), "projected resource current_pose"),
                _unit_pose(resources[rid].get("grasp_transform"), "projected grasp_transform"),
            )
            part_pose = _unit_pose(part_state.get("current_pose"), "projected part current_pose")
            if any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(carried_pose, part_pose, strict=True)):
                raise ValueError(f"Projected part pose conflicts with its attachment for {part}")
            parent_pose = _compose(
                _unit_pose(attachment.get("link_to_reference"), "attachment link_to_reference"),
                _unit_pose(resources[rid].get("grasp_transform"), "projected grasp_transform"),
            )
            parent_frame = attachment["link_name"]
        else:
            parent_pose = _unit_pose(part_state.get("current_pose"), "projected part current_pose")
            parent_frame = frame
        for object_binding in binding["objects"]:
            object_id = object_binding.get("object_id")
            template = object_binding.get("part_local_object")
            if object_id in replaced or object_id not in objects or not isinstance(template, dict):
                raise ValueError(f"Missing or conflicting collision object mapping for {part}")
            if template.get("id") != object_id:
                raise ValueError(f"Collision object mapping changes identity for {part}")
            original = objects[object_id]
            projected = copy.deepcopy(template)
            component_parent = _compose(parent_pose, object_pose(template))
            recorded_geometry = object_binding.get("recorded_geometry")
            if recorded_geometry is not None:
                from cais_spade_llm.recovery_framework.recovery_observations import fingerprint

                if (recorded_geometry.get("source") != "recorded physics collision geometry"
                        or recorded_geometry.get("object_fingerprint") != fingerprint(template)
                        or not recorded_geometry.get("collision_fingerprint")):
                    raise ValueError(f"Recorded collision geometry changed without evidence for {object_id}")
            for shapes, poses in (("primitives", "primitive_poses"), ("meshes", "mesh_poses"), ("planes", "plane_poses")):
                expected = (recorded_geometry.get("captured_geometry", {}).get(shapes, [])
                            if recorded_geometry is not None else template.get(shapes, []))
                if (expected != original.get(shapes, [])
                        and not (recorded_geometry is not None and template.get(shapes, []) == original.get(shapes, []))):
                    raise ValueError(f"Collision geometry changed without evidence for {object_id}")
                if len(template.get(shapes, [])) != len(template.get(poses, [])):
                    raise ValueError(f"Collision component poses are missing for {object_id}")
                result_poses = []
                for pose in template.get(poses, []):
                    values = [pose["position"][axis] for axis in ("x", "y", "z")]
                    values += [pose["orientation"][axis] for axis in ("x", "y", "z", "w")]
                    composed = _compose(component_parent, _unit_pose(values, "part-local collision pose"))
                    result_poses.append({"position": dict(zip(("x", "y", "z"), map(float, composed[:3]), strict=True)),
                                         "orientation": dict(zip(("x", "y", "z", "w"), map(float, composed[3:]), strict=True))})
                projected[poses] = result_poses
            if template.get("subframe_names") or template.get("subframe_poses"):
                raise ValueError(f"Projected subframe transformations are unsupported for {object_id}")
            projected["header"] = copy.deepcopy(original.get("header", {}))
            projected["header"]["frame_id"] = parent_frame
            if "pose" in projected:
                projected["pose"] = {"position": {"x": 0., "y": 0., "z": 0.},
                                     "orientation": {"x": 0., "y": 0., "z": 0., "w": 1.}}
            projected["operation"] = 0
            replaced.add(object_id)
            if attachment is None:
                projected_world.append(projected)
            else:
                projected_attachments.append({"link_name": parent_frame,
                                              "touch_links": copy.deepcopy(attachment["touch_links"]),
                                              "object": projected})
    scene["world"]["collision_objects"] = [item for item in world if item["id"] not in replaced] + projected_world
    scene["robot_state"]["attached_collision_objects"] = [
        item for item in attachments if item["object"]["id"] not in replaced
    ] + projected_attachments
    return {"planning_scene": scene, "start_state": copy.deepcopy(scene["robot_state"])}


def build_scene_object_bindings(
    captured_scene: Mapping[str, Any], physical_evidence: Mapping[str, Any], *,
    payload_evidence: Mapping[str, Any], attachment_targets: Mapping[str, Any],
    required_attachment_resources: set[str] | None = None,
) -> dict[str, Any]:
    """Ground exact MoveIt collision IDs and part-local geometry from owner records.

    ``payload_evidence`` contains owner-recorded model_name/collision_objects IDs;
    ``attachment_targets`` contains declared link_name/touch_links per resource.
    Actual geometry comes from the captured complete MoveIt scene. Frame and part
    transforms come exclusively from the accompanying fresh physical observations.
    Required attachment bindings may be restricted by the owner proposal and
    recorded custody. This never removes collision geometry from the scene.

    Args:
        captured_scene: Complete recorded MoveIt scene.
        physical_evidence: Recorded part, resource, and frame observations.
        payload_evidence: Owner-recorded collision object identities.
        attachment_targets: Declared attachment links and touch links.
        required_attachment_resources: Owners whose attachment bindings are
            needed; None retains the historical all-targets behavior.

    Returns:
        Part-local collision mappings with their recorded sources.
    """
    from cais_spade_llm.recovery_framework.geometry import rotate
    from cais_spade_llm.resources.primitive_observations_3d import _compose, _unit_pose

    def inverse(pose):
        value = _unit_pose(pose, "observed frame pose")
        orientation = [-float(value[3]), -float(value[4]), -float(value[5]), float(value[6])]
        return tuple(rotate(orientation, [-float(item) for item in value[:3]]) + orientation)

    snapshot = physical_evidence["snapshot"]
    frame = snapshot["frame"]
    scene = captured_scene.get("planning_scene", captured_scene)
    frames = {frame: [0., 0., 0., 0., 0., 0., 1.]}
    for rid, row in physical_evidence.get("resources", {}).items():
        for link, observation in row.get("link_states", {}).items():
            frames[link] = observation["pose"]
    for rid, model in physical_evidence.get("resource_models", {}).items():
        reference = model.get("configuration", {}).get("reference_link")
        if reference:
            value = snapshot["resources"][rid]["current_pose"]
            if reference in frames and list(frames[reference]) != list(value):
                raise ValueError(f"Conflicting observed link poses for {reference}")
            frames[reference] = value
    objects = {}
    for item in [*scene.get("world", {}).get("collision_objects", []),
                 *(entry["object"] for entry in scene["robot_state"].get("attached_collision_objects", []))]:
        if item["id"] in objects:
            raise ValueError(f"Conflicting captured collision object {item['id']}")
        objects[item["id"]] = item
    attachments = {}
    if required_attachment_resources is not None:
        missing_targets = required_attachment_resources - attachment_targets.keys()
        if missing_targets:
            raise SceneObjectBindingError(f"Attachment declarations are missing for {sorted(missing_targets)}",
                [f"/physical_evidence/resources/{rid}/attachment" for rid in sorted(missing_targets)])
    for rid, target in attachment_targets.items():
        if required_attachment_resources is not None and rid not in required_attachment_resources:
            continue
        link = target.get("link_name")
        configuration = physical_evidence.get("resource_models", {}).get(rid, {}).get("configuration", {})
        reference = configuration.get("reference_link")
        calculated = None
        if link not in frames and reference in frames:
            from cais_spade_llm.recovery_framework.recovery_observations import _captured_tcp_transform

            calculated = _captured_tcp_transform({"controlled_link": reference, "tcp_link": link},
                configuration, source_path=f"physical_evidence/resource_models/{rid}/configuration")
            frames[link] = list(map(float, _compose(_unit_pose(frames[reference], "reference pose"),
                _unit_pose([*calculated["translation"], *calculated["rotation"]], "fixed attachment transform"))))
        if link not in frames or reference not in frames or not isinstance(target.get("touch_links"), list):
            raise SceneObjectBindingError(f"Attachment link observations/declarations are missing for {rid}",
                [f"/physical_evidence/resource_models/{rid}/configuration",
                 f"/physical_evidence/resources/{rid}/link_states"])
        attachments[rid] = {
            "link_name": link, "touch_links": copy.deepcopy(target["touch_links"]),
            "link_to_reference": list(map(float, _compose(inverse(frames[link]), _unit_pose(frames[reference], "reference pose")))),
            "source": {"attachment_target": copy.deepcopy(target), "link_frame": link,
                       "reference_link": reference, "physical_provenance": copy.deepcopy(physical_evidence.get("provenance")),
                       **({"fixed_link_calculation": calculated} if calculated is not None else {})},
        }
    bindings = {}
    for part, row in snapshot.get("parts", {}).items():
        model_name = row.get("model_name") or physical_evidence.get("parts", {}).get(part, {}).get("model_name")
        contributors = [[rid, evidence] for rid, evidence in payload_evidence.items()
                        if model_name and evidence.get("model_name") == model_name]
        ids = {item["id"] for _, evidence in contributors for item in evidence.get("collision_objects", [])}
        if not ids:
            raise SceneObjectBindingError(f"Owner-recorded collision object identities are missing for {part}",
                [f"/physical_evidence/parts/{part}/collision_scene_identity"])
        local_objects = []
        for object_id in sorted(ids):
            original = objects.get(object_id)
            if original is None:
                raise SceneObjectBindingError(f"Recorded collision object is missing from the current scene: {object_id}",
                    [f"/physical_evidence/parts/{part}/collision_scene_identity",
                     "/planning_scene_capture/planning_scene/world/collision_objects",
                     "/planning_scene_capture/planning_scene/robot_state/attached_collision_objects"])
            native = physical_evidence.get("physics_evidence", {})
            if native.get("models"):
                identities = [item for _, evidence in contributors for item in evidence.get("collision_objects", [])
                              if item.get("id") == object_id and all(item.get(key) for key in ("model", "link", "collision"))]
                unique = {(item["model"], item["link"], item["collision"]) for item in identities}
                if len(unique) != 1:
                    raise ValueError(f"One exact recorded physics collision identity is required for {object_id}")
                local, provenance = _recorded_part_collision_object(
                    original, physical_evidence, part_model=model_name, identity=next(iter(unique)))
                local_objects.append({"object_id": object_id, "part_local_object": local,
                                      "recorded_geometry": provenance})
                continue
            source_frame = original.get("header", {}).get("frame_id")
            if source_frame not in frames:
                raise ValueError(f"Collision object frame observation is missing: {source_frame}")
            local = copy.deepcopy(original)
            parent = _compose(inverse(row["current_pose"]), _unit_pose(frames[source_frame], "object frame pose"))
            object_pose = original.get("pose")
            if object_pose is not None:
                values = [object_pose["position"][axis] for axis in ("x", "y", "z")]
                values += [object_pose["orientation"][axis] for axis in ("x", "y", "z", "w")]
                parent = _compose(parent, _unit_pose(values, "collision object pose"))
                local["pose"] = {"position": {"x": 0., "y": 0., "z": 0.},
                                 "orientation": {"x": 0., "y": 0., "z": 0., "w": 1.}}
            for poses in ("primitive_poses", "mesh_poses", "plane_poses"):
                transformed = []
                for pose in original.get(poses, []):
                    value = [pose["position"][axis] for axis in ("x", "y", "z")]
                    value += [pose["orientation"][axis] for axis in ("x", "y", "z", "w")]
                    result = _compose(parent, _unit_pose(value, "collision component pose"))
                    transformed.append({"position": dict(zip(("x", "y", "z"), map(float, result[:3]), strict=True)),
                                        "orientation": dict(zip(("x", "y", "z", "w"), map(float, result[3:]), strict=True))})
                local[poses] = transformed
            local_objects.append({"object_id": object_id, "part_local_object": local})
        bindings[part] = {
            "objects": local_objects, "attachments": copy.deepcopy(attachments),
            "source": {"model_name": model_name, "owner_records": copy.deepcopy(contributors),
                       "scene_fingerprint": captured_scene.get("scene_fingerprint"),
                       "physical_provenance": copy.deepcopy(physical_evidence.get("provenance"))},
        }
    return bindings


def _recorded_part_collision_object(original: dict, evidence: Mapping[str, Any], *,
                                    part_model: str, identity: tuple[str, str, str]) -> tuple[dict, dict]:
    """Rebuild part-local geometry from one saved physics record and mesh data.

    MoveIt may still contain a pre-drop pose or payload contact allowance. Its
    identity is retained, while the recorded native collision supplies geometry.
    The native bounds check rejects inconsistent mesh scales or source data.
    """
    from cais_spade_llm.recovery_framework.geometry import rotate
    from cais_spade_llm.recovery_framework.recovery_observations import fingerprint
    from cais_spade_llm.resources.primitive_observations_3d import _compose, _unit_pose

    physics = evidence["physics_evidence"]
    model_name, link_name, collision_name = identity
    scoped_id = f"{model_name}::{link_name}::{collision_name}"
    collisions = physics["models"][model_name]["links"][link_name]["collisions"]
    matches = [row for row in collisions if row.get("id") == scoped_id]
    if len(matches) != 1:
        raise ValueError(f"Recorded collision is missing or ambiguous: {scoped_id}")
    collision = matches[0]
    shape = collision.get("geometry") or {}
    pose = _unit_pose(collision.get("pose"), "recorded collision pose")
    model_pose = _unit_pose(physics["models"][part_model].get("pose"), "recorded part model pose")
    orientation = [-float(model_pose[3]), -float(model_pose[4]), -float(model_pose[5]), float(model_pose[6])]
    inverse = [*rotate(orientation, [-float(value) for value in model_pose[:3]]), *orientation]
    local_pose = _compose(_unit_pose(inverse, "inverse part model pose"), pose)
    component = {"position": dict(zip(("x", "y", "z"), map(float, local_pose[:3]), strict=True)),
                 "orientation": dict(zip(("x", "y", "z", "w"), map(float, local_pose[3:]), strict=True))}
    result = copy.deepcopy(original)
    result["pose"] = {"position": {"x": 0., "y": 0., "z": 0.},
                      "orientation": {"x": 0., "y": 0., "z": 0., "w": 1.}}
    for field in ("primitives", "primitive_poses", "meshes", "mesh_poses", "planes", "plane_poses"):
        result[field] = []
    mesh_source = None
    if shape.get("type") == "box":
        size = shape.get("size")
        if not isinstance(size, list) or len(size) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in size):
            raise ValueError(f"Invalid recorded box geometry for {scoped_id}")
        result["primitives"] = [{"type": 1, "dimensions": copy.deepcopy(size)}]
        result["primitive_poses"] = [component]
        vertices = [[x * size[0] / 2, y * size[1] / 2, z * size[2] / 2]
                    for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    elif shape.get("type") == "cylinder":
        radius, length = shape.get("radius"), shape.get("length")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (radius, length)):
            raise ValueError(f"Invalid recorded cylinder geometry for {scoped_id}")
        result["primitives"] = [{"type": 3, "dimensions": [length, radius]}]
        result["primitive_poses"] = [component]
        vertices = [[radius * math.cos(i * math.pi / 8), radius * math.sin(i * math.pi / 8), z * length / 2]
                    for i in range(16) for z in (-1, 1)]
    elif shape.get("type") == "mesh":
        mesh = evidence.get("physics_meshes", {}).get(shape.get("uri")) or {}
        scale = shape.get("scale")
        raw = mesh.get("vertices")
        triangles = mesh.get("triangles")
        if (mesh.get("uri") != shape.get("uri") or not mesh.get("source_sha256")
                or mesh.get("source_kind") != "recorded_collision_mesh"
                or not isinstance(raw, list) or not raw or len(raw) != mesh.get("vertex_count")
                or not isinstance(triangles, list) or not triangles or len(triangles) != mesh.get("triangle_count")
                or not isinstance(scale, list) or len(scale) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in scale)):
            raise ValueError(f"Complete saved collision mesh and scale are required for {scoped_id}")
        if any(not isinstance(vertex, list) or len(vertex) != 3
               or any(type(v) not in (int, float) or not math.isfinite(v) for v in vertex) for vertex in raw):
            raise ValueError(f"Invalid recorded collision mesh vertices for {scoped_id}")
        if any(not isinstance(triangle, list) or len(triangle) != 3
               or any(type(index) is not int or not 0 <= index < len(raw) for index in triangle) for triangle in triangles):
            raise ValueError(f"Invalid recorded collision mesh triangles for {scoped_id}")
        vertices = [[value * scale[index] for index, value in enumerate(vertex)] for vertex in raw]
        result["meshes"] = [{"vertices": [dict(zip(("x", "y", "z"), vertex, strict=True)) for vertex in vertices],
                             "triangles": [{"vertex_indices": copy.deepcopy(triangle)} for triangle in triangles]}]
        result["mesh_poses"] = [component]
        mesh_source = {key: copy.deepcopy(mesh[key]) for key in ("uri", "source_sha256", "vertex_count", "triangle_count", "source_kind")}
    else:
        raise ValueError(f"Unsupported recorded collision geometry for {scoped_id}")
    bounds = collision.get("bounds")
    if (not isinstance(bounds, list) or len(bounds) != 3
            or any(not isinstance(axis, list) or len(axis) != 2
                   or any(type(v) not in (int, float) or not math.isfinite(v) for v in axis)
                   or axis[0] > axis[1] for axis in bounds)):
        raise ValueError(f"Recorded collision bounds are missing for {scoped_id}")
    for vertex in vertices:
        point = [float(pose[axis]) + value for axis, value in enumerate(rotate(
            [float(value) for value in pose[3:]], vertex))]
        if any(point[axis] < bounds[axis][0] - 1e-7 or point[axis] > bounds[axis][1] + 1e-7 for axis in range(3)):
            raise ValueError(f"Recorded collision shape/scale conflicts with native bounds for {scoped_id}")
    return result, {"source": "recorded physics collision geometry", "collision_id": scoped_id,
                    "instance_id": physics.get("instance_id"), "simulation_time": physics.get("simulation_time"),
                    "collision_fingerprint": fingerprint(collision), "object_fingerprint": fingerprint(result),
                    "mesh_source": mesh_source,
                    "captured_geometry": {field: copy.deepcopy(original.get(field, []))
                                          for field in ("primitives", "meshes", "planes")}}


def apply_declared_mounting_contacts(
    planning_scene: Mapping[str, Any], *, robot_description: str, robots: list[dict[str, Any]],
    physical_evidence: Mapping[str, Any], declarations: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply verified mounting contacts only to a detached private scene.

    Fixed joints must be explicitly selected. External contacts additionally
    require a declared static collision, its recorded geometry, and a robot
    mounting frame fixed to the same recorded world pose. The returned evidence
    is a model calculation; it does not change observations or authorize motion.
    """
    from cais_spade_llm.recovery_framework.geometry import compose, quaternion, rotate
    from cais_spade_llm.recovery_framework.recovery_observations import fingerprint

    if (not isinstance(declarations, Mapping) or set(declarations) != {"fixed_joints", "world_contacts"}
            or not isinstance(declarations["fixed_joints"], list)
            or not isinstance(declarations["world_contacts"], list)):
        raise ValueError("Explicit fixed_joints and world_contacts declarations are required")
    model = ET.fromstring(robot_description)
    links = {link.get("name"): link for link in model.findall("link")}
    joints = {joint.get("name"): joint for joint in model.findall("joint")}
    if (len(links) != len(model.findall("link")) or len(joints) != len(model.findall("joint"))
            or None in links or None in joints):
        raise ValueError("Recorded URDF identities are ambiguous")
    parents = {}
    for joint in joints.values():
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None or parent.get("link") not in links or child.get("link") not in links:
            raise ValueError("Recorded URDF joint references are incomplete")
        if child.get("link") in parents:
            raise ValueError("Recorded URDF child has multiple parents")
        parents[child.get("link")] = joint
    roots = set(links) - set(parents)
    if len(roots) != 1:
        raise ValueError("Recorded URDF requires one exact root frame")
    world_frame = next(iter(roots))

    def numbers(values: Any, count: int, field: str) -> list[float]:
        if (not isinstance(values, list) or len(values) != count
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
            raise ValueError(f"Missing finite {field}")
        return list(values)

    def same(first: list[float], second: list[float], field: str) -> None:
        # Only arithmetic roundoff is allowed when composing declared transforms.
        if len(first) != len(second) or any(abs(a - b) > 1e-10 for a, b in zip(first, second, strict=True)):
            raise ValueError(f"Conflicting {field}")

    def fixed_transform(child: str, ancestor: str) -> tuple[list[float], list[str]]:
        if child not in links or ancestor not in links:
            raise ValueError("Mounting link is absent from the recorded URDF")
        chain, transforms, seen = [], [], set()
        while child != ancestor:
            if child in seen or child not in parents:
                raise ValueError("Mounting link has no fixed path to its declared frame")
            seen.add(child)
            joint = parents[child]
            if joint.get("type") != "fixed":
                raise ValueError("A moving joint cannot establish a mounting contact")
            origin = joint.find("origin")
            # Omitted URDF origins are defined by URDF itself as identity.
            xyz = [float(value) for value in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
            rpy = [float(value) for value in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            transforms.append([*numbers(xyz, 3, "URDF xyz"), *quaternion(numbers(rpy, 3, "URDF rpy"))])
            chain.append(joint.get("name"))
            child = joint.find("parent").get("link")
        pose = [0., 0., 0., 0., 0., 0., 1.]
        for transform in reversed(transforms):
            pose = compose(pose, transform)
        return pose, chain

    def scene_pose(value: Any) -> list[float]:
        if not isinstance(value, dict) or set(value) != {"position", "orientation"}:
            raise ValueError("A complete recorded collision pose is required")
        return numbers([value["position"][axis] for axis in ("x", "y", "z")]
            + [value["orientation"][axis] for axis in ("x", "y", "z", "w")], 7, "collision pose")

    contacts = []
    selected = declarations["fixed_joints"]
    if any(not isinstance(name, str) or not name for name in selected) or len(set(selected)) != len(selected):
        raise ValueError("Fixed mounting joint declarations are invalid or repeated")
    for name in selected:
        joint = joints.get(name)
        if joint is None or joint.get("type") != "fixed":
            raise ValueError(f"Declared mounting joint is not fixed: {name}")
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        if not links[parent].findall("collision") or not links[child].findall("collision"):
            raise ValueError(f"Declared fixed mounting pair has no collision geometry: {name}")
        contacts.append({"pair": [parent, child], "fixed_joint": name,
                         "joint_fingerprint": fingerprint(ET.tostring(joint, encoding="unicode"))})

    physics = physical_evidence.get("physics_evidence") or {}
    objects = planning_scene.get("world", {}).get("collision_objects", [])
    if declarations["world_contacts"] and (
            not isinstance(physics.get("instance_id"), str) or not physics["instance_id"]
            or type(physics.get("simulation_time")) not in (int, float)
            or not math.isfinite(physics["simulation_time"]) or physics["simulation_time"] < 0):
        raise ValueError("Recorded mounting observations require their instance and capture time")
    for declaration in declarations["world_contacts"]:
        required = {"resource_id", "robot_collision_link", "mounting_frame", "world_collision_object_id",
                    "physics_collision_id", "geometry", "pose"}
        if not isinstance(declaration, dict) or set(declaration) != required:
            raise ValueError("External mounting contact requires exact declared fields")
        owner = [robot for robot in robots if robot.get("resource_id") == declaration["resource_id"]]
        if len(owner) != 1:
            raise ValueError("Declared mounting resource is missing or ambiguous")
        robot = owner[0]
        link, frame = declaration["robot_collision_link"], declaration["mounting_frame"]
        if not isinstance(robot.get("prefix"), str) or not robot["prefix"] or not all(
                isinstance(name, str) and name.startswith(robot["prefix"]) for name in (link, frame)):
            raise ValueError("Mounting links do not belong to the recorded resource prefix")
        if link not in links or not links[link].findall("collision"):
            raise ValueError("Declared mounting collision link has no recorded geometry")
        _, collision_chain = fixed_transform(link, frame)
        base_pose, world_chain = fixed_transform(frame, world_frame)
        declared_base = [*numbers(robot.get("base_xyz"), 3, "robot base_xyz"),
                         *quaternion(numbers(robot.get("base_rpy"), 3, "robot base_rpy"))]
        same(base_pose, declared_base, "recorded robot mounting pose")
        geometry, pose = declaration["geometry"], numbers(declaration["pose"], 7, "mounting plate pose")
        if not isinstance(geometry, dict) or set(geometry) != {"type", "size"} or geometry["type"] != "box":
            raise ValueError("Only explicitly declared box mounting plates are supported")
        size = numbers(geometry["size"], 3, "mounting plate size")
        if any(value <= 0 for value in size):
            raise ValueError("Mounting plate dimensions must be positive")
        same(pose[3:], [0., 0., 0., 1.], "horizontal mounting plate orientation")
        same(base_pose[:3], [pose[0], pose[1], pose[2] + size[2] / 2], "mounting plate top and robot base")
        same(rotate(base_pose[3:], [0., 0., 1.]), [0., 0., 1.], "robot mounting axis")
        native = [(model_name, link_name, collision, body) for model_name, body in physics.get("models", {}).items()
                  for link_name, recorded_link in body.get("links", {}).items()
                  for collision in recorded_link.get("collisions", [])
                  if collision.get("id") == declaration["physics_collision_id"]]
        if len(native) != 1 or native[0][3].get("static") is not True:
            raise ValueError("Declared mounting plate lacks unique recorded static collision evidence")
        model_name, link_name, collision, body = native[0]
        collision_prefix = f"{model_name}::{link_name}::"
        if (not declaration["physics_collision_id"].startswith(collision_prefix)
                or declaration["world_collision_object_id"] !=
                f"{model_name}/{link_name}/{declaration['physics_collision_id'][len(collision_prefix):]}"):
            raise ValueError("Mounting scene object identity conflicts with the recorded collision")
        if collision.get("geometry") != geometry:
            raise ValueError("Declared mounting geometry conflicts with recorded physics")
        same(numbers(collision.get("pose"), 7, "native mounting pose"), pose, "recorded mounting plate pose")
        matches = [obj for obj in objects if obj.get("id") == declaration["world_collision_object_id"]]
        if len(matches) != 1:
            raise ValueError("Declared mounting scene object is missing or ambiguous")
        obj = matches[0]
        if (obj.get("header", {}).get("frame_id") != world_frame or obj.get("operation") not in (0, "\x00")
                or obj.get("meshes") or obj.get("planes") or obj.get("subframe_names") or obj.get("subframe_poses")
                or len(obj.get("primitives", [])) != 1 or len(obj.get("primitive_poses", [])) != 1):
            raise ValueError("Mounting scene object is not one complete box in the recorded world frame")
        primitive = obj["primitives"][0]
        if primitive.get("type") != 1 or primitive.get("dimensions") != size:
            raise ValueError("Mounting scene object geometry conflicts with its declaration")
        same(compose(scene_pose(obj.get("pose")), scene_pose(obj["primitive_poses"][0])), pose,
             "mounting scene object pose")
        contacts.append({"pair": [link, obj["id"]], "resource_id": declaration["resource_id"],
            "fixed_joint_chain": collision_chain + world_chain, "declaration_fingerprint": fingerprint(declaration),
            "robot_configuration_fingerprint": fingerprint(robot), "world_object_fingerprint": fingerprint(obj),
            "physics_collision_id": declaration["physics_collision_id"], "physics_collision_fingerprint": fingerprint(collision),
            "physics_model": model_name, "physics_link": link_name, "physics_model_static": body["static"],
            "physics_instance_id": physics.get("instance_id"), "simulation_time": physics.get("simulation_time")})

    scene = copy.deepcopy(dict(planning_scene))
    matrix = scene.get("allowed_collision_matrix")
    if not isinstance(matrix, dict) or matrix.get("default_entry_names") or matrix.get("default_entry_values"):
        raise ValueError("Mounting contacts require an explicit collision matrix without default allowances")
    names, rows = matrix.get("entry_names"), matrix.get("entry_values")
    if (not isinstance(names, list) or not isinstance(rows, list) or len(set(names)) != len(names)
            or len(rows) != len(names) or any(not isinstance(row, dict) or set(row) != {"enabled"}
            or len(row["enabled"]) != len(names) or any(type(value) is not bool for value in row["enabled"]) for row in rows)):
        raise ValueError("Recorded collision matrix is incomplete")
    if any(rows[i]["enabled"][j] != rows[j]["enabled"][i] for i in range(len(names)) for j in range(len(names))):
        raise ValueError("Recorded collision matrix is asymmetric")
    for contact in contacts:
        for name in contact["pair"]:
            if name not in names:
                for row in rows:
                    row["enabled"].append(False)
                names.append(name)
                rows.append({"enabled": [False] * len(names)})
        first, second = (names.index(name) for name in contact["pair"])
        contact["previously_allowed"] = rows[first]["enabled"][second]
        rows[first]["enabled"][second] = rows[second]["enabled"][first] = True
    return {"planning_scene": scene, "evidence": {
        "source": "declared mounting contacts verified against recorded geometry and fixed joints",
        "robot_description_sha256": hashlib.sha256(robot_description.encode()).hexdigest(),
        "declarations_fingerprint": fingerprint(declarations), "physical_evidence_fingerprint": fingerprint(physical_evidence),
        "input_scene_fingerprint": fingerprint(planning_scene), "output_scene_fingerprint": fingerprint(scene),
        "contacts": contacts, "execution_authorized": False, "command_sent": False,
    }}


def preview_scene_fingerprint(scene: Mapping[str, Any]) -> str:
    """Hash exact ROS scene field values during an explicit preview operation.

    ROS imports remain lazy. Pure offline validation compares saved fingerprints
    and never calls this transport-format helper.
    """
    from moveit_msgs.msg import PlanningScene

    message = PlanningScene()
    _set_recorded_message_fields(message, scene)
    return _scene_message_fingerprint(message)


def _set_recorded_message_fields(message: Any, values: Mapping[str, Any]) -> None:
    """Restore exact recorded ROS fields, including JSON-encoded octets."""
    from rosidl_parser.definition import AbstractNestedType, BasicType, NamespacedType
    from rosidl_runtime_py.convert import get_message_slot_types
    from rosidl_runtime_py.import_message import import_message_from_namespaced_type
    from rosidl_runtime_py.set_message import set_message_fields

    def restored(target: Any, data: Mapping[str, Any]) -> dict[str, Any]:
        result = {}
        descriptors = get_message_slot_types(target)
        for field, value in data.items():
            descriptor = descriptors[field]
            if isinstance(descriptor, BasicType) and descriptor.typename == "octet":
                if isinstance(value, str) and len(value) == 1 and ord(value) <= 255:
                    result[field] = bytes([ord(value)])
                elif type(value) is int and 0 <= value <= 255:
                    result[field] = bytes([value])
                elif not isinstance(value, bytes) or len(value) != 1:
                    raise ValueError(f"Recorded octet is invalid: {field}")
                else:
                    result[field] = value
            elif isinstance(descriptor, NamespacedType):
                result[field] = restored(getattr(target, field), value)
            elif isinstance(descriptor, AbstractNestedType) and isinstance(descriptor.value_type, NamespacedType):
                element_type = import_message_from_namespaced_type(descriptor.value_type)
                result[field] = [restored(element_type(), item) for item in value]
            else:
                result[field] = copy.deepcopy(value)
        return result

    set_message_fields(message, restored(message, values))


def _scene_message_fingerprint(message: Any) -> str:
    """Encode typed ROS fields identically to preview_scene_fingerprint.hpp."""
    from rosidl_parser.definition import AbstractSequence, AbstractString, Array, BasicType, NamespacedType

    scalar_formats = {"boolean": "?", "octet": "B", "char": "B", "uint8": "B", "int8": "b",
                      "uint16": "H", "int16": "h", "uint32": "I", "int32": "i",
                      "uint64": "Q", "int64": "q", "float": "f", "double": "d"}
    digest = hashlib.sha256()

    def field(value: Any, descriptor: Any) -> None:
        if isinstance(descriptor, (AbstractSequence, Array)):
            digest.update(struct.pack("<Q", len(value)))
            for item in value:
                field(item, descriptor.value_type)
        elif isinstance(descriptor, BasicType):
            if descriptor.typename in {"char", "octet"} and isinstance(value, (str, bytes)):
                value = ord(value)
            digest.update(struct.pack("<" + scalar_formats[descriptor.typename], value))
        elif isinstance(descriptor, AbstractString):
            data = value.encode("utf-8")
            digest.update(struct.pack("<Q", len(data)))
            digest.update(data)
        elif isinstance(descriptor, NamespacedType):
            encode(value)
        else:
            raise ValueError("Unsupported ROS field type in scene fingerprint")

    def encode(value: Any) -> None:
        for name, descriptor in zip(value.get_fields_and_field_types(), value.SLOT_TYPES, strict=True):
            field(getattr(value, name), descriptor)

    encode(message)
    return digest.hexdigest()

def _native_preview_executable() -> Path:
    from ament_index_python.packages import get_package_prefix

    path = Path(get_package_prefix('cais_lab_robotics')) / 'lib/cais_lab_robotics/preview_recovery_motion'
    if not path.is_file():
        raise ValueError('Installed file-based native preview executable is unavailable')
    return path.resolve()


async def _settle_native_task(task: asyncio.Task) -> tuple[Any, bool]:
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    return task.result(), cancelled


async def _reap_native_preview(process: Any, waiter: asyncio.Task) -> None:
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(asyncio.shield(waiter), timeout=5.)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await waiter


async def preview_recovery_motion_from_files(
    request: Mapping[str, Any], *, execution_mode: str, model_archive: dict,
    artifact_directory: str | Path, timeout_seconds: float = 60., cancel_event: Any = None,
) -> dict[str, Any]:
    """Calculate a private motion using saved parameters and Goal/Result files.

    Args:
        request: Exact owner scene, target, settings and evidence bindings.
        execution_mode: Must be ``simulation``.
        model_archive: Verified source-bound companion from saved observations.
        artifact_directory: This attempt's directory for immutable native calls.
        timeout_seconds: Deadline for this native calculation only.
        cancel_event: Owned cancellation event; cancellation reaps the child.

    Returns:
        Checked native evidence with original CDR files and hash provenance.
        ROS serialization and MoveIt libraries are local dependencies. No node,
        action client, live scene, controller, perception or SPADE call is made.
    """
    from cais_spade_llm.recovery_framework.recovery_observations import _file_sha256, read_native_preview_model, fingerprint
    from cais_spade_llm.recovery_framework.diagnostics import write_record

    if execution_mode != 'simulation':
        raise ValueError('Recovery motion previews are restricted to simulation')
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('timeout_seconds must be positive and finite')
    prepared = validate_preview_request(request)
    source = Path(model_archive['source_path'])
    observations = json.loads(source.read_bytes())
    archive = read_native_preview_model(source, observations)
    if fingerprint(archive) != fingerprint(model_archive) or prepared['model_fingerprint'] != archive['binding']['model_fingerprint']:
        raise ValueError('Native preview model changed or differs from the request')
    geometry = {}
    for row in observations.get('physical_evidence', {}).get('resources', {}).values():
        for name, digest in (row.get('continuous_motion_configuration') or {}).get('geometry_sources', {}).items():
            if name in geometry and geometry[name] != digest:
                raise ValueError('Captured geometry source fingerprints conflict')
            geometry[name] = digest
    for name, digest in geometry.items():
        if _file_sha256(Path(name)) != digest:
            raise ValueError('Native preview geometry source changed: ' + name)
    from cais_lab_robotics.action import PreviewRecoveryMotion
    from rclpy.serialization import deserialize_message, serialize_message
    from rosidl_runtime_py.convert import message_to_ordereddict

    goal = PreviewRecoveryMotion.Goal()
    _set_recorded_message_fields(goal, prepared)
    if _scene_message_fingerprint(goal.planning_scene) != prepared['scene_fingerprint']:
        raise ValueError('Planning scene fingerprint differs from the captured request')
    if cancel_event is not None and cancel_event.is_set():
        raise asyncio.CancelledError()
    executable = _native_preview_executable()
    directory = Path(artifact_directory).resolve() / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    goal_path, result_path = directory / 'goal.cdr', directory / 'result.cdr'
    goal_path.write_bytes(serialize_message(goal))
    files = {**archive['files'], **geometry, str(source): archive['binding']['source_sha256'],
             archive['manifest_path']: archive['manifest_sha256'], str(goal_path): _file_sha256(goal_path),
             str(executable): _file_sha256(executable)}
    provenance = {'sha256': files, 'model_archive': archive['manifest_path'],
        'capture_binding': archive['binding'], 'capture_verification': archive['capture_verification'],
        'request_binding': {key: prepared[key] for key in _FINGERPRINTS},
        'transport': 'recorded Goal and Result files', 'started_at_unix': time.time(),
        'status': 'preparing', 'command_sent': False, 'execution_authorized': False}
    write_record(directory / 'provenance.json', provenance)
    process, waiter = None, None
    started = time.monotonic()
    try:
        with (directory / 'native.log').open('w') as log:
            # SPADE can own a selector loop under a policy without a child
            # watcher. Keep this owned file process independent of that policy.
            spawning = asyncio.create_task(asyncio.to_thread(subprocess.Popen, [str(executable), '--input-goal', str(goal_path),
                '--output-result', str(result_path), '--ros-args', '--params-file', archive['parameters_path']],
                stdout=log, stderr=log))
            process, cancelled = await _settle_native_task(spawning)
            waiter = asyncio.create_task(asyncio.to_thread(process.wait))
            if cancelled:
                raise asyncio.CancelledError()
            provenance.update(status='calculating', pid=process.pid)
            write_record(directory / 'provenance.json', provenance)
            deadline = time.monotonic() + timeout_seconds
            while not waiter.done():
                if cancel_event is not None and cancel_event.is_set():
                    raise asyncio.CancelledError()
                if time.monotonic() >= deadline:
                    raise TimeoutError('Recorded native preview exceeded its request deadline')
                await asyncio.wait({waiter}, timeout=.05)
            if await waiter:
                raise RuntimeError('Native preview failed: ' + (directory / 'native.log').read_text()[-4000:])
        if any(_file_sha256(Path(name)) != digest for name, digest in files.items()):
            raise ValueError('Recorded native preview inputs changed during calculation')
        result_bytes = result_path.read_bytes()
        raw = message_to_ordereddict(deserialize_message(result_bytes, PreviewRecoveryMotion.Result))
        record = preview_result_record(prepared, raw)
        provenance.update(status='complete', result_sha256=hashlib.sha256(result_bytes).hexdigest())
        record['recorded_file_evidence'] = {'directory': str(directory),
            'result_sha256': provenance['result_sha256'], 'model_archive_sha256': archive['manifest_sha256']}
        return record
    except asyncio.CancelledError:
        provenance.update(status='cancelled', error='Owned preview cancelled')
        raise
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, TimeoutError) as exc:
        provenance.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if process is not None and waiter is not None:
            cleanup = asyncio.create_task(_reap_native_preview(process, waiter))
            _, cancelled = await _settle_native_task(cleanup)
            provenance.update(exit_code=process.returncode, cancellation_complete=process.returncode is not None)
            if cancelled:
                provenance.update(status='cancelled', error='Owned preview cancelled during cleanup',
                                  elapsed_seconds=time.monotonic() - started)
                if result_path.is_file():
                    provenance['result_sha256'] = _file_sha256(result_path)
                write_record(directory / 'provenance.json', provenance)
                raise asyncio.CancelledError()
        provenance['elapsed_seconds'] = time.monotonic() - started
        if result_path.is_file():
            provenance['result_sha256'] = _file_sha256(result_path)
        write_record(directory / 'provenance.json', provenance)


async def verify_native_model_parameters(parameters_path: str | Path, *, expected_model_fingerprint: str,
                                        artifact_directory: str | Path, cancel_event=None) -> dict:
    """Recompute the native model identity with a Goal that cannot plan motion."""
    from cais_lab_robotics.action import PreviewRecoveryMotion
    from rclpy.serialization import serialize_message, deserialize_message
    from cais_spade_llm.recovery_framework.recovery_observations import _file_sha256

    directory = Path(artifact_directory)
    parameters, executable = Path(parameters_path), _native_preview_executable()
    hashes = {str(path): _file_sha256(path) for path in (parameters, executable)}
    goal = PreviewRecoveryMotion.Goal()
    goal.timeout_seconds = 1.
    # Empty resource/capture/proposal identities fail before scene or movement
    # handling. initialize(false) still computes the exact parameter fingerprint.
    source, target = directory / 'fingerprint-goal.cdr', directory / 'fingerprint-result.cdr'
    source.write_bytes(serialize_message(goal))
    if cancel_event is not None and cancel_event.is_set():
        raise asyncio.CancelledError()
    with (directory / 'fingerprint-native.log').open('w') as log:
        spawning = asyncio.create_task(asyncio.to_thread(subprocess.Popen, [str(executable), '--input-goal', str(source),
            '--output-result', str(target), '--ros-args', '--params-file', str(parameters)], stdout=log, stderr=log))
        process, waiter = None, None
        try:
            process, cancelled = await _settle_native_task(spawning)
            waiter = asyncio.create_task(asyncio.to_thread(process.wait))
            if cancelled:
                raise asyncio.CancelledError()
            deadline = time.monotonic() + 60.
            while not waiter.done():
                if cancel_event is not None and cancel_event.is_set():
                    raise asyncio.CancelledError()
                if time.monotonic() >= deadline:
                    raise TimeoutError('Native model fingerprint calculation exceeded its request deadline')
                await asyncio.wait({waiter}, timeout=.05)
            if await waiter:
                raise ValueError('Native model fingerprint calculation failed; inspect fingerprint-native.log')
        finally:
            if process is not None:
                if waiter is None:
                    waiter = asyncio.create_task(asyncio.to_thread(process.wait))
                cleanup = asyncio.create_task(_reap_native_preview(process, waiter))
                _, cancelled = await _settle_native_task(cleanup)
                if cancelled:
                    raise asyncio.CancelledError()
    result = deserialize_message(target.read_bytes(), PreviewRecoveryMotion.Result)
    if (any(_file_sha256(Path(name)) != value for name, value in hashes.items())
            or result.model_fingerprint != expected_model_fingerprint or result.path_complete
            or result.trajectory.joint_trajectory.points
            or result.reason != 'Resource, capture, and proposal fingerprints are required'):
        raise ValueError('Saved typed parameters did not reproduce the captured native model')
    return {'parameters_sha256': hashes[str(parameters)], 'model_fingerprint': result.model_fingerprint,
        'executable': str(executable), 'executable_sha256': hashes[str(executable)],
        'goal_sha256': _file_sha256(source), 'result_sha256': _file_sha256(target),
        'fingerprint_verified': True, 'native_exit_code': process.returncode,
        'command_sent': False, 'execution_authorized': False}
