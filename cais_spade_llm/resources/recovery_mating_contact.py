from __future__ import annotations

"""Recorded mating declarations and non-executing continuous corridor evidence."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from types import SimpleNamespace

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
from cais_spade_llm.recovery_framework.geometry import compose, rotate
from cais_spade_llm.recovery_framework.part_collision import mating_contacts_allowed, mating_pose_valid
from cais_spade_llm.resources.continuous_motion import (
    ContinuousMotion, add, cosine, interval, kinematic_transforms, mul, neg,
    polynomial_bounds, quaternion_transform, representative_pose, sine,
)
from cais_spade_llm.resources.primitive_observations_3d import _unit_pose
from cais_spade_llm.resources.recovery_support_departure import _matrix, _object_id


_LIMIT = 4096
_mating_corridor_cache: dict | None = None


def _number(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def _pose(value, label: str) -> list[float]:
    return [float(item) for item in _unit_pose(value, label)]


def _owner_inputs(owner_evidence: dict) -> dict:
    return {"execution_mode": owner_evidence["execution_mode"],
            "source": deepcopy(owner_evidence.get("source")),
            "controller_config": deepcopy(owner_evidence["controller_config"]),
            "inputs": {"geometry": deepcopy(owner_evidence["inputs"]["geometry"])}}


def _pose_matches(first, second) -> bool:
    from cais_spade_llm.resources.recovery_geometry import (
        _cartesian_endpoint_residual, owner_cartesian_endpoint_precision,
    )

    residual = _cartesian_endpoint_residual(_pose(first, "first pose"), _pose(second, "second pose"))
    precision = owner_cartesian_endpoint_precision()
    return (residual["position_residual"] <= precision["position_precision"]
            and residual["orientation_residual"] <= precision["orientation_precision"])


def _binding(resource_id: str, state: dict, evidence: dict, owner_evidence: dict) -> dict:
    return {"state_fingerprint": recovery_validation_fingerprint(state),
        "model_fingerprint": recovery_validation_fingerprint({key: value for key, value in
            evidence["resource_models"][resource_id].items() if key != "event_projection_contracts"}),
        "owner_evidence_fingerprint": recovery_validation_fingerprint(_owner_inputs(owner_evidence)),
        "physics_fingerprint": recovery_validation_fingerprint(evidence["physics_evidence"]),
        "scene_object_bindings_fingerprint": recovery_validation_fingerprint({
            "parts": evidence.get("scene_object_bindings"),
            "supports": evidence.get("support_collision_object_bindings")}),
        "peer_fingerprint": recovery_validation_fingerprint({rid: {"state": row,
            "model": evidence.get("resource_models", {}).get(rid),
            "observation": evidence.get("resources", {}).get(rid)}
            for rid, row in state["resources"].items() if rid != resource_id})}


def prepare_mating_contact(*, resource_id: str, part_name: str, state: dict, evidence: dict,
                           owner_evidence: dict, target_origin_pose: dict,
                           target_tool_pose: list) -> dict:
    """Bind a declared insertion corridor to its exact projected starting scene.

    Args:
        resource_id: Exact owning resource identity.
        part_name: Exact declared part identity.
        state: Current accepted or privately projected complete snapshot.
        evidence: Recorded models, geometry and collision identities.
        owner_evidence: Recorded controller configuration and geometry declarations.
        target_origin_pose: Calculated declared part-origin destination.
        target_tool_pose: Tool pose calculated using the supported attachment.

    Returns:
        Prepared private policy evidence, or explicit missing/conflicting facts.
    """
    result = {"status": "NEEDS_CONTEXT", "version": 1, "command_sent": False,
              "execution_authorized": False, "continuous_collision_verified": False}
    try:
        inputs = _owner_inputs(owner_evidence)
        config, geometry = inputs["controller_config"], inputs["inputs"]["geometry"]
        if inputs["execution_mode"] != "simulation" or config["payload_collision"].get("enabled") is not True:
            raise ValueError("Mating evidence requires the recorded simulation payload-collision policy")
        board = geometry["assembly_board"]
        policy = board["mating_contact_by_part"][part_name]
        for field in ("axis_tolerance_m", "max_contact_depth_m"):
            if not _number(policy[field]) or not 0 < policy[field] <= .0005:
                raise ValueError("Mating tolerance exceeds the existing owner declaration bounds")
        if (not _number(policy["target_yaw_rad"]) or not _number(policy["yaw_tolerance_rad"])
                or not 0 < policy["yaw_tolerance_rad"] < math.pi / 2):
            raise ValueError("Recorded mating orientation bounds are unavailable")
        ignored = policy.get("ignore_tooth_contact_with", [])
        if (not isinstance(ignored, list) or any(not isinstance(value, str) or not value for value in ignored)
                or len(set(ignored)) != len(ignored)):
            raise ValueError("Recorded tooth-contact identities are invalid")
        part, owner = state["parts"][part_name], state["resources"][resource_id]
        if owner.get("held_part") != part_name or sum(row.get("held_part") == part_name
                for row in state["resources"].values()) != 1:
            raise ValueError("Mating requires unambiguous recorded or projected custody")
        model_name = geometry["parts"]["model_map"][part_name]
        if part["model_name"] != model_name:
            raise ValueError("Mating part identity differs from the captured model declaration")
        target_id = geometry["parts"]["assembly_target_map"][part_name] + "/collision"
        if policy["target_collision_object"] != target_id:
            raise ValueError("Mating target differs from the declared assembly target")
        slot, center = board["slots"][part_name], board["center"]
        expected_origin = {"x": float(center["x"]) + float(slot[0]),
                           "y": float(center["y"]) + float(slot[1]),
                           "z": float(board["target_origin_z_m"][part_name])}
        if (set(target_origin_pose) != {"x", "y", "z"}
                or any(not _number(target_origin_pose[axis]) for axis in expected_origin)
                or target_origin_pose != expected_origin):
            raise ValueError("Mating destination differs from recorded geometry")
        physics = evidence["physics_evidence"]
        body = physics["models"][model_name]
        if body.get("joint_count") != 0 or len(body["links"]) != 1:
            raise ValueError("Mating requires a recorded rigid part model")
        own = [row for link in body["links"].values() for row in link["collisions"]]
        if len(own) != 1:
            raise ValueError("Mating requires one exactly identified part collision object")
        part_object_id = _object_id(own[0]["id"], evidence)
        if part_object_id != model_name + "/link/collision":
            raise ValueError("Mating part collision identity differs from its owner policy")
        collision_sources = {}
        for name, recorded in physics["models"].items():
            for link in recorded["links"].values():
                for collision in link["collisions"]:
                    try:
                        object_id = _object_id(collision["id"], evidence)
                    except ValueError:
                        continue
                    if object_id in {target_id, *ignored}:
                        if object_id in collision_sources:
                            raise ValueError("Mating collision identity is ambiguous")
                        collision_sources[object_id] = {"model_name": name, "static": recorded.get("static"),
                            "collision_id": collision["id"], "geometry_fingerprint": recovery_validation_fingerprint(collision)}
        if set(collision_sources) != {target_id, *ignored} or collision_sources[target_id]["static"] is not True:
            raise ValueError("Exact static target and declared tooth-contact geometry are required")
        configuration = evidence["resource_models"][resource_id]["configuration"]
        if (owner["joint_names"] != configuration["joint_names"]
                or config["move_group"]["ee_link"] != configuration["reference_link"]
                or config["move_group"]["frame_id"] != state["frame"]):
            raise ValueError("Mating frames, link or joint identities differ from the recorded owner")
        positions = {name: interval(value) for name, value in
                     zip(owner["joint_names"], owner["joint_positions"], strict=True)}
        matrices = kinematic_transforms(configuration, positions)
        tool = representative_pose(matrices[configuration["reference_link"]])
        grasp = _pose(owner["grasp_transform"], "recorded grasp transform")
        if not _pose_matches(tool, owner["current_pose"]):
            raise ValueError("Mating start joints and tool pose conflict")
        actual = compose(tool, grasp)
        if not _pose_matches(actual, part["current_pose"]):
            raise ValueError("Mating attachment and part pose conflict")
        target = _pose(target_tool_pose, "mating target tool pose")
        target_part = compose(target, grasp)
        authorization = {**deepcopy(policy), "model_name": model_name,
            "target_origin_pose": deepcopy(expected_origin), "start_part_z": actual[2]}
        if not mating_pose_valid(authorization, actual) or not mating_pose_valid(authorization, target_part):
            raise ValueError("Mating start or endpoint is outside the declared part corridor")
        tolerance = policy["axis_tolerance_m"]
        if (math.hypot(tool[0] - target[0], tool[1] - target[1]) > tolerance
                or abs(sum(a * b for a, b in zip(tool[3:], target[3:], strict=True))) < math.cos(.01)
                or target[2] > tool[2] + tolerance):
            raise ValueError("Mating evidence requires the owner's aligned vertical insertion stage")
        return {**result, "status": "prepared", "resource_id": resource_id, "part_name": part_name,
            "model_name": model_name, "part_object_id": part_object_id,
            "reference_link": configuration["reference_link"], "frame_id": state["frame"],
            "simulation_mating_contact": deepcopy(policy), "target_origin_pose": deepcopy(expected_origin),
            "target_tool_pose": target, "start_tool_pose": tool, "start_part_pose": actual,
            "start_part_z": actual[2], "tool_to_part_pose": grasp,
            "collision_sources": collision_sources, "binding": _binding(resource_id, state, evidence, inputs),
            "max_refinements": _LIMIT, "source": {"kind": "recorded declaration and projected attachment",
                "owner_source": deepcopy(inputs["source"]), "physics_instance_id": physics.get("instance_id"),
                "physics_simulation_time": physics.get("simulation_time"),
                "policy_path": f"/inputs/geometry/assembly_board/mating_contact_by_part/{part_name}"}}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}


def _record_matches(expected: dict, actual: dict) -> bool:
    # The client adds the outer issued-request binding after preparation.
    return all(key in actual and actual[key] == value for key, value in expected.items())


def validate_mating_contact_samples(record: dict) -> None:
    """Verify every retained native waypoint and exact sampled contact policy.

    This does not establish continuous collision or contact-depth coverage.
    """
    text = record.get("mating_contact_json")
    if (not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest()
            != record.get("mating_contact_fingerprint")):
        raise ValueError("Native mating request fingerprint differs")
    prepared = json.loads(text)
    if record.get("mating_contact") is not None and prepared != record["mating_contact"]:
        raise ValueError("Native mating request differs from retained evidence")
    audit = record.get("native_mating_contact_evidence")
    if (not isinstance(audit, dict) or audit.get("version") != 1
            or audit.get("source") != "PreviewRecoveryMotion" or audit.get("pairs_complete") is not True
            or audit.get("continuous_collision_verified") is not False):
        raise ValueError("Complete native mating contact evidence is unavailable")
    preview_binding = prepared.get("preview_binding")
    if preview_binding is not None:
        actual_binding = audit.get("request_binding")
        if (not isinstance(actual_binding, dict)
                or any(actual_binding.get(key) != value for key, value in preview_binding.items())
                or actual_binding.get("mating_contact_fingerprint") != record["mating_contact_fingerprint"]):
            raise ValueError("Native mating audit request binding differs")
    points, samples = record.get("joint_trajectory", {}).get("points"), audit.get("samples")
    if (not isinstance(points, list) or len(points) < 2 or not isinstance(samples, list)
            or len(samples) != len(points)):
        raise ValueError("Native mating audit does not cover every trajectory waypoint")
    authorization = {**prepared["simulation_mating_contact"], "model_name": prepared["model_name"],
        "target_origin_pose": prepared["target_origin_pose"], "start_part_z": prepared["start_part_z"]}
    names = record["joint_trajectory"]["joint_names"]
    for index, (sample, point) in enumerate(zip(samples, points, strict=True)):
        if sample.get("state_index") != index or type(sample.get("state_index")) is not int:
            raise ValueError("Native mating waypoint order differs")
        joints = sample.get("joint_state") or {}
        if (not isinstance(joints.get("name"), list) or len(joints["name"]) != len(set(joints["name"]))
                or len(joints["name"]) != len(joints.get("position", []))):
            raise ValueError("Native mating joint inventory is invalid")
        positions = dict(zip(joints["name"], joints["position"], strict=True))
        if any(positions.get(name) != position for name, position in zip(names, point["positions"], strict=True)):
            raise ValueError("Native mating sample joints differ from the trajectory")
        pose = _pose(sample.get("part_pose"), "sampled mating part pose")
        if not mating_pose_valid(authorization, pose):
            raise ValueError("Native mating sample leaves the declared part corridor")
        contacts = sample.get("contacts")
        count = sample.get("checked_contacts")
        if (not isinstance(contacts, list) or type(count) is not int or count != len(contacts)
                or not 0 <= count < _LIMIT):
            raise ValueError("Native mating contact inventory is incomplete")
        if any(not isinstance(row, dict) or not isinstance(row.get("body_1"), str)
               or not isinstance(row.get("body_2"), str) or not _number(row.get("depth"))
               or row["depth"] < 0 for row in contacts):
            raise ValueError("Native mating contact inventory has invalid pairs or depths")
        native_contacts = [SimpleNamespace(contact_body_1=row["body_1"], contact_body_2=row["body_2"],
                                          depth=row["depth"]) for row in contacts]
        if contacts and not mating_contacts_allowed(authorization, native_contacts):
            raise ValueError("Native mating contact is unrelated or exceeds the declared depth")
    if not _pose_matches(samples[0]["part_pose"], prepared["start_part_pose"]):
        raise ValueError("Native mating audit starts at a different part pose")


def _corridor_bounds(matrix: list, prepared: dict) -> tuple[bool, dict]:
    policy, target = prepared["simulation_mating_contact"], prepared["target_origin_pose"]
    tolerance = policy["axis_tolerance_m"]
    dx, dy = add(matrix[0][3], interval(-target["x"])), add(matrix[1][3], interval(-target["y"]))
    radial_upper = add(mul(interval(max(abs(v) for v in dx)), interval(max(abs(v) for v in dx))),
                       mul(interval(max(abs(v) for v in dy)), interval(max(abs(v) for v in dy))))[1]
    z = matrix[2][3]
    c, s = cosine(interval(policy["target_yaw_rad"])), sine(interval(policy["target_yaw_rad"]))
    forward = add(mul(c, matrix[0][0]), mul(s, matrix[1][0]))
    lateral = add(mul(neg(s), matrix[0][0]), mul(c, matrix[1][0]))
    # atan2(lateral, forward) uses the exact declared yaw convention. The
    # inequalities avoid introducing a separate quaternion or angle tolerance.
    sin_t = sine(interval(policy["yaw_tolerance_rad"]))
    cos_t = cosine(interval(policy["yaw_tolerance_rad"]))
    yaw_limit = mul(sin_t, forward)
    yaw_error = mul(cos_t, interval(max(abs(v) for v in lateral)))
    upright_limit = cosine(interval(.02))
    bounds = {"radial_squared_upper": radial_upper, "z": list(z),
              "yaw_forward": list(forward), "yaw_lateral": list(lateral),
              "upright": list(matrix[2][2])}
    allowed = (radial_upper <= mul(interval(tolerance), interval(tolerance))[0]
        and z[0] >= add(interval(target["z"]), interval(-tolerance))[1]
        and z[1] <= add(interval(prepared["start_part_z"]), interval(tolerance))[0] and forward[0] > 0
        and yaw_error[1] <= yaw_limit[0] and matrix[2][2][0] >= upright_limit[1])
    return allowed, bounds


def _corridor_violations(matrix: list, prepared: dict) -> list[dict]:
    """Return only inequalities whose outward bounds establish a violation."""
    policy, target = prepared["simulation_mating_contact"], prepared["target_origin_pose"]
    tolerance = interval(policy["axis_tolerance_m"])
    dx, dy = add(matrix[0][3], interval(-target["x"])), add(matrix[1][3], interval(-target["y"]))
    distances = [interval(0. if row[0] <= 0 <= row[1] else min(abs(v) for v in row),
                          max(abs(v) for v in row)) for row in (dx, dy)]
    radial = add(mul(distances[0], distances[0]), mul(distances[1], distances[1]))
    c, s = cosine(interval(policy["target_yaw_rad"])), sine(interval(policy["target_yaw_rad"]))
    forward = add(mul(c, matrix[0][0]), mul(s, matrix[1][0]))
    lateral = add(mul(neg(s), matrix[0][0]), mul(c, matrix[1][0]))
    sin_t, cos_t = sine(interval(policy["yaw_tolerance_rad"])), cosine(interval(policy["yaw_tolerance_rad"]))
    yaw_limit = mul(sin_t, forward)
    checks = [
        ("axis_tolerance_m", ">", radial, mul(tolerance, tolerance)),
        ("target_origin_pose.z", "<", matrix[2][3], add(interval(target["z"]), neg(tolerance))),
        ("start_part_z", ">", matrix[2][3], add(interval(prepared["start_part_z"]), tolerance)),
        ("upright", "<", matrix[2][2], cosine(interval(.02))),
        ("yaw_tolerance_rad", ">", mul(cos_t, lateral), yaw_limit),
        ("yaw_tolerance_rad", ">", mul(cos_t, neg(lateral)), yaw_limit),
    ]
    violations = []
    for field, relation, value, limit in checks:
        if (value[0] > limit[1] if relation == ">" else value[1] < limit[0]):
            violations.append({"condition": field, "relation": relation,
                               "value_bounds": list(value), "limit_bounds": list(limit)})
    return violations


def verify_mating_contact(*, resource_id: str, state: dict, evidence: dict, owner_evidence: dict,
                          joint_trajectory: dict, mating_contact: dict) -> dict:
    """Prove the declared part corridor over the unchanged interpolated trajectory.

    Native sampled collisions remain a separate required record. This proof
    does not authorize a trajectory or establish continuous contact depth.
    """
    global _mating_corridor_cache

    result = {"status": "NEEDS_CONTEXT", "continuous_mating_corridor_verified": False,
              "continuous_collision_verified": False, "physical_execution_verified": False}
    try:
        prepared = prepare_mating_contact(resource_id=resource_id, part_name=mating_contact["part_name"],
            state=state, evidence=evidence, owner_evidence=owner_evidence,
            target_origin_pose=mating_contact["target_origin_pose"], target_tool_pose=mating_contact["target_tool_pose"])
        if prepared.get("status") != "prepared" or not _record_matches(prepared, mating_contact):
            raise ValueError("Mating evidence does not match its recorded declaration and current scene: "
                             + str(prepared.get("reason", "binding differs")))
        key = recovery_validation_fingerprint({"prepared": prepared, "joint_trajectory": joint_trajectory})
        calculations = (ContinuousMotion, kinematic_transforms, polynomial_bounds, quaternion_transform,
                        _matrix, _corridor_bounds, _corridor_violations, _pose_matches, representative_pose,
                        mating_pose_valid)
        cached = _mating_corridor_cache
        if cached is not None and cached["key"] == key and cached["calculations"] == calculations:
            return deepcopy(cached["result"])
        _mating_corridor_cache = None
        configuration = evidence["resource_models"][resource_id]["configuration"]
        motion = ContinuousMotion(joint_trajectory, configuration)
        if motion.points[0]["positions"] != state["resources"][resource_id]["joint_positions"]:
            raise ValueError("Mating trajectory starts from a different joint configuration")
        grasp = prepared["tool_to_part_pose"]
        attachment = quaternion_transform(grasp[:3], grasp[3:])
        cells, explored = [], 0
        for segment, (a, b) in enumerate(zip(motion.exact_times, motion.exact_times[1:], strict=False)):
            pending = [(Fraction(0), Fraction(1))]
            while pending:
                low, high = pending.pop()
                explored += 1
                if explored > _LIMIT:
                    raise ValueError("Continuous mating corridor exceeded 4096 refinement cells")
                positions = {name: polynomial_bounds(row, low, high) for name, row in
                             zip(motion.names, motion.coefficients[segment], strict=True)}
                for name, error in motion.joint_position_error.items():
                    positions[name] = add(positions[name], interval(-error, error))
                matrices = kinematic_transforms(configuration, positions, motion.origins)
                matrix = _matrix(matrices[configuration["reference_link"]], attachment)
                allowed, bounds = _corridor_bounds(matrix, prepared)
                if allowed:
                    cells.append({"segment": segment, "start": float(a + low * (b - a)),
                                  "end": float(a + high * (b - a)), "bounds": bounds})
                    continue
                midpoint = (low + high) / 2
                midpoint_positions = {name: polynomial_bounds(row, midpoint, midpoint) for name, row in
                                      zip(motion.names, motion.coefficients[segment], strict=True)}
                for name, error in motion.joint_position_error.items():
                    midpoint_positions[name] = add(midpoint_positions[name], interval(-error, error))
                middle = kinematic_transforms(configuration, midpoint_positions, motion.origins)[configuration["reference_link"]]
                part_matrix = _matrix(middle, attachment)
                violations = _corridor_violations(part_matrix, prepared)
                if violations:
                    return {**result, "status": "rejected",
                        "reason": "The interpolated part leaves the declared mating corridor",
                        "mating_contact_fingerprint": recovery_validation_fingerprint(prepared),
                        "trajectory_fingerprint": recovery_validation_fingerprint(joint_trajectory),
                        "binding": deepcopy(prepared["binding"]), "counterexample": {
                            "segment": segment, "time_from_start": float(a + midpoint * (b - a)),
                            "fraction": str(midpoint), "joint_intervals": {
                                name: list(value) for name, value in midpoint_positions.items()},
                            "part_transform_bounds": [[list(value) for value in row] for row in part_matrix],
                            "violated_conditions": violations}}
                actual = representative_pose(part_matrix)
                authorization = {**prepared["simulation_mating_contact"], "model_name": prepared["model_name"],
                    "target_origin_pose": prepared["target_origin_pose"], "start_part_z": prepared["start_part_z"]}
                if not mating_pose_valid(authorization, actual):
                    raise ValueError("The interpolated part's mating corridor remains unresolved")
                if high - low <= Fraction(1, 2**48):
                    raise ValueError("Continuous mating corridor remains unresolved")
                pending.extend(((midpoint, high), (low, midpoint)))
        endpoint = kinematic_transforms(configuration, {name: interval(value) for name, value in
            zip(motion.names, motion.points[-1]["positions"], strict=True)}, motion.origins)[configuration["reference_link"]]
        if not _pose_matches(representative_pose(endpoint), prepared["target_tool_pose"]):
            raise ValueError("Mating trajectory endpoint differs from its calculated target")
        supported = {**result, "status": "supported", "continuous_mating_corridor_verified": True,
            "mating_contact_fingerprint": recovery_validation_fingerprint(prepared),
            "trajectory_fingerprint": recovery_validation_fingerprint(joint_trajectory),
            "binding": deepcopy(prepared["binding"]), "refinement_cells": explored, "cells": cells,
            "model_assumptions": ["The part remains rigidly attached to the recorded reference link",
                "The trajectory follows the recorded spline interpolation",
                "The static assembly fixture and peer configuration remain unchanged"],
            "sampled_contact_depth_requires_native_evidence": True}
        # Only a local successful calculation populates this entry. Requests,
        # native contact audits and supplied proof records are checked separately.
        _mating_corridor_cache = {"key": key, "calculations": calculations, "result": deepcopy(supported)}
        return supported
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}


def projected_fixture_handoff(*, resource_id: str, part_name: str, state: dict, evidence: dict,
                              owner_evidence: dict, mating_contact: dict, endpoint_evidence: dict) -> dict:
    """Establish the declared private fixture attachment after a supported seat.

    The resulting fixed attachment is a model prediction, never an observed
    release, attachment acknowledgement, settling outcome, or task completion.
    """
    result = {"status": "NEEDS_CONTEXT", "source": {"kind": "projected fixture attachment"},
              "physical_execution_verified": False, "execution_authorized": False, "settling_modeled": False}
    try:
        native = endpoint_evidence["mating_native_record"]
        validate_mating_contact_samples(native)
        if json.loads(native["mating_contact_json"]) != mating_contact:
            raise ValueError("Fixture handoff uses a different mating request")
        start = endpoint_evidence["mating_start_state"]
        if (state["resources"][resource_id].get("held_part") != part_name
                or any(row.get("held_part") == part_name for rid, row in state["resources"].items()
                       if rid != resource_id)):
            raise ValueError("Fixture handoff requires the same unambiguous owner custody before release")
        if {rid: row for rid, row in state["resources"].items() if rid != resource_id} != {
                rid: row for rid, row in start["resources"].items() if rid != resource_id}:
            raise ValueError("Fixture handoff peer states differ from the validated insertion scene")
        # Revalidation uses the original complete scene, whose fingerprint was
        # fixed before this private insertion interval, not the later endpoint.
        start_evidence = {**evidence, "snapshot": start}
        proof = verify_mating_contact(resource_id=resource_id, state=start, evidence=start_evidence,
            owner_evidence=owner_evidence, joint_trajectory=native["joint_trajectory"], mating_contact=mating_contact)
        if proof.get("status") != "supported" or proof != endpoint_evidence["mating_corridor_evidence"]:
            raise ValueError("Fixture handoff lacks matching continuous mating evidence")
        policy = mating_contact["simulation_mating_contact"]
        if policy.get("retain_fixture_attachment") is not True or mating_contact["part_name"] != part_name:
            raise ValueError("A retained fixture attachment is not declared for this part")
        last = native["joint_trajectory"]["points"][-1]
        positions, velocities = last["positions"], last.get("velocities")
        if (not isinstance(velocities, list) or len(velocities) != len(positions)
                or any(value != 0 for value in velocities)
                or state["resources"][resource_id]["joint_positions"] != positions):
            raise ValueError("Fixture handoff requires the supported stopped insertion endpoint")
        configuration = evidence["resource_models"][resource_id]["configuration"]
        transforms = kinematic_transforms(configuration, dict(zip(configuration["joint_names"],
            [interval(value) for value in positions], strict=True)))
        endpoint = compose(representative_pose(transforms[configuration["reference_link"]]), mating_contact["tool_to_part_pose"])
        part = state["parts"][part_name]
        if not _pose_matches(part["current_pose"], endpoint):
            raise ValueError("Fixture handoff part pose differs from the supported insertion endpoint")
        seated = {**policy, "target_origin_pose": mating_contact["target_origin_pose"],
                  "start_part_z": mating_contact["target_origin_pose"]["z"]}
        if not mating_pose_valid(seated, endpoint):
            raise ValueError("Fixture handoff requires the declared seated part pose")
        geometry = owner_evidence["inputs"]["geometry"]
        board = geometry["assembly_board"]
        fixture_name = board["component_attachment_model"]
        if fixture_name != board["model_name"]:
            raise ValueError("Fixture attachment and board identity differ")
        physics = evidence["physics_evidence"]
        fixture = physics["models"][fixture_name]
        if fixture.get("static") is not True or "link" not in fixture["links"]:
            raise ValueError("Fixture handoff requires its exact recorded static link")
        config = owner_evidence["controller_config"]
        if not config["services"].get("attach") or not config["services"].get("detach"):
            raise ValueError("The owner has no recorded fixture attachment interfaces")
        model_name = part["model_name"]
        if model_name != mating_contact["model_name"] or "link" not in physics["models"][model_name]["links"]:
            raise ValueError("Fixture handoff part link identity differs")
        pose = _pose(fixture["links"]["link"]["pose"], "fixture link pose")
        inverse_q = [-value for value in pose[3:6]] + pose[6:]
        inverse = [*rotate(inverse_q, [-value for value in pose[:3]]), *inverse_q]
        attachment = compose(inverse, endpoint)
        return {**result, "status": "supported", "part_name": part_name, "model_name": model_name,
            "kind": "projected_fixture_handoff", "seated": True, "retained_fixture": True,
            "current_pose": deepcopy(part["current_pose"]), "fixture_model": fixture_name, "fixture_link": "link",
            "target_origin_pose": deepcopy(mating_contact["target_origin_pose"]),
            "target_collision_object": policy["target_collision_object"],
            "part_link": "link", "fixture_to_part_pose": attachment,
            "mating_contact_fingerprint": recovery_validation_fingerprint(mating_contact),
            "trajectory_fingerprint": recovery_validation_fingerprint(native["joint_trajectory"]),
            "fixture_fingerprint": recovery_validation_fingerprint(fixture),
            "policy_fingerprint": recovery_validation_fingerprint(policy),
            "model_assumptions": ["The declared fixture attachment succeeds before subsequent motion",
                "The recorded static fixture remains fixed"],
            "source": {"kind": "projected fixture attachment", "owner_function":
                "GazeboPickPlaceController._detach_part", "capture_source": deepcopy(owner_evidence.get("source"))}}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}
