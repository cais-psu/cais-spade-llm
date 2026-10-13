from __future__ import annotations

"""Recorded geometry and rule-local recovery outline checks without live calls."""

import asyncio
import hashlib
import json
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from cais_spade_llm.agents.central_controller.outline_evidence_safety import (
    validate_outline_evidence_safety,
)
from cais_spade_llm.resources.recovery_geometry import (
    outline_event_binding,
    project_outline_evidence,
    project_recovery_outline_event,
    project_recovery_geometry,
)
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint


def _placement_arrival_model():
    steps = [{"id": "targets", "op": "compute_place_targets", "store_as": "placement"}]
    for name in ("approach_pose", "target_pose"):
        steps.append({"id": name, "op": "move_cartesian", "params": {
            key: {"$step": "placement", "path": [name, key]}
            for key in ("x", "y", "z", "qx", "qy", "qz", "qw")}})
    return {"events": [{"event_name": "place_approach", "composition_contract": {"steps": steps}}]}


@pytest.mark.parametrize("release", [False, True])
def test_outline_place_arrival_uses_declared_approach_and_recorded_insertion_depth(release):
    from cais_spade_llm.resources.resource_safety_preparation import _outline_place_arrival

    model = _placement_arrival_model()
    task = {"event_name": "new_proposed_event", "part_name": "gear_small"}
    target = {"insert_pose": {"x": .1, "y": .2, "z": .3,
                              "qx": 0., "qy": 0., "qz": 1., "qw": 0.}}
    evidence = {"controller_config": {"parts_tuning": {"insertion_depth_m": .0025}}}
    before = deepcopy((task, model, target, evidence))
    result = _outline_place_arrival(task=task, model=model, target=target,
                                    resource_evidence=evidence, release=release)
    requests = result["motion_requests"]
    assert len(requests) == 2
    assert [waypoint["position"]["z"] for request in requests for waypoint in request["waypoints"]] == (
        [.3 + .0025 + .05, .3 + .0025, .3] if release else [.3 + .0025 + .05, .3 + .0025])
    assert all(waypoint["position"]["x"] == .1 and waypoint["position"]["y"] == .2
               and waypoint["orientation"] == {"x": 0., "y": 0., "z": 1., "w": 0.}
               for request in requests for waypoint in request["waypoints"])
    assert result["mating_motion_indices"] == [1]
    contributor = result["source"]["owner_declarations"][0]
    assert contributor["event_name"] == "place_approach"
    assert contributor["event_fingerprint"] == recovery_validation_fingerprint(model["events"][0])
    assert result["source"]["insertion_depth_m"] == .0025
    assert (task, model, target, evidence) == before


def test_outline_place_arrival_retains_required_nominal_release_checkpoint():
    from cais_spade_llm.resources.resource_safety_preparation import _outline_place_arrival

    task = {"required_task_contract": "place_release", "part_name": "gear_small"}
    pose = {"x": .1, "y": .2, "z": .3, "qx": 0., "qy": 0., "qz": 0., "qw": 1.}
    result = _outline_place_arrival(task=task, model=_placement_arrival_model(),
        target={"insert_pose": pose},
        resource_evidence={"controller_config": {"parts_tuning": {"insertion_depth_m": .0025}}},
        release=True)
    assert result["motion_requests"] == [{"mode": "cartesian", "waypoints": [{
        "position": {"x": .1, "y": .2, "z": .3},
        "orientation": {"x": 0., "y": 0., "z": 0., "w": 1.}}]}]
    assert result["mating_motion_indices"] == [0]
    assert result["source"]["required_task_contract"] == "place_release"


@pytest.mark.parametrize("depth", [None, True, -.001, float("nan"), float("inf")])
def test_outline_place_arrival_requires_recorded_finite_depth(depth):
    from cais_spade_llm.resources.resource_safety_preparation import _outline_place_arrival

    with pytest.raises(ValueError, match="recorded insertion depth"):
        _outline_place_arrival(task={}, model=_placement_arrival_model(), target={},
            resource_evidence={"controller_config": {"parts_tuning": {"insertion_depth_m": depth}}},
            release=True)


@pytest.mark.parametrize("defect", ["absent", "conditional", "different_binding", "extra_step"])
def test_outline_place_arrival_requires_supported_declared_waypoints(defect):
    from cais_spade_llm.resources.resource_safety_preparation import _outline_place_arrival

    model = _placement_arrival_model()
    steps = model["events"][0]["composition_contract"]["steps"]
    if defect == "absent":
        model["events"] = []
    elif defect == "conditional":
        steps[1]["when"] = {"unsupported": True}
    elif defect == "different_binding":
        steps[2]["params"]["z"]["path"] = ["insert_pose", "z"]
    else:
        steps.append({"id": "extra", "op": "move_relative"})
    pose = {"x": .1, "y": .2, "z": .3, "qx": 0., "qy": 0., "qz": 0., "qw": 1.}
    with pytest.raises(ValueError, match="recorded target-calculation and waypoint declaration"):
        _outline_place_arrival(task={}, model=model, target={"insert_pose": pose},
            resource_evidence={"controller_config": {"parts_tuning": {"insertion_depth_m": .0025}}},
            release=True)


def _support_departure_case():
    from cais_spade_llm.resources.recovery_geometry import gravity_supported_part

    scene, _, _, task = _custody_event_case(False)
    rid, part = "ur5e-3", task["part_name"]
    configuration = scene["resource_models"][rid]["configuration"]
    configuration["joints"][0]["axis"] = [0., 0., 1.]
    owner = scene["snapshot"]["resources"][rid]
    owner.update(joint_positions=[0.], joint_names=["slide"], current_pose=[0., 0., 0., 0., 0., 0., 1.])
    support = gravity_supported_part(model_name="square", physics=scene["physics_evidence"],
        policy=scene["part_support_policy"], owners_idle=True)
    scene["stationary_part_evidence"] = {part: {**support, "current_pose": [0., 0., 0., 0., 0., 0., 1.]}}
    scene["scene_object_bindings"] = {part: {"objects": [{"object_id": "square/link/collision",
        "recorded_geometry": {"collision_id": "square::collision"}}]}}
    scene["support_collision_object_bindings"] = {"table::collision": {
        "object_id": "table/link/collision", "source": "recorded fixture declaration"}}
    trajectory = {"joint_names": ["slide"], "duration_ns": 1_000_000_000, "points": [
        {"positions": [position], "time_from_start": {"sec": index, "nanosec": 0}}
        for index, position in enumerate((0., .2))]}
    return scene, rid, trajectory


def _support_departure_audit(support):
    def pair_rows(index):
        return [{"part_object_id": row["part_object_id"], "support_object_id": row["support_object_id"],
            "contact": index == 0, "maximum_depth": row["initial_penetration_bound"] if index == 0 else 0.,
            "minimum_signed_distance": -row["initial_penetration_bound"] if index == 0 else .15}
            for row in support["pairs"]]
    return {"version": 1, "source": "PreviewRecoveryMotion", "pairs_complete": True,
        "continuous_collision_verified": False, "samples": [
            {"state_index": index, "checked_contacts": len(support["pairs"]) if index == 0 else 0,
             "pairs": pair_rows(index)} for index in (0, 1)],
        "pairs": [{"part_object_id": row["part_object_id"], "support_object_id": row["support_object_id"],
            "first_separation_index": 1, "endpoint_separated": True,
            "initial_penetration_bound": row["initial_penetration_bound"]} for row in support["pairs"]]}


@pytest.mark.parametrize("defect", [None, "missing_audit", "wire_hash", "incomplete_contacts", "unknown_pair",
                                   "deeper", "false_continuous_claim", "reentry", "missing_separation",
                                   "missing_initial", "missing_terminal", "missing_interior", "extra_waypoint"])
def test_support_departure_replay_requires_complete_separate_native_contact_audit(defect):
    from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure, validate_support_departure_samples

    scene, rid, trajectory = _support_departure_case()
    support = prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    text = json.dumps(support)
    record = {"support_departure": support, "support_departure_json": text,
        "support_departure_fingerprint": hashlib.sha256(text.encode()).hexdigest(),
        "joint_trajectory": trajectory,
        "native_support_departure_evidence": _support_departure_audit(support)}
    audit = record["native_support_departure_evidence"]
    if defect == "missing_audit":
        record.pop("native_support_departure_evidence")
    elif defect == "wire_hash":
        record["support_departure_json"] += " "
    elif defect == "incomplete_contacts":
        audit["samples"][0]["checked_contacts"] = 4096
    elif defect == "unknown_pair":
        audit["samples"][0]["pairs"][0]["support_object_id"] = "different_table"
    elif defect == "deeper":
        audit["samples"][0]["pairs"][0]["maximum_depth"] += .001
    elif defect == "false_continuous_claim":
        audit["continuous_collision_verified"] = True
    elif defect == "reentry":
        audit["samples"].append({**deepcopy(audit["samples"][0]), "state_index": 2})
    elif defect == "missing_separation":
        audit["pairs"][0]["endpoint_separated"] = False
    elif defect == "missing_initial":
        audit["samples"].pop(0)
    elif defect == "missing_terminal":
        audit["samples"].pop()
    elif defect == "missing_interior":
        trajectory["points"].insert(1, {"positions": [.1], "time_from_start": {"sec": 0, "nanosec": 500_000_000}})
        audit["samples"][-1]["state_index"] = 2
    elif defect == "extra_waypoint":
        audit["samples"].append({**deepcopy(audit["samples"][-1]), "state_index": 2})
    if defect is None:
        validate_support_departure_samples(record)
    else:
        with pytest.raises(ValueError):
            validate_support_departure_samples(record)


def test_support_departure_proves_declared_lift_without_adding_contact_tolerance():
    from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure, verify_support_departure

    scene, rid, trajectory = _support_departure_case()
    original = deepcopy(scene)
    record = prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    assert record["status"] == "prepared", record
    assert record["pairs"][0]["contact_tolerance_added"] == 0
    result = verify_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene,
        joint_trajectory=trajectory, support_departure=record)
    assert result["status"] == "supported", result
    assert result["continuous_support_departure_verified"] is True
    assert result["physical_execution_verified"] is False
    assert scene == original


@pytest.mark.parametrize("stale", [False, True])
def test_support_departure_replays_established_projected_deposit_before_next_pick(stale):
    from cais_spade_llm.resources.recovery_geometry import projected_part_support
    from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure, verify_support_departure

    scene, rid, trajectory = _support_departure_case()
    part = scene["snapshot"]["resources"][rid]["held_part"]
    scene["snapshot"]["resources"][rid]["grasp_transform"][0] = .2
    scene["snapshot"]["parts"][part]["current_pose"][0] = .2
    projected = projected_part_support(part_name=part, state=scene["snapshot"], evidence=scene)
    assert projected["status"] == "supported", projected
    scene["stationary_part_evidence"][part] = {key: value for key, value in projected.items() if key != "physics_evidence"}
    original = deepcopy(scene)
    if stale:
        scene["stationary_part_evidence"][part]["source"]["physics_fingerprint"] = "different source"
    support = prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    if stale:
        assert support["status"] == "NEEDS_CONTEXT", support
        assert "model calculation" in support["reason"]
    else:
        assert support["status"] == "prepared", support
        assert support["source"]["kind"] == "projected support and projected attachment"
        result = verify_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene,
            joint_trajectory=trajectory, support_departure=support)
        assert result["status"] == "supported", result
        assert scene == original
        assert scene["physics_evidence"]["models"]["square"]["pose"][0] == 0


@pytest.mark.parametrize("defect", ["missing_contacts", "missing_mapping", "moving_support", "changed_pose", "unsupported_shape"])
def test_support_departure_preserves_missing_or_conflicting_support(defect):
    from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure

    scene, rid, _ = _support_departure_case()
    if defect == "missing_contacts":
        scene["physics_evidence"]["contacts"] = []
    elif defect == "missing_mapping":
        scene["support_collision_object_bindings"] = {}
    elif defect == "moving_support":
        scene["physics_evidence"]["models"]["table"]["static"] = False
    elif defect == "changed_pose":
        scene["snapshot"]["parts"][scene["snapshot"]["resources"][rid]["held_part"]]["current_pose"][2] += .01
    else:
        scene["physics_evidence"]["models"]["square"]["links"]["link"]["collisions"][0]["geometry"] = {
            "type": "cylinder", "radius": .05, "length": .1}
    result = prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    assert result["status"] == "NEEDS_CONTEXT", result


@pytest.mark.parametrize("defect", ["deeper", "reentry", "no_departure", "changed_peer", "changed_mesh", "different_start", "tracking_error"])
def test_support_departure_rejects_unsupported_motion_and_stale_evidence(defect):
    from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure, verify_support_departure

    scene, rid, trajectory = _support_departure_case()
    record = prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    assert record["status"] == "prepared", record
    if defect == "deeper":
        trajectory["points"][-1]["positions"] = [-.1]
    elif defect == "reentry":
        trajectory["points"].append({"positions": [-.01], "time_from_start": {"sec": 2, "nanosec": 0}})
        trajectory["duration_ns"] = 2_000_000_000
    elif defect == "no_departure":
        trajectory["points"][-1]["positions"] = [0.]
    elif defect == "changed_peer":
        scene["snapshot"]["resources"]["Buffer For Machined parts"]["held_part"] = "other"
    elif defect == "changed_mesh":
        record["pairs"][0]["part_vertices"][0][0] += .001
    elif defect == "different_start":
        trajectory["points"][0]["positions"] = [.01]
    else:
        scene["resource_models"][rid]["configuration"]["joint_position_error"] = {"slide": .01}
        record = prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    result = verify_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene,
        joint_trajectory=trajectory, support_departure=record)
    assert result["status"] == "NEEDS_CONTEXT", result
    assert result["continuous_support_departure_verified"] is False


def test_support_departure_checks_native_spline_interior_not_only_endpoints():
    from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure, verify_support_departure

    scene, rid, trajectory = _support_departure_case()
    record = prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    trajectory["points"][0]["velocities"] = [-1.]
    trajectory["points"][1]["velocities"] = [0.]
    result = verify_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene,
        joint_trajectory=trajectory, support_departure=record)
    assert result["status"] == "NEEDS_CONTEXT", result
    assert "penetration" in result["reason"]


def test_support_departure_proves_zero_endpoint_velocity_spline():
    from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure, verify_support_departure

    scene, rid, trajectory = _support_departure_case()
    record = prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    for point in trajectory["points"]:
        point["velocities"] = [0.]
        point["accelerations"] = [0.]
    result = verify_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene,
        joint_trajectory=trajectory, support_departure=record)
    assert result["status"] == "supported", result


@pytest.mark.parametrize("acceleration,expected", [(-1e-15, "supported"), (-.01, "NEEDS_CONTEXT")])
def test_support_departure_compares_distance_with_original_bound_not_vertex_slope(acceleration, expected):
    from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure, verify_support_departure

    scene, rid, trajectory = _support_departure_case()
    scene["physics_evidence"]["models"]["table"]["links"]["link"]["collisions"][0]["pose"][2] += 1e-6
    for point in trajectory["points"]:
        point["velocities"] = [0.]
        point["accelerations"] = [0.]
    # The saved native lift has zero starting velocity and a small negative
    # normal derivative after its common endpoint factor is removed.
    trajectory["points"][0]["accelerations"] = [acceleration]
    record = prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    original = deepcopy(record)
    result = verify_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene,
        joint_trajectory=trajectory, support_departure=record)
    assert result["status"] == expected, result
    assert record == original
    assert record["pairs"][0]["contact_tolerance_added"] == 0
    if expected == "supported":
        bounded = [cell for cell in result["pairs"][0]["cells"]
            if cell["status"] == "within_initial_penetration_bound"]
        assert bounded
        assert any(cell["scaled_normal_derivative_bounds"][0] < 0 for cell in bounded)
        assert all(cell["signed_distance_bounds"][0] >= -record["pairs"][0]["initial_penetration_bound"]
                   and cell["signed_distance_bounds"][1] <= 0 for cell in bounded)
    else:
        assert "increases its initial support penetration" in result["reason"]


@pytest.mark.parametrize("depth,expected", [(.01, "supported"), (.001, "NEEDS_CONTEXT")])
def test_support_departure_distinguishes_bounded_contact_from_hidden_separation_and_reentry(depth, expected):
    from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure, verify_support_departure

    scene, rid, trajectory = _support_departure_case()
    scene["physics_evidence"]["models"]["table"]["links"]["link"]["collisions"][0]["pose"][2] += depth
    trajectory["points"] = [
        {"positions": [position], "velocities": [velocity], "time_from_start": {"sec": index, "nanosec": 0}}
        for index, (position, velocity) in enumerate(((0., .02), (.0005, -.015), (.2, 0.)))]
    trajectory["duration_ns"] = 2_000_000_000
    record = prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    result = verify_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene,
        joint_trajectory=trajectory, support_departure=record)
    assert result["status"] == expected, result
    if expected == "supported":
        assert result["pairs"][0]["cells"][0]["status"] == "within_initial_penetration_bound"
        assert result["pairs"][0]["cells"][-1]["status"] == "separated"
    else:
        assert "re-enters its supporting surface" in result["reason"]


@pytest.mark.parametrize("reference_joint", [False, True])
def test_support_departure_preserves_missing_reference_joint_interval_derivatives(reference_joint):
    from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure, verify_support_departure

    scene, rid, trajectory = _support_departure_case()
    configuration = scene["resource_models"][rid]["configuration"]
    configuration["joints"].append({"name": "turn", "parent": "body", "child": "tool", "type": "revolute",
        "xyz": [0., 0., 0.], "rpy": [0., 0., 0.], "axis": [0., 0., 1.],
        "frozen_position": 0., "frozen_interval": [-.01, .01]})
    if reference_joint:
        configuration["reference_link"] = "tool"
    record = prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    assert record["status"] == "prepared", record
    result = verify_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene,
        joint_trajectory=trajectory, support_departure=record)
    if reference_joint:
        assert result["status"] == "NEEDS_CONTEXT", result
        assert "reference-link joint interval" in result["reason"]
    else:
        assert result["status"] == "supported", result


@pytest.mark.parametrize("axis,expected", [([0., 0., 1.], "supported"), ([1., 0., 0.], "NEEDS_CONTEXT")])
def test_support_departure_uses_complete_rotating_mesh_not_only_tool_height(axis, expected):
    from itertools import product
    from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure, verify_support_departure

    scene, rid, trajectory = _support_departure_case()
    configuration = scene["resource_models"][rid]["configuration"]
    configuration["joint_names"].append("turn")
    configuration["joints"].append({"name": "turn", "parent": "body", "child": "tool", "type": "revolute",
        "xyz": [0., 0., 0.], "rpy": [0., 0., 0.], "axis": axis})
    configuration["reference_link"] = "tool"
    owner = scene["snapshot"]["resources"][rid]
    owner["joint_names"].append("turn")
    owner["joint_positions"].append(0.)
    mesh = {"uri": "recorded-square-mesh", "source_kind": "recorded_collision_mesh", "source_sha256": "saved-mesh-hash",
        "vertices": [list(row) for row in product((-.05, .05), repeat=3)], "vertex_count": 8}
    scene["physics_meshes"] = {mesh["uri"]: mesh}
    scene["physics_evidence"]["models"]["square"]["links"]["link"]["collisions"][0]["geometry"] = {
        "type": "mesh", "uri": mesh["uri"], "scale": [1., 1., 1.]}
    trajectory["joint_names"].append("turn")
    trajectory["points"][0]["positions"].append(0.)
    trajectory["points"][1]["positions"] = [.01, .5]
    support = prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    assert support["status"] == "prepared", support
    verified = verify_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene,
        joint_trajectory=trajectory, support_departure=support)
    assert verified["status"] == expected, verified


def test_support_departure_search_limit_preserves_unresolved_evidence(monkeypatch):
    from cais_spade_llm.resources import recovery_support_departure

    scene, rid, trajectory = _support_departure_case()
    monkeypatch.setattr(recovery_support_departure, "_LIMIT", 0)
    support = recovery_support_departure.prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    result = recovery_support_departure.verify_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene,
        joint_trajectory=trajectory, support_departure=support)
    assert result["status"] == "NEEDS_CONTEXT"
    assert "refinement" in result["reason"]


@pytest.mark.parametrize("motion_case", ["departure", "rotating", "deeper", "reentry", "zero_velocity"])
def test_support_departure_static_reuse_preserves_complete_uncached_proof(monkeypatch, motion_case):
    from cais_spade_llm.resources import recovery_support_departure as departure

    scene, rid, trajectory = _support_departure_case()
    if motion_case == "rotating":
        configuration = scene["resource_models"][rid]["configuration"]
        configuration["joint_names"].append("turn")
        configuration["joints"].append({"name": "turn", "parent": "body", "child": "tool", "type": "revolute",
            "xyz": [0., 0., 0.], "rpy": [0., 0., 0.], "axis": [0., 0., 1.]})
        configuration["reference_link"] = "tool"
        scene["snapshot"]["resources"][rid]["joint_names"].append("turn")
        scene["snapshot"]["resources"][rid]["joint_positions"].append(0.)
        trajectory["joint_names"].append("turn")
        trajectory["points"][0]["positions"].append(0.)
        trajectory["points"][1]["positions"].append(.5)
    elif motion_case == "deeper":
        trajectory["points"][-1]["positions"] = [-.1]
    elif motion_case == "reentry":
        trajectory["points"].append({"positions": [-.01], "time_from_start": {"sec": 2, "nanosec": 0}})
        trajectory["duration_ns"] = 2_000_000_000
    elif motion_case == "zero_velocity":
        for point in trajectory["points"]:
            point.update(velocities=[0.], accelerations=[0.])
    support = departure.prepare_support_departure(resource_id=rid, state=scene["snapshot"], evidence=scene)
    assert support["status"] == "prepared"
    inputs = {"resource_id": rid, "state": scene["snapshot"], "evidence": scene,
              "joint_trajectory": trajectory, "support_departure": support}
    original = deepcopy(inputs)
    result = departure.verify_support_departure(**inputs)
    derivative, kinematics = departure._transforms_with_derivative, departure.kinematic_transforms
    with monkeypatch.context() as local:
        local.setattr(departure, "_transforms_with_derivative", lambda configuration, positions, derivatives, **_:
            derivative(configuration, positions, derivatives))
        local.setattr(departure, "kinematic_transforms", lambda configuration, positions, origins=None:
            kinematics(configuration, positions))
        uncached = departure.verify_support_departure(**inputs)
    assert result == uncached
    assert result["status"] == ("NEEDS_CONTEXT" if motion_case in {"deeper", "reentry"} else "supported")
    assert inputs == original


def test_support_departure_static_matrices_are_per_configuration_and_read_only():
    from cais_spade_llm.resources import recovery_support_departure as departure

    scene, rid, _ = _support_departure_case()
    configuration = scene["resource_models"][rid]["configuration"]
    configuration["joints"].append({"name": "turn", "parent": "body", "child": "tool", "type": "revolute",
        "xyz": [.1, .2, .3], "rpy": [.1, .2, .3], "axis": [0., 0., 1.]})
    positions, derivatives = {"slide": (0., .2), "turn": (0., .5)}, {"slide": (.1, .2), "turn": (.1, .5)}
    results = []
    for shift in (0., .125):
        current = deepcopy(configuration)
        current["root_xyz"][0] += shift
        current["joints"][-1]["xyz"][1] += shift
        current["joints"][-1]["axis"] = [0., 0., 1.] if shift == 0 else [.6, 0., .8]
        constants = {"origins": {row["child"]: departure.transform(row["xyz"], row["rpy"])
                                for row in current["joints"]},
                     "root": (departure.quaternion_transform(current["root_xyz"], current["root_quaternion"])
                              if "root_quaternion" in current else departure.transform(current["root_xyz"], current["root_rpy"])),
                     "axis_skews": {row["child"]: departure._axis_skew(row["axis"])
                                    for row in current["joints"] if row["type"] in {"revolute", "continuous"}}}
        original = deepcopy((current, constants))
        actual = departure._transforms_with_derivative(current, positions, derivatives, **constants)
        assert actual == departure._transforms_with_derivative(current, positions, derivatives)
        assert (current, constants) == original
        results.append(actual)
    assert results[0] != results[1]


def _configuration(offset=0):
    return {"joint_names": ["slide"], "interpolation": "splines", "root": "world",
            "root_xyz": [offset, 0, 0], "root_rpy": [0, 0, 0], "reference_link": "body",
            "joints": [{"name": "slide", "parent": "world", "child": "body", "type": "prismatic",
                        "xyz": [0, 0, 0], "rpy": [0, 0, 0], "axis": [1, 0, 0]}],
            "components": [{"id": "body_box", "link": "body", "bounds": [[-0.05, 0.05]] * 3}]}


def test_pending_preview_binds_the_same_recorded_start_as_validation():
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import pending_outline_event

    scene, snapshot, model, proposed = _event_case()
    proposed.pop("expected_start_state", None)
    validated = {**deepcopy(proposed), "expected_start_state": {"resource_state": "picked"}}
    pending = {"kind": "event", "response": {"thought": "Captured proposal", "next_transition": proposed},
               "task": validated}
    before = deepcopy(pending)
    recovery = SimpleNamespace(_prepared={"multi_turn_session_state": {"pending_outline_validation": pending}})
    selected = pending_outline_event(recovery)
    assert selected == validated
    assert outline_event_binding(selected, snapshot, model, scene) == outline_event_binding(validated, snapshot, model, scene)
    assert outline_event_binding(selected, snapshot, model, scene) != outline_event_binding(proposed, snapshot, model, scene)
    selected["expected_start_state"]["resource_state"] = "changed"
    assert pending == before


@pytest.mark.parametrize("field", ["expected_end_state", "part_name", "event_name"])
def test_pending_preview_rejects_a_changed_proposal(field):
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import pending_outline_event

    _, _, _, proposed = _event_case()
    proposed["part_name"] = "KET4_Square_4mm"
    validated = {**deepcopy(proposed), field: "changed"}
    recovery = SimpleNamespace(_prepared={"multi_turn_session_state": {"pending_outline_validation": {
        "kind": "event", "response": {"next_transition": proposed}, "task": validated}}})
    with pytest.raises(ValueError, match="changes the retained LLM proposal"):
        pending_outline_event(recovery)


def _case():
    resources = {rid: {"joint_names": ["slide"], "joint_positions": [0],
                       "current_pose": [offset, 0, 0, 0, 0, 0, 1], "held_part": None,
                       "grasp_transform": None, "contained_parts": [], "resource_location": "home"}
                 for rid, offset in (("ur5e-3", 0), ("ur5e-4", 3))}
    parts = {name: {"current_pose": [-3, index, 0, 0, 0, 0, 1], "contained_by": None,
                    "processCompleted": [], "processCompleted_complete": True,
                    "processCompleted_evidence": {"complete": True, "source_kind": "recorded", "checkpoint": "capture"}}
             for index, name in enumerate(("gear_small", "KET4_Square_4mm"))}
    geometry = {"frame": "world", "resources": {rid: {} for rid in resources},
                "parts": {name: {"local_bounds": [[-0.05, 0.05]] * 3} for name in parts},
                "regions": {"assembly_board-v1": {"bounds": [[0.8, 1.2], [-0.1, 0.1], [-0.1, 0.1]]}}}
    geometry["parts"]["gear_small"]["target"] = "Gear_Plate/Gear_Shaft_1"
    models = {rid: {"configuration": _configuration(offset),
                    "named_joint_poses": {"home": {"joint_names": ["slide"], "positions": [0]}}}
              for rid, offset in (("ur5e-3", 0), ("ur5e-4", 3))}
    command = {"primitive": "move_to_pose", "params": {"pose_name": "home"}}
    support = {"semantics": "primitive_transition", "primitives": [command], "trace": [deepcopy(command)]}
    evidence = {**deepcopy(command), "validated": True, "source": "recorded_owner", "named_pose": "home"}
    return dict(resource_id="ur5e-3", snapshot={"frame": "world", "resources": resources, "parts": parts},
                geometry=geometry, resource_models=models, primitive_support=support, step_evidence=[evidence])


def _motion(case, target, *, start=0):
    evidence = case["step_evidence"][0]
    evidence.pop("named_pose", None)
    evidence["joint_trajectory"] = {
        "joint_names": ["slide"], "duration_ns": 1_000_000_000,
        "points": [{"positions": [value], "velocities": [], "accelerations": [], "effort": [],
                    "time_from_start": {"sec": index, "nanosec": 0}}
                   for index, value in enumerate((start, target))]}
    evidence["stationary_resources"] = ["ur5e-4"]
    evidence["stationary_parts"] = [name for name in case["snapshot"]["parts"]
                                     if name != case["snapshot"]["resources"]["ur5e-3"]["held_part"]]


def _validate(projection, *, all_rules=False, **kwargs):
    directory = Path(__file__).resolve().parents[1] / "cais_spade_llm/specification/safety"
    document = json.loads((directory / "safety_assembly_board-v1_mutex.txt").read_text())
    if all_rules:
        other = json.loads((directory / "safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt").read_text())
        document["catalog"]["specifications"].extend(other["catalog"]["specifications"])
        document["requirement_scopes"].extend(other["requirement_scopes"])
    return validate_outline_evidence_safety(projection=projection, catalog=document["catalog"],
                                            requirement_scopes=document["requirement_scopes"], **kwargs)


def _event_case():
    case = _case()
    scene = {key: deepcopy(case[key]) for key in ("snapshot", "geometry", "resource_models")}
    scene["jids"] = {rid: rid + "@localhost" for rid in scene["snapshot"]["resources"]}
    scene["resources"] = {rid: {"idle": True, "stationary_contract": {"kind": "recorded hold"}}
                          for rid in scene["jids"]}
    scene["stationary_parts"] = list(scene["snapshot"]["parts"])
    snapshot = {"resource_location": "home", "resource_state": "failed", "held_part": None,
                "current_pose": [0, 0, 0, 0, 0, 0, 1]}
    model = {"state_variables": {"resource_location": {"scope": "resource", "domain": ["home"]},
        "resource_state": {"scope": "resource", "domain": ["failed", "idle"]}},
        "events": [{"event_name": "move_to_pose", "updates": {
            "resource_location": {"set": "home"}, "resource_state": {"set": "idle"}},
            "primitive_support": [{"primitive": "move_to_pose", "params": {"pose_name": "home"}}]}]}
    task = {"resource_jid": "ur5e-3@localhost", "outline_id": "RECOVERY1", "event_name": "new_event",
            "expected_end_state": {"resource_location": "home", "resource_state": "idle"}, "params": {}}
    return scene, snapshot, model, task


def test_new_event_clock_preserves_null_part_and_binds_full_owner_ledger():
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )
    from cais_spade_llm.agents.central_controller.outline_evidence_safety import bind_validated_outline_projection

    scene, snapshot, model, task = _event_case()
    task["part_name"] = None
    scene.update(physical_preview_contract_version=2, owner_projection_clock_version=1)
    scene["environment_models"] = {rid: {"state_variables": {
        "resource_state": {"type": "string"}, "resource_location": {"type": "string"}}}
        for rid in scene["jids"]}
    scene["environment_valuation"] = {rid: {"resource_state": "failed" if rid == "ur5e-3" else "idle",
                                                   "resource_location": "home"} for rid in scene["jids"]}
    original = deepcopy((scene, snapshot, model, task))
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence={})
    result = finalize_outline_event_preview(prepared_preview=prepared, motion_results=[_issued_native_preview(prepared)])
    assert result["status"] == "prepared", result
    projection = result["geometry_evidence"]
    assert projection["observations"][-1]["time_exact"] == "1"
    assert projection["successor_evidence"]["environment_valuation"]["ur5e-3"]["resource_state"] == "idle"
    bound = bind_validated_outline_projection(projection, task=task, physical_evidence=scene)
    assert bound["owner_safety_evidence"]["task_evidence"]["horizon"] == [0., 1.]
    assert bound["owner_safety_evidence"]["state_evidence"]["updates"][0]["values"] == {"resource_state": "idle"}
    from cais_spade_llm.recovery_framework.recovery_outline_audit import recorded_initial_checkpoint

    expected = recorded_initial_checkpoint(scene, projection)
    actual = {key: value for key, value in projection["observations"][0].items()
              if key not in {"state_snapshot", "owner_safety_evidence"}}
    assert expected == actual
    for defect in ("contract", "clock_type", "declaration", "conflict", "private_pose"):
        changed_scene, changed_projection = deepcopy(scene), deepcopy(projection)
        if defect == "contract":
            changed_projection["owner_interval_contract"]["source_contract_fingerprint"] = "stale"
        elif defect == "clock_type":
            changed_projection["projection_clock_version"] = True
        elif defect == "declaration":
            changed_scene["environment_models"]["ur5e-3"]["state_variables"].pop("resource_state")
        elif defect == "conflict":
            changed_scene["environment_valuation"]["ur5e-3"]["resource_location"] = False
        else:
            changed_scene["environment_models"]["ur5e-3"]["state_variables"]["base_pose"] = {"type": "array"}
            changed_scene["environment_valuation"]["ur5e-3"]["base_pose"] = [0, 0, 0]
        with pytest.raises(ValueError):
            recorded_initial_checkpoint(changed_scene, changed_projection)
    assert (scene, snapshot, model, task) == original


def test_event_clock_offsets_exact_native_times_without_rounding():
    from fractions import Fraction
    from cais_spade_llm.resources.recovery_geometry import _event_clock_motion

    point = {"continuous_interval": ["0", "3/10"],
             "continuous_cells": [{"interval": ["0", "1/10"]}, {"interval": ["1/10", "3/10"]}]}
    assert _event_clock_motion(point, Fraction(1, 5)) == Fraction(1, 2)
    assert point["time_exact"] == "1/5"
    assert point["continuous_interval"] == ["1/5", "1/2"]
    assert [row["interval"] for row in point["continuous_cells"]] == [["1/5", "3/10"], ["3/10", "1/2"]]


@pytest.mark.parametrize("alternative", ["absent", "conflicting", "compatible", "unbound", "unsupported", "missing"])
def test_owner_effect_rejection_requires_every_successor_alternative_to_conflict(alternative):
    from cais_spade_llm.resources.recovery_geometry import rejected_owner_event_effects

    scene, snapshot, model, task = _event_case()
    model["state_variables"]["part_state"] = {"scope": "part", "domain": ["in_gripper", "in_transit", None]}
    model["events"][0]["updates"]["part_state"] = {"set": "in_transit"}
    model["events"][0]["composition_contract"] = {"part_transition": {"completed": {
        "state": "in_transit", "location_template": "{resource_jid}"}}}
    task["expected_end_state"]["part_state"] = "in_gripper"
    task["event_name"] = "independently_proposed_event"
    if alternative != "absent":
        second = deepcopy(model["events"][0])
        second["event_name"] = "another_declared_transition"
        effect = {"conflicting": {"set": None}, "compatible": {"set": "in_gripper"},
                  "unbound": {"set_from_param": "unrecorded_parameter"},
                  "unsupported": {"unsupported": "in_transit"}, "missing": None}[alternative]
        if effect is None:
            second["updates"].pop("part_state")
        else:
            second["updates"]["part_state"] = effect
            if "set" in effect:
                second["composition_contract"]["part_transition"]["completed"]["state"] = effect["set"]
        model["events"].append(second)
    original = deepcopy((scene, snapshot, model, task))

    result = rejected_owner_event_effects(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene)

    if alternative in {"absent", "conflicting"}:
        assert result["status"] == result["validation_status"] == "rejected"
        assert result["owner_effect_result"]["feasibility_status"] == "INFEASIBLE"
        assert result["owner_effect_result"]["alternatives_complete"] is True
        assert len(result["owner_effect_result"]["alternatives"]) == len(model["events"])
        assert result["owner_effect_result"]["conflicts"][0] == {
            "field": "part_state", "requested": "in_gripper", "derived": "in_transit"}
        assert result["event_record_binding"] == outline_event_binding(task, snapshot, model, scene)
        assert result["task"] == task
        assert result["observations"] == [] and result["interval_complete"] is False
        assert result["command_sent"] is False and result["execution_authorized"] is False
        assert not set(result) & {"projected_snapshot", "successor_evidence", "monitor_context_after"}
    else:
        assert result is None
    assert (scene, snapshot, model, task) == original


@pytest.mark.parametrize("missing", ["model_complete", "declaration", "primitive_support", "identity"])
def test_owner_effect_rejection_does_not_turn_missing_declarations_into_contradictions(missing):
    from cais_spade_llm.resources.recovery_geometry import rejected_owner_event_effects

    scene, snapshot, model, task = _event_case()
    model["state_variables"]["part_state"] = {"scope": "part", "domain": [None, "in_transit"]}
    model["events"][0]["updates"]["part_state"] = {"set": "in_transit"}
    model["events"][0]["composition_contract"] = {"part_transition": {"completed": {
        "state": "in_transit", "location_template": "{resource_jid}"}}}
    task["expected_end_state"]["part_state"] = None
    if missing == "model_complete":
        model["model_complete"] = False
    elif missing == "declaration":
        model["state_variables"].pop("part_state")
    elif missing == "primitive_support":
        model["events"][0].pop("primitive_support")
    else:
        scene["jids"] = {}
    assert rejected_owner_event_effects(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene) is None


def test_owner_effect_rejection_distinguishes_missing_null_and_explicit_parameter_values():
    from cais_spade_llm.resources.recovery_geometry import rejected_owner_event_effects

    scene, snapshot, model, task = _event_case()
    model["state_variables"]["part_state"] = {"scope": "part", "domain": [None, "in_transit"]}
    model["events"][0]["updates"]["part_state"] = {"set_from_param": "part_state_value"}
    model["events"][0]["composition_contract"] = {"part_transition": {"completed": {
        "state": None, "location_template": "{resource_jid}"}}}
    task["params"]["part_state_value"] = None
    assert rejected_owner_event_effects(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene) is None
    task["expected_end_state"]["part_state"] = None
    assert rejected_owner_event_effects(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene) is None
    task["expected_end_state"]["part_state"] = "in_transit"
    rejected = rejected_owner_event_effects(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene)
    assert rejected["owner_effect_result"]["conflicts"] == [
        {"field": "part_state", "requested": "in_transit", "derived": None}]
    task["params"].pop("part_state_value")
    assert rejected_owner_event_effects(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene) is None


def test_owner_effect_contradiction_is_recomputed_without_native_or_trusting_saved_verdict(monkeypatch):
    from cais_spade_llm.resources.recovery_geometry import rejected_owner_event_effects
    from cais_spade_llm.resources import resource_safety_preparation

    scene, snapshot, model, task = _event_case()
    model["state_variables"]["part_state"] = {"scope": "part", "domain": ["in_gripper", "in_transit"]}
    model["events"][0]["updates"]["part_state"] = {"set": "in_transit"}
    model["events"][0]["composition_contract"] = {"part_transition": {"completed": {
        "state": "in_transit", "location_template": "{resource_jid}"}}}
    task["expected_end_state"]["part_state"] = "in_gripper"
    expected = rejected_owner_event_effects(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene)
    assert project_recovery_outline_event(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, evidence=scene) == expected
    assert resource_safety_preparation.prepare_outline_event_preview_request(task=task,
        recovery_snapshot=snapshot, recovery_des_model=model, physical_evidence=scene,
        resource_evidence={}) == expected
    scene["event_evidence"] = [deepcopy(expected)]
    model["events"][0]["updates"]["part_state"] = {"set": "in_gripper"}
    unresolved = project_recovery_outline_event(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, evidence=scene)
    assert unresolved["status"] == "needs_context"
    assert not unresolved.get("owner_effect_rejected")


@pytest.mark.parametrize("completed", [None, {"state": "assembled", "verify_camera": True,
                                           "location_param": "destination_location"}])
def test_owner_effect_rejection_keeps_receiving_product_effects_deferred(completed):
    from cais_spade_llm.resources.recovery_geometry import rejected_owner_event_effects

    scene, snapshot, model, task = _event_case()
    model["state_variables"]["part_state"] = {"scope": "part", "domain": ["ready", "assembled"]}
    model["events"][0]["updates"]["part_state"] = {"set": "assembled"}
    model["events"][0]["composition_contract"] = {"part_transition": {"completed": completed}}
    task["expected_end_state"]["part_state"] = "ready"
    assert rejected_owner_event_effects(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene) is None


@pytest.mark.parametrize("value", [{"$arg": "unresolved"}, ["in_transit"], float("nan"), "outside_closed_domain"])
def test_owner_effect_rejection_does_not_treat_unsupported_values_as_literal_contradictions(value):
    from cais_spade_llm.resources.recovery_geometry import rejected_owner_event_effects

    scene, snapshot, model, task = _event_case()
    model["state_variables"]["part_state"] = {"scope": "part", "domain": ["in_gripper", "in_transit"],
                                               "domain_complete": True}
    model["events"][0]["updates"]["part_state"] = {"set": value}
    model["events"][0]["composition_contract"] = {"part_transition": {"completed": {
        "state": value, "location_template": "{resource_jid}"}}}
    task["expected_end_state"]["part_state"] = "in_gripper"
    assert rejected_owner_event_effects(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene) is None


@pytest.mark.parametrize("missing", ["jids", "snapshot", "resources", "composition_contract", "part_transition"])
def test_owner_effect_rejection_keeps_null_source_metadata_unresolved(missing):
    from cais_spade_llm.resources.recovery_geometry import rejected_owner_event_effects

    scene, snapshot, model, task = _event_case()
    model["state_variables"]["part_state"] = {"scope": "part", "domain": ["in_gripper", "in_transit"]}
    model["events"][0]["updates"]["part_state"] = {"set": "in_transit"}
    model["events"][0]["composition_contract"] = {"part_transition": {"completed": {
        "state": "in_transit", "location_template": "{resource_jid}"}}}
    task["expected_end_state"]["part_state"] = "in_gripper"
    if missing in {"jids", "snapshot"}:
        scene[missing] = None
    elif missing == "resources":
        scene["snapshot"]["resources"] = None
    elif missing == "composition_contract":
        model["events"][0][missing] = None
    else:
        model["events"][0]["composition_contract"][missing] = None

    assert rejected_owner_event_effects(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene) is None


@pytest.mark.parametrize("effect", [
    {"set": {"$arg": "unavailable"}}, {"set": ["idle"]}, {"set": float("nan")},
    {"set": "idle", "set_from_param": "state"}, {"set_from_param": ""},
    {"unsupported": "idle"}, None,
])
def test_owner_effect_rejection_does_not_discard_unsupported_resource_alternatives(effect):
    from cais_spade_llm.resources.recovery_geometry import rejected_owner_event_effects

    scene, snapshot, model, task = _event_case()
    model["state_variables"]["part_state"] = {"scope": "part", "domain": ["in_gripper", "in_transit"]}
    model["events"][0]["updates"]["part_state"] = {"set": "in_transit"}
    model["events"][0]["composition_contract"] = {"part_transition": {"completed": {
        "state": "in_transit", "location_template": "{resource_jid}"}}}
    task["expected_end_state"]["part_state"] = "in_gripper"
    second = deepcopy(model["events"][0])
    second["event_name"] = "unresolved_resource_alternative"
    second["updates"]["resource_state"] = effect
    second["updates"]["part_state"] = {"set": "in_gripper"}
    second["composition_contract"]["part_transition"]["completed"]["state"] = "in_gripper"
    model["events"].append(second)

    assert rejected_owner_event_effects(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene) is None
    second["updates"]["resource_state"] = {"set": "failed"}
    rejected = rejected_owner_event_effects(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene)
    assert rejected["owner_effect_rejected"] is True
    assert len(rejected["owner_effect_result"]["alternatives"]) == 1


def _commanded_hold_case():
    scene, _, _, _ = _event_case()
    rid = "ur5e-4"
    state, model = scene["snapshot"]["resources"][rid], scene["resource_models"][rid]
    state["joint_positions"] = [.005]
    physical = {"idle": False, "observation_state": deepcopy(state),
        "stationary_contract": {"kind": "idle_commanded_hold", "requires_no_running_tasks": True,
            "requires_no_active_goals": True, "future_execution_tracking": "not_established"},
        "controller_hold_observation": {"holding": True, "has_active_goal": False, "has_pending_goal": False,
            "instance_id": "controller-1", "joint_names": ["slide"], "commanded_positions": [0.],
            "commanded_positions_source": "native position command interfaces", "command_revision": 4},
        "controller_hold_model": {"value": .01, "node": "/move_group",
            "parameter": "trajectory_execution.allowed_start_tolerance", "kind": "conditional model assumption",
            "physical_execution_verified": False, "future_execution_tracking": "not_established"},
        "controller_hold_activity": {"controller": {"holding": True, "has_active_goal": False,
                                                     "has_pending_goal": False}}}
    return scene, rid, state, model, physical


@pytest.mark.parametrize("change", [None, "active", "pending", "target_missing", "outside_bound",
    "empirical_error", "other_controller_active", "passive_missing", "stale_projection"])
def test_commanded_hold_is_conditional_and_preserves_observed_motion(change):
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence

    _, rid, state, model, physical = _commanded_hold_case()
    if change in {"active", "pending"}:
        physical["controller_hold_observation"]["has_" + change + "_goal"] = True
    elif change == "target_missing":
        physical["controller_hold_observation"]["commanded_positions"] = None
    elif change == "outside_bound":
        state["joint_positions"] = [.02]
        physical["observation_state"] = deepcopy(state)
    elif change == "empirical_error":
        physical["controller_hold_model"].pop("value")
        physical["controller_hold_observation"]["maximum_observed_position_error"] = [100.]
    elif change == "other_controller_active":
        physical["controller_hold_activity"]["controller"]["has_active_goal"] = True
    elif change == "passive_missing":
        model["configuration"]["joints"].append({"name": "tool", "type": "prismatic", "frozen_position": 0.})
    elif change == "stale_projection":
        physical["projected_controller_hold"] = {"kind": "projected", "joint_names": ["slide"],
            "positions": [1.], "binding": {"conditions_fingerprint": "prior"}}
    original = deepcopy(physical)
    result = commanded_hold_evidence(resource_id=rid, state=state, model=model, physical=physical)
    assert result["status"] == ("supported" if change is None else "NEEDS_CONTEXT")
    assert result["physical_execution_verified"] is False
    assert physical == original
    if change is None:
        assert result["observed_idle"] is False
        assert result["joint_intervals"]["slide"][0] <= -.01
        assert result["joint_intervals"]["slide"][1] >= .01


@pytest.mark.parametrize("change", [None, "missing", "open", "active", "publisher", "mismatch"])
def test_commanded_hold_checks_owned_gate_without_claiming_nav2_idle(change):
    from test_recovery_case_safety import _base_hold_separate_descriptions
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence

    _, rid, state, model, physical = _commanded_hold_case()
    _, base, _, _ = _base_hold_separate_descriptions()
    controller = base['resources']['KMR']['base_hold_evidence']['controller_observation']
    controller.update(has_active_goal=True, holding=False)
    if change == 'missing':
        controller.pop('command_gate_evidence')
    elif change == 'open':
        controller['command_gate']['closed'] = False
    elif change == 'active':
        controller['owned_activity']['has_active_goal'] = True
    elif change == 'publisher':
        controller['command_gate_evidence']['output_publisher_node'] = '/another'
    elif change == 'mismatch':
        controller['controller_source_sha256'] = 'd'*64
    physical['controller_hold_activity']['/KMR/recovery_state'] = controller
    before = deepcopy(physical)
    result = commanded_hold_evidence(resource_id=rid, state=state, model=model, physical=physical)
    assert result['status'] == ('supported' if change is None else 'NEEDS_CONTEXT')
    assert physical == before


@pytest.mark.parametrize('change', [None, 'missing', 'pose', 'boolean_pose', 'malformed_xml'])
def test_commanded_hold_mobile_base_requires_independent_physical_hold(change):
    from test_recovery_case_safety import _base_hold_separate_descriptions
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence

    _, _, state, model, physical = _commanded_hold_case()
    _, base, _, _ = _base_hold_separate_descriptions()
    physical.update(base['resources']['KMR'])
    config = base['resource_models']['KMR']['configuration']
    for key in ('root', 'root_xyz', 'root_quaternion', 'description_sha256'):
        model['configuration'][key] = deepcopy(config[key])
    state['base_pose'] = deepcopy(physical['base_hold_evidence']['root_pose'])
    if change == 'missing':
        physical.pop('base_hold_evidence')
    elif change == 'pose':
        state['base_pose'][0] += 1.
    elif change == 'boolean_pose':
        state['base_pose'][0] = False
    elif change == 'malformed_xml':
        import hashlib

        evidence = physical['base_hold_evidence']
        evidence['physics_robot_description'] = '<robot'
        evidence['physics_description_sha256'] = hashlib.sha256(b'<robot').hexdigest()
    result = commanded_hold_evidence(resource_id='KMR', state=state, model=model, physical=physical)
    assert result['status'] == ('supported' if change is None else 'NEEDS_CONTEXT')
    if change is None:
        assert result['base_hold']['base_velocity_bounds'] == {axis: [0., 0.] for axis in ('x','y','yaw')}


def test_peer_hold_encloses_tool_and_carried_part_and_retains_projected_command():
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence, _checkpoint, _prepared_motion

    scene, rid, state, model, physical = _commanded_hold_case()
    state["held_part"] = "gear_small"
    state["grasp_transform"] = [0, 0, 0, 0, 0, 0, 1]
    physical["observation_state"] = deepcopy(state)
    scene["geometry"]["regions"]["assembly_board-v1"]["bounds"][0] = [3.057, 3.059]
    hold = commanded_hold_evidence(resource_id=rid, state=state, model=model, physical=physical)
    case = _case()
    _motion(case, .1)
    case["step_evidence"][0]["stationary_parts"] = ["KET4_Square_4mm"]
    motion = _prepared_motion(scene["snapshot"], "ur5e-3", scene["resource_models"], case["step_evidence"][0])
    row = _checkpoint(scene["snapshot"], scene["geometry"], scene["resource_models"], index=1,
        moving=("ur5e-3", motion, 0, motion.duration), peer_holds={rid: hold})
    assert row["resources"][rid]["joint_positions"] == [.005]
    assert row["parts"]["gear_small"]["current_pose"] == scene["snapshot"]["parts"]["gear_small"]["current_pose"]
    assert row["occupancy_possibilities"]["assembly_board-v1"][rid] == [False, True]
    assert row["part_occupancy_possibilities"]["assembly_board-v1"]["gear_small"] == [False, True]
    physical["projected_controller_hold"] = {"kind": "projected", "joint_names": ["slide"],
        "positions": [1.], "binding": {"conditions_fingerprint": "accepted owner preview"}}
    state["joint_positions"] = [1.]
    projected = commanded_hold_evidence(resource_id=rid, state=state, model=model, physical=physical)
    assert projected["status"] == "supported"
    assert projected["joint_intervals"]["slide"][0] <= .99
    assert projected["joint_intervals"]["slide"][1] >= 1.01


def test_hold_passive_joint_bounds_are_declared_and_zero_model_error_stays_zero():
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence, _joint_geometry

    _, rid, state, model, physical = _commanded_hold_case()
    model["configuration"]["joints"].append({"name": "passive", "type": "prismatic", "parent": "body",
        "child": "tool", "xyz": [0, 0, 0], "rpy": [0, 0, 0], "axis": [1, 0, 0],
        "frozen_position": .05, "position_bounds": [0., .2]})
    model["configuration"]["components"][0]["link"] = "tool"
    supported = commanded_hold_evidence(resource_id=rid, state=state, model=model, physical=physical)
    assert supported["status"] == "supported"
    assert supported["joint_intervals"]["passive"] == [0., .2]
    boxes, _ = _joint_geometry(state, model, joint_intervals=supported["joint_intervals"])
    assert boxes[0]["maximum"][0][1] >= 3.26
    physical["controller_hold_model"]["value"] = 0.
    physical["controller_hold_observation"]["maximum_observed_position_error"] = [1.]
    rejected = commanded_hold_evidence(resource_id=rid, state=state, model=model, physical=physical)
    assert rejected["status"] == "NEEDS_CONTEXT"
    assert "outside" in rejected["reason"]


@pytest.mark.parametrize(("case", "supported"), [
    ("static", True), ("contained", True), ("loose", False), ("moving", False),
    ("held", False), ("conflicting_static", False), ("missing_speed", False),
])
def test_stationary_part_capture_requires_physics_or_declared_containment(case, supported):
    from cais_spade_llm.recovery_framework.recovery_observations import _stationary_part_observation

    name, rid = "gear_small", "Buffer For Machined parts"
    contract = {"kind": "static_body_and_idle_containment", "requires_no_running_tasks": True,
                "requires_no_active_goals": True, "future_execution_tracking": "not_established"}
    configuration = {"resources": {rid: {"stationary_contract": contract,
        "stationary_linear_speed_max": .001, "stationary_angular_speed_max": .001}}}
    part = {"name": "gear_model", "pose": [0, 0, 0, 0, 0, 0, 1]}
    record = {"instance_id": "physics-owner", "simulation_time": 1.,
        "attachment_complete": True, "attachments": [], "models": {"gear_model": {
        "static": case in {"static", "conflicting_static"}, "links": {"link": {
            "linear_speed": .01 if case in {"moving", "conflicting_static"} else 0., "angular_speed": 0.}}}}}
    if case == "missing_speed":
        record["models"]["gear_model"]["links"]["link"].pop("angular_speed")
    evidence = {"snapshot": {"resources": {rid: {"held_part": name if case == "held" else None}},
        "parts": {name: {"contained_by": None if case == "loose" else rid}}},
        "resources": {rid: {"model_static": True, "idle": True, "stationary_contract": contract,
            "observation_state": {"contained_parts": [name]}}}}
    before = deepcopy((part, record, evidence, configuration))
    result = _stationary_part_observation(name, part, record, evidence, configuration)
    assert (result["status"] == "supported") is supported
    assert result["source"] == {"service": "GETRECOVERYSTATE", "instance_id": "physics-owner", "simulation_time": 1.}
    assert result["future_execution_tracking"] == "not_established"
    assert (part, record, evidence, configuration) == before


def test_held_part_capture_uses_owning_attachment_snapshot(monkeypatch):
    from cais_spade_llm.recovery_framework import gazebo_safety_preparation, part_collision
    from cais_spade_llm.recovery_framework.recovery_observations import capture_recovery_physical_evidence
    import time

    now = time.monotonic()
    name, rid, model_name = "KET4_Square_4mm", "ur5e-3", "KET4_Square_4mm"
    pose = [1., 2., 3., 0., 0., 0., 1.]
    bounds = [[-.002, .002], [-.002, .002], [-.025, .025]]
    physical = {"frame": "world", "observed_monotonic": now, "simulation_time": 7.,
        "launch_id": "launch", "observation_state": {"held_part": name},
        "attachment": {"part_name": name, "model_name": model_name, "part_pose": pose,
            "part_local_bounds": bounds, "observed_attachment_complete": True, "instance_id": "physics"}}
    later = {"instance_id": "physics", "simulation_time": 8., "observed_monotonic": now, "models": {model_name: {
        "static": False, "pose": [9., 9., 9., 0., 0., 0., 1.], "links": {"link": {"collisions": []}}}}}
    for free in ("gear_small", "gear_large"):
        later["models"][free] = {"static": True, "pose": [0., 0., 0., 0., 0., 0., 1.], "links": {
            "link": {"linear_speed": 0., "angular_speed": 0., "collisions": [{"bounds": [[-.1, .1]] * 3}]}}}
    reads = []

    def snapshot(**kwargs):
        reads.append(kwargs)
        return deepcopy(later)

    observer = SimpleNamespace(provider=SimpleNamespace(reader=SimpleNamespace(snapshot=snapshot)))
    owner = SimpleNamespace(agent_name=rid, recovery_safety_observer=observer,
        capture_recovery_safety_state=lambda **_: deepcopy(physical))
    context = SimpleNamespace(revision=1, run_id="run", resources={rid: SimpleNamespace(valuation={"held_part": name})},
        part_tracker={name: {"location": rid, "state": "in_gripper"}}, inputs={
            "scene": {}, "geometry": {"parts": {"model_map": {name: model_name}}}})
    for free in ("gear_small", "gear_large"):
        context.part_tracker[free] = {"location": None, "state": "ready"}
        context.inputs["geometry"]["parts"]["model_map"][free] = free
    runtime = SimpleNamespace(context=context, resource_agents=[owner], jids={rid: rid + "@localhost"})
    monkeypatch.setattr(gazebo_safety_preparation, "_entity_reader", lambda _: object())
    monkeypatch.setattr(gazebo_safety_preparation, "_entity", lambda *args, **kwargs: pytest.fail(
        "Held-part capture must not read a separate physics pose"))
    monkeypatch.setattr(part_collision, "observed_collision_identities", lambda *args: {})

    result = capture_recovery_physical_evidence(runtime)

    assert result["unresolved"] == []
    assert result["snapshot"]["parts"][name]["current_pose"] == pose
    assert result["parts"][name]["simulation_time"] == 7.
    assert result["parts"][name]["source"] == "owning resource attachment observation"
    assert result["geometry"]["parts"][name]["local_bounds"] == bounds
    assert result["geometry"]["parts"][name]["source"]["simulation_time"] == 7.
    assert result["physics_evidence"]["models"][model_name]["pose"] == later["models"][model_name]["pose"]
    assert name not in result["stationary_parts"]
    assert len(reads) == 1
    for free in ("gear_small", "gear_large"):
        assert result["parts"][free]["simulation_time"] == 8.
        assert result["parts"][free]["pose"] == later["models"][free]["pose"]


def test_event_preview_does_not_reuse_stationary_evidence_after_part_pose_changes():
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )

    scene, snapshot, model, task = _event_case()
    scene["stationary_part_evidence"] = {name: {"status": "supported", "current_pose": deepcopy(row["current_pose"])}
        for name, row in scene["snapshot"]["parts"].items()}
    scene["stationary_part_evidence"]["gear_small"]["current_pose"][0] += .1
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence={})
    result = finalize_outline_event_preview(prepared_preview=prepared,
        motion_results=[_preview_response(prepared["motion_requests"][0])])
    assert result["status"] == "NEEDS_CONTEXT"
    assert "gear_small" in result["reason"]


def _preview_response(request, start=0, target=0):
    case = _case()
    _motion(case, target, start=start)
    return {"status": "PREVIEW_COMPLETE", "path_complete": True, "fraction": 1.0, "command_sent": False,
            "cartesian_position_precision": 1e-13, "cartesian_orientation_precision": 1e-13,
            "cartesian_position_residual": 0., "cartesian_orientation_residual": 0.,
            "request": deepcopy(request), "joint_trajectory": case["step_evidence"][0]["joint_trajectory"]}


def _issued_native_preview(prepared):
    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record

    state = {"is_diff": False, "joint_state": {"name": ["slide"], "position": [0.]}}
    request = {"planning_scene": {"is_diff": False, "robot_state": state}, "start_state": deepcopy(state),
        "resource_id": prepared["resource_id"], "capture_fingerprint": "fixture-capture",
        "proposal_fingerprint": recovery_validation_fingerprint(prepared["task"]), "scene_fingerprint": "fixture-scene",
        "model_fingerprint": "fixture-model", "group_name": "owner", "link_name": "body", "frame_id": "world",
        "max_translation_step": .01, "max_rotation_step": .01, "max_joint_step": .01,
        "velocity_scaling": .1, "acceleration_scaling": .1, "timeout_seconds": 10.,
        **deepcopy(prepared["motion_requests"][0])}
    raw = _preview_response(request)
    trajectory = raw.pop("joint_trajectory")
    trajectory.pop("duration_ns", None)
    raw.update(trajectory={"joint_trajectory": trajectory}, continuous_collision_verified=False,
        **{field: request[field] for field in ("resource_id", "capture_fingerprint", "proposal_fingerprint",
                                               "scene_fingerprint", "model_fingerprint")})
    prepared["native_requests"] = [request]
    return preview_result_record(request, raw)


def _incomplete_cartesian_preview(monkeypatch, failure_kind="ik_search_failed"):
    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record
    from cais_spade_llm.resources import recovery_parameter_evidence, resource_safety_preparation
    from cais_spade_llm.resources.robot import recovery_capability

    scene, snapshot, model, task = _custody_event_case(False)
    scene["physical_preview_contract_version"] = 3
    target = {"x": 0., "y": 0., "z": 0., "qx": 0., "qy": 0., "qz": 0., "qw": 1.}
    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **_: {
        "status": "calculated", "insert_pose": target})
    evidence = {"state": {"_task_ctx": {"part_name": task["part_name"], "travel_z": .2}}}
    overlay = {**deepcopy(evidence), "source": {"kind": "projected", "contract_fingerprint": "fixture-contract"}}
    monkeypatch.setattr(recovery_parameter_evidence, "projected_resource_evidence", lambda **_: deepcopy(overlay))
    prepared = resource_safety_preparation.prepare_outline_event_preview_request(task=task,
        recovery_snapshot=snapshot, recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence)
    assert prepared["status"] == "ready", prepared
    assert prepared["resource_evidence_fingerprint"] == recovery_validation_fingerprint(evidence)
    assert prepared["resource_evidence_fingerprint"] != recovery_validation_fingerprint(overlay)
    response = _issued_native_preview(prepared)
    request = response["request"]
    response.update(status="NEEDS_CONTEXT", path_complete=False, fraction=.4, trajectory={},
                    reason="The recorded Cartesian search did not complete")
    diagnostic = {"version": 1, "source": "PreviewRecoveryMotion", "diagnostic_only": True,
        "complete_inputs": True, "failure_kind": failure_kind, "fraction": .4,
        "request_binding": {key: request[key] for key in ("resource_id", "capture_fingerprint",
            "proposal_fingerprint", "scene_fingerprint", "model_fingerprint")},
        **{key: request[key] for key in ("group_name", "link_name", "frame_id", "max_translation_step",
                                       "max_rotation_step", "max_joint_step")},
        "ik_queries": [{"index": 0, "success": failure_kind == "joint_jump", "error_code": -6,
                        "timeout_seconds": .05, "seed": [0.], "consistency_limits": [.01]}],
        "waypoints": [{"index": 0, "query_start": 0, "query_end": 1, "fraction": .4,
                       "jump_check_truncated": failure_kind == "joint_jump",
                       "last_accepted_state": deepcopy(request["start_state"])}]}
    response["cartesian_diagnostics_json"] = json.dumps(diagnostic)
    return prepared, preview_result_record(request, response), evidence


@pytest.mark.parametrize("failure_kind", ["ik_search_failed", "joint_jump"])
def test_incomplete_cartesian_search_records_bound_unresolved_evidence_without_projection(monkeypatch, failure_kind):
    from cais_spade_llm.resources.resource_safety_preparation import finalize_outline_event_preview

    prepared, response, evidence = _incomplete_cartesian_preview(monkeypatch, failure_kind)
    original = deepcopy((prepared, response, evidence))
    result = finalize_outline_event_preview(prepared_preview=prepared, motion_results=[response])
    assert result["status"] == result["validation_status"] == "NEEDS_CONTEXT", result
    assert result["physical_motion_unresolved"] is True
    assert result["incomplete_motion"] == response
    assert result["event_record_binding"] == prepared["event_record"]["binding"]
    assert result["resource_evidence_fingerprint"] == recovery_validation_fingerprint(evidence)
    assert result["task"] == prepared["task"]
    assert result["reason"] == response["reason"]
    assert result["command_sent"] is False and result["execution_authorized"] is False
    assert not set(result) & {"event_evidence", "geometry_evidence", "physical_motion_rejected", "resource_models"}
    assert (prepared, response, evidence) == original


@pytest.mark.parametrize("defect", ["timeout", "solver_unavailable", "incomplete_inputs", "missing_raw",
    "missing_registry", "stale_result", "stale_diagnostic", "request_mismatch", "forged_kind",
    "forged_complete_inputs", "missing_owner_evidence", "path_complete", "fraction", "raw_trajectory"])
def test_incomplete_cartesian_search_rejects_unbound_or_unavailable_evidence(monkeypatch, defect):
    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record
    from cais_spade_llm.resources.resource_safety_preparation import finalize_outline_event_preview

    prepared, response, _ = _incomplete_cartesian_preview(monkeypatch)
    diagnostic = json.loads(response["cartesian_diagnostics_json"])
    if defect in {"timeout", "solver_unavailable", "incomplete_inputs", "forged_complete_inputs"}:
        if defect == "timeout":
            diagnostic["failure_kind"] = "canceled_or_deadline"
        elif defect == "solver_unavailable":
            diagnostic["failure_kind"] = "solver_unavailable"
        else:
            diagnostic["complete_inputs"] = False
        response["cartesian_diagnostics_json"] = json.dumps(diagnostic)
        response = preview_result_record(response["request"], response)
        if defect == "forged_complete_inputs":
            response["cartesian_failure_evidence"]["complete_inputs"] = True
    elif defect == "missing_raw":
        response.pop("cartesian_diagnostics_json")
    elif defect == "missing_registry":
        prepared.pop("native_requests")
    elif defect == "stale_result":
        response["scene_fingerprint"] = "different-scene"
    elif defect == "stale_diagnostic":
        diagnostic["request_binding"]["proposal_fingerprint"] = "different-proposal"
        response["cartesian_diagnostics_json"] = json.dumps(diagnostic)
    elif defect == "request_mismatch":
        response["request"]["max_joint_step"] = .02
    elif defect == "forged_kind":
        response["cartesian_failure_evidence"]["failure_kind"] = "joint_jump"
    elif defect == "missing_owner_evidence":
        prepared.pop("resource_evidence_fingerprint")
    elif defect == "path_complete":
        response["path_complete"] = True
    elif defect == "fraction":
        response["fraction"] = 1.
    elif defect == "raw_trajectory":
        response["joint_trajectory"] = {"joint_names": ["slide"], "points": [{"positions": [.1]}]}
    original = deepcopy((prepared, response))
    result = finalize_outline_event_preview(prepared_preview=prepared, motion_results=[response])
    assert result["status"] == "NEEDS_CONTEXT", result
    assert result.get("physical_motion_unresolved") is not True
    assert result["command_sent"] is False and result["execution_authorized"] is False
    assert not set(result) & {"incomplete_motion", "event_evidence", "geometry_evidence", "physical_motion_rejected"}
    assert (prepared, response) == original


@pytest.mark.parametrize("defect", [None, "missing_registry", "request_scene", "result_scene", "raw_trajectory",
                                   "request_target", "proposal", "execution", "source"])
@pytest.mark.parametrize("version", [2, 3, 4])
def test_new_owner_preview_requires_exact_issued_request_and_native_result(defect, version):
    from cais_spade_llm.resources.resource_safety_preparation import prepare_outline_event_preview_request, finalize_outline_event_preview

    scene, snapshot, model, task = _event_case()
    scene["physical_preview_contract_version"] = version
    evidence = {}
    if version == 4:
        scene["resource_models"]["ur5e-3"]["static_capabilities"] = {}
        evidence = {"source": {"kind": "recorded", "capture": "fixture", "revision": 1}}
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence)
    response = _issued_native_preview(prepared)
    if defect == "missing_registry":
        prepared.pop("native_requests")
    elif defect == "request_scene":
        response["request"]["planning_scene"]["world"] = {"collision_objects": []}
    elif defect == "result_scene":
        response["scene_fingerprint"] = "different-scene"
    elif defect == "raw_trajectory":
        response["trajectory"]["joint_trajectory"]["points"][-1]["positions"] = [.1]
    elif defect == "request_target":
        response["request"]["joint_target"]["position"] = [.1]
    elif defect == "proposal":
        response["proposal_fingerprint"] = "different-proposal"
    elif defect == "execution":
        response["command_sent"] = True
    elif defect == "source":
        response["source"] = "another provider"
    result = finalize_outline_event_preview(prepared_preview=prepared, motion_results=[response])
    assert result["status"] == ("prepared" if defect is None else "NEEDS_CONTEXT"), result
    if defect is not None:
        return
    original = deepcopy(result["event_evidence"])
    assert result["geometry_evidence"]["successor_evidence"]["physical_preview_contract_version"] == version
    for field, value in (("stage_index", 1), ("stage_state_fingerprint", "stale"),
            ("trajectory_fingerprint", "stale"), ("proposal_fingerprint", "stale"),
            ("resource_id", "ur5e-4"), ("request_fingerprint", ""), ("response_fingerprint", "")):
        records = deepcopy(original)
        records[0]["intervals"][0]["native_preview_binding"][field] = value
        rejected = project_recovery_outline_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
            evidence=prepared["physical_evidence"], event_evidence=records)
        assert rejected["status"] == "needs_context", rejected
    records = deepcopy(original)
    records[0]["intervals"][0].pop("native_preview_binding")
    assert project_recovery_outline_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        evidence=prepared["physical_evidence"], event_evidence=records)["status"] == "needs_context"


def test_independent_named_event_preview_supports_cca_without_union_order_or_live_calls():
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )

    scene, snapshot, model, task = _event_case()
    original = deepcopy((scene, snapshot, model, task))
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence={})
    assert prepared["status"] == "ready", prepared
    result = finalize_outline_event_preview(prepared_preview=prepared,
        motion_results=[_preview_response(prepared["motion_requests"][0])])
    assert result["status"] == "prepared", result
    assert not set(result["event_evidence"][0]) & {"trace", "primitives", "primitive_support"}
    projection = result["geometry_evidence"]
    assert projection["projected_resource_snapshot"]["resource_state"] == "idle"
    assert projection["interval_complete"] is True
    assert _validate(projection, all_rules=True, prospective_activation=True)["is_safe"]
    assert (scene, snapshot, model, task) == original
    assert result["execution_authorized"] is False and result["command_sent"] is False


@pytest.mark.parametrize("historical_version,required_version", [(1, 2), (2, 3), (3, 4)])
def test_new_attempt_rejects_historical_event_preview_contract(historical_version, required_version):
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )

    scene, snapshot, model, task = _event_case()
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence={})
    result = finalize_outline_event_preview(prepared_preview=prepared,
        motion_results=[_preview_response(prepared["motion_requests"][0])])
    historical = prepared["physical_evidence"]
    record = result["event_evidence"][0]
    contract = historical["resource_models"]["ur5e-3"]["event_projection_contracts"][record["contract_id"]]
    contract["version"] = historical_version
    record["contract_fingerprint"] = recovery_validation_fingerprint(contract)
    assert project_recovery_outline_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        evidence=historical, event_evidence=[record])["status"] == "projected"
    historical["physical_preview_contract_version"] = required_version
    record["binding"] = outline_event_binding(task, snapshot, model, historical)
    rejected = project_recovery_outline_event(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, evidence=historical, event_evidence=[record])
    assert rejected["status"] == "needs_context"
    assert "required version" in str(rejected["unresolved"])


def test_loaded_destination_preview_carries_declared_private_context_without_releasing(monkeypatch):
    from cais_spade_llm.resources import resource_safety_preparation as preparation
    from cais_spade_llm.resources.robot import recovery_capability
    from cais_spade_llm.resources.recovery_parameter_evidence import projected_resource_evidence

    scene, snapshot, model, task = _event_case()
    part, rid = "gear_small", "ur5e-3"
    owner = scene["snapshot"]["resources"][rid]
    owner.update(held_part=part, grasp_transform=[0., 0., 0., 0., 0., 0., 1.])
    scene["resource_models"][rid]["event_endpoint_tolerance"] = {"current_pose": 1e-12}
    scene["snapshot"]["parts"][part]["model_name"] = part
    snapshot.update(held_part=part, resource_state="picked")
    model["state_variables"].update({
        "resource_location": {"scope": "resource", "domain": ["home", "assembly_board-v1"]},
        "resource_state": {"scope": "resource", "domain": ["picked", "positioned"]},
        "held_part": {"scope": "resource", "domain": [part]},
        "part_state": {"scope": "part", "domain": ["in_gripper", "in_transit"]},
        "part_location": {"scope": "part", "domain": [task["resource_jid"]]},
        "task_ctx.destination_location": {"scope": "resource", "private": True,
                                           "domain": [None, "assembly_board-v1"]}})
    model["events"] = [{"event_name": "recorded_loaded_transfer", "requires_part_binding": True,
        "updates": {"resource_state": {"set": "positioned"},
            "resource_location": {"set_from_param": "destination_location"},
            "part_state": {"set": "in_transit"}, "part_location": {"set": task["resource_jid"]},
            "task_ctx.destination_location": {"set_from_param": "destination_location"}},
        "primitive_support": [{"primitive": "compute_place_targets", "params": {}}],
        "composition_contract": {"part_transition": {"completed": {
            "state": "in_transit", "location_template": "{resource_jid}"}}}}]
    task.update(part_name=part, expected_end_state={"resource_state": "positioned",
        "resource_location": "assembly_board-v1"})
    evidence = {"controller_config": {"parts_tuning": {"insertion_depth_m": .0025}}}
    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **_: {
        "status": "calculated", "target_reference": {"surface_role": "assembly_slot"},
        "insert_pose": {"x": 1., "y": 0., "z": -.0025, "qx": 0., "qy": 0., "qz": 0., "qw": 1.}})
    prepared = preparation.prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence)
    assert prepared["status"] == "ready", prepared
    assert prepared["custody_operation"] is None
    assert prepared["motion_requests"][0]["waypoints"][-1]["position"] == {"x": 1., "y": 0., "z": 0.}
    result = preparation.finalize_outline_event_preview(prepared_preview=prepared,
        motion_results=[_preview_response(prepared["motion_requests"][0], target=1. + 1e-14)])
    assert result["status"] == "prepared", result.get("geometry_evidence", {}).get("unresolved", result)
    projection = result["geometry_evidence"]
    derived = projection["projected_resource_snapshot"]
    assert 1. < derived["current_pose"][0] < 1. + 1e-13
    assert derived["current_pose"] == projection["endpoint_evidence"]["current_pose"]["actual"]
    assert projection["endpoint_evidence"]["current_pose"]["position_residual"] > 0.
    assert derived["held_part"] == part
    assert projection["part_updates"] == {"part_state": "in_transit", "part_location": task["resource_jid"]}
    assert derived["task_ctx.destination_location"] == "assembly_board-v1"
    restored = projected_resource_evidence(resource_jid=task["resource_jid"], resource_evidence=evidence,
        physical_evidence=projection["successor_evidence"], recovery_snapshot=derived)
    assert restored["state"]["_task_ctx"]["destination_location"] == "assembly_board-v1"
    assert projection["projected_snapshot"]["resources"]["ur5e-4"]["joint_positions"] == [0]
    assert _validate(projection, all_rules=True, prospective_activation=True)["is_safe"]
    record = result["event_evidence"][0]
    scene["resource_models"] = result["resource_models"]
    scene["event_evidence"] = result["event_evidence"]
    contract = scene["resource_models"][rid]["event_projection_contracts"][record["contract_id"]]
    contract["private_resource_updates"]["task_ctx.destination_location"] = "home"
    record["contract_fingerprint"] = recovery_validation_fingerprint(contract)
    invalid = project_recovery_outline_event(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, evidence=scene)
    assert invalid["status"] == "needs_context"
    assert "differs from its declaration" in invalid["unresolved"][0]


def test_independent_event_preview_requires_exact_scene_and_owner_binding():
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )

    scene, snapshot, model, task = _event_case()
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence={})
    result = finalize_outline_event_preview(prepared_preview=prepared,
        motion_results=[_preview_response(prepared["motion_requests"][0])])
    scene["resource_models"] = result["resource_models"]
    scene["event_evidence"] = result["event_evidence"]
    scene["snapshot"]["resources"]["ur5e-4"]["joint_positions"] = [0.2]
    blocked = project_recovery_outline_event(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, evidence=scene)
    assert blocked["status"] == "needs_context"
    assert "projected starting scene" in blocked["unresolved"][0]


def _numerical_endpoint_case():
    from cais_spade_llm.resources.recovery_geometry import owner_cartesian_endpoint_precision

    target = [1., 0., 0., 0., 0., 0., 1.]
    actual = [1. + 1e-14, 0., 0., 0., 0., -2e-15, -1.]
    contract = {"endpoint": {"current_pose": {"set_from_param": "target_pose"}},
                "endpoint_numerical_precision": {"current_pose": owner_cartesian_endpoint_precision()}}
    trajectory = {"joint_names": ["slide"], "points": [{"positions": [0.]}, {"positions": [actual[0]]}]}
    proof = {"target_pose": deepcopy(target),
        "trajectory_fingerprint": recovery_validation_fingerprint(trajectory),
        "cartesian_position_precision": 1e-13, "cartesian_orientation_precision": 1e-13,
        "cartesian_position_residual": 1e-14, "cartesian_orientation_residual": 4e-15}
    return contract, {"target_pose": target}, {"resources": {"owner": {"current_pose": actual}}}, [
        {"joint_trajectory": trajectory, "native_endpoint_evidence": proof}]


def test_owner_numerical_endpoint_retains_actual_fk_and_quaternion_sign():
    from cais_spade_llm.resources.recovery_geometry import _event_endpoint

    contract, bindings, state, intervals = _numerical_endpoint_case()
    before = deepcopy((contract, bindings, state, intervals))
    checked = _event_endpoint(contract, bindings, state, "owner", None, intervals)
    assert 0 < checked["current_pose"]["position_residual"] < 1e-13
    assert checked["current_pose"]["orientation_residual"] == pytest.approx(4e-15, abs=1e-28)
    assert checked["current_pose"]["actual"] != checked["current_pose"]["target"]
    assert (contract, bindings, state, intervals) == before


@pytest.mark.parametrize("defect", ["contract_bound", "native_bound", "native_residual", "target",
                                  "trajectory", "missing_native", "position", "orientation"])
def test_owner_numerical_endpoint_rejects_incompatible_evidence(defect):
    from cais_spade_llm.resources.recovery_geometry import _event_endpoint

    contract, bindings, state, intervals = _numerical_endpoint_case()
    proof = intervals[0]["native_endpoint_evidence"]
    if defect == "contract_bound":
        contract["endpoint_numerical_precision"]["current_pose"]["position_precision"] = .02
    elif defect == "native_bound":
        proof["cartesian_orientation_precision"] = .02
    elif defect == "native_residual":
        proof["cartesian_position_residual"] = .001
    elif defect == "target":
        proof["target_pose"][0] += .001
    elif defect == "trajectory":
        intervals[0]["joint_trajectory"]["points"][-1]["positions"][0] += .001
    elif defect == "missing_native":
        intervals[0].pop("native_endpoint_evidence")
    elif defect == "position":
        state["resources"]["owner"]["current_pose"][0] += 2e-13
    else:
        state["resources"]["owner"]["current_pose"][5] = 2e-13
    with pytest.raises(ValueError):
        _event_endpoint(contract, bindings, state, "owner", None, intervals)


def test_endpoint_without_owner_numerical_contract_keeps_exact_requirements():
    from cais_spade_llm.resources.recovery_geometry import _event_endpoint

    contract, bindings, state, intervals = _numerical_endpoint_case()
    contract.pop("endpoint_numerical_precision")
    with pytest.raises(ValueError, match="does not establish current_pose"):
        _event_endpoint(contract, bindings, state, "owner", None, intervals)


def test_event_binding_preserves_recorded_fingerprints_without_mutating_inputs():
    scene, snapshot, model, task = _event_case()
    before = deepcopy((scene, snapshot, model, task))
    result = outline_event_binding(task, snapshot, model, scene)
    assert result == {
        "version": 1,
        "resource_jid": "ur5e-3@localhost",
        "conditions_fingerprint": "eb837d1238bf90a5c7bcbfd1fdf59bd4cd80900dc24a84335d9058f3542f7a80",
        "snapshot_fingerprint": "9378dc17454aef1166d93fac47a7d2bddf1f8964d5b0bc19922cae00a3690162",
        "model_fingerprint": "1d82b7083c4c4fd0fda7962f9a1d71bc0165dfc57d43e734dfa0111fbb15d3a0",
        "scene_fingerprint": "9d9c7d9d1efe6e8bafd40563032a21b8f72f27aaf6d378f9799691aef0f9184e",
    }
    assert (scene, snapshot, model, task) == before
    scene["snapshot"]["resources"]["ur5e-4"]["joint_positions"][0] += .1
    changed = outline_event_binding(task, snapshot, model, scene)
    assert changed["scene_fingerprint"] != result["scene_fingerprint"]
    assert {key: value for key, value in changed.items() if key != "scene_fingerprint"} == {
        key: value for key, value in result.items() if key != "scene_fingerprint"}


def test_additional_saved_preview_does_not_change_bound_physical_scene():
    scene, snapshot, model, task = _event_case()
    before = outline_event_binding(task, snapshot, model, scene)
    scene["resource_models"]["ur5e-3"]["event_projection_contracts"] = {"another": {"version": 1}}
    scene["resource_models"]["ur5e-3"]["named_joint_poses"]["another"] = {"joint_names": ["slide"], "positions": [.2]}
    assert outline_event_binding(task, snapshot, model, scene) == before
    scene["resource_models"]["ur5e-3"]["configuration"]["root_xyz"] = [.2, 0, 0]
    assert outline_event_binding(task, snapshot, model, scene) != before


def test_event_motion_retains_projected_peer_and_carried_part_occupancy():
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )

    scene, snapshot, model, task = _event_case()
    peer = scene["snapshot"]["resources"]["ur5e-4"]
    peer.update(joint_positions=[-2], held_part="gear_small", grasp_transform=[0, 0, 0, 0, 0, 0, 1])
    scene["resource_models"]["ur5e-3"]["named_joint_poses"]["home"]["positions"] = [1]
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence={})
    result = finalize_outline_event_preview(prepared_preview=prepared,
        motion_results=[_preview_response(prepared["motion_requests"][0], target=1)])
    assert result["status"] == "prepared", result
    projection = result["geometry_evidence"]
    assert projection["projected_snapshot"]["resources"]["ur5e-4"]["joint_positions"] == [-2]
    assert projection["observations"][-1]["carried_parts"]["ur5e-4"] == ["gear_small"]
    assert projection["observations"][-1]["part_region_occupancy"]["assembly_board-v1"]["gear_small"] is True
    assert not _validate(projection, prospective_activation=True)["is_safe"]


def _correlated_motion_preview(*, crosses=False, tracking_error=False):
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )
    from cais_spade_llm.resources.recovery_geometry import _checkpoint, _prepared_motion

    scene, snapshot, model, task = _event_case()
    rid, part = "ur5e-3", "KET4_Square_4mm"
    config = scene["resource_models"][rid]["configuration"]
    config["joint_names"].append("return")
    config["joints"][0]["child"] = "intermediate"
    config["joints"].append({**deepcopy(config["joints"][0]), "name": "return",
                           "parent": "intermediate", "child": "body"})
    if tracking_error:
        config["joint_position_error"] = {"slide": .5, "return": .5}
    owner = scene["snapshot"]["resources"][rid]
    owner.update(joint_names=["slide", "return"], joint_positions=[0., 0.],
                 held_part=part, grasp_transform=[0., 0., 0., 0., 0., 0., 1.])
    snapshot["held_part"] = part
    scene["snapshot"]["resources"]["ur5e-4"]["joint_positions"] = [-2.]
    positions = [[0., 0.], [1., 0.], [0., 0.]] if crosses else [[0., 0.], [1., -1.]]
    scene["resource_models"][rid]["named_joint_poses"]["home"] = {
        "joint_names": ["slide", "return"], "positions": positions[-1]}
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence={})
    assert prepared["status"] == "ready", prepared
    response = _preview_response(prepared["motion_requests"][0])
    trajectory = response["joint_trajectory"]
    trajectory["joint_names"] = ["slide", "return"]
    trajectory["points"] = [{"positions": values, "velocities": [], "accelerations": [], "effort": [],
        "time_from_start": {"sec": index, "nanosec": 0}} for index, values in enumerate(positions)]
    trajectory["duration_ns"] = (len(positions) - 1) * 1_000_000_000
    record = {"joint_trajectory": trajectory, "stationary_resources": ["ur5e-4"],
              "stationary_parts": ["gear_small"]}
    motion = _prepared_motion(scene["snapshot"], rid, scene["resource_models"], record)
    coarse = _checkpoint(scene["snapshot"], scene["geometry"], scene["resource_models"], index=1,
                         moving=(rid, motion, 0, motion.duration))
    result = finalize_outline_event_preview(prepared_preview=prepared, motion_results=[response])
    assert result["status"] == "prepared", result
    return coarse, result["geometry_evidence"]


def test_certified_cells_tighten_correlated_motion_without_extra_monitor_ticks():
    coarse, projection = _correlated_motion_preview()
    assert coarse["region_occupancy"]["assembly_board-v1"]["ur5e-3"] is None
    assert coarse["part_region_occupancy"]["assembly_board-v1"]["KET4_Square_4mm"] is None
    between = projection["observations"][1]
    assert len(projection["observations"]) == 4
    assert 1 < len(between["continuous_cells"]) <= 16
    for cell in between["continuous_cells"]:
        assert cell["occupancy_possibilities"]["assembly_board-v1"]["ur5e-3"] == [False]
        assert cell["occupancy_possibilities"]["assembly_board-v1"]["ur5e-4"] == [True]
        assert cell["part_entry_possibilities"]["assembly_board-v1"]["KET4_Square_4mm"] == [False]
    result = _validate(projection, all_rules=True, prospective_activation=True)
    assert result["is_safe"], result["reason"]
    assert len(result["monitor_context"]["observations"]) == len(projection["observations"])
    assert len(result["monitor_context"]["observations"][1]["rule_cells"]) == len(between["continuous_cells"])
    assert result["execution_authorized"] is False


@pytest.mark.parametrize("defect", ["crosses", "tracking_error", "gap"])
def test_refinement_preserves_intermediate_violations_uncertainty_and_coverage(defect):
    _, projection = _correlated_motion_preview(crosses=defect == "crosses", tracking_error=defect == "tracking_error")
    between = projection["observations"][1]
    if defect == "gap":
        between["continuous_cells"].pop(0)
    if defect == "tracking_error":
        assert len(between["continuous_cells"]) == 16
        assert between["region_occupancy"]["assembly_board-v1"]["ur5e-3"] is None
    result = _validate(projection, all_rules=True, prospective_activation=True)
    assert result["is_safe"] is False
    if defect == "gap":
        assert "cover the exact motion interval" in result["reason"]


def test_cca_reports_the_definite_refined_cell_instead_of_coarse_occupancy():
    _, projection = _correlated_motion_preview(crosses=True)
    before = deepcopy(projection)
    result = _validate(projection, prospective_activation=True)
    assert result["validation_status"] == "INFEASIBLE", result
    counterexample = result["counterexample"]
    assert counterexample["definite_violation"] is True
    assert counterexample["ap_values"] == {"ap001": [True], "ap002": [True]}
    assert counterexample["observation_ap_values"]["ap001"] == [False, True]
    observation = projection["observations"][counterexample["observation_index"]]
    cell = observation["continuous_cells"][counterexample["cell_index"]]
    assert counterexample["interval"] == cell["interval"]
    assert counterexample["from"] and counterexample["to"] == []
    assert "monitor_context" not in result
    assert result["execution_authorized"] is False
    assert projection == before


def test_cca_uncertain_cells_request_bound_evidence_without_claiming_violation():
    _, projection = _correlated_motion_preview(tracking_error=True)
    before = deepcopy(projection)
    result = _validate(projection, prospective_activation=True)
    assert result["validation_status"] == "NEEDS_CONTEXT", result
    counterexample = result["counterexample"]
    assert counterexample["definite_violation"] is False
    assert counterexample["ap_values"] == {"ap001": [False, True], "ap002": [True]}
    observation = projection["observations"][counterexample["observation_index"]]
    assert counterexample["interval"] == observation["continuous_cells"][counterexample["cell_index"]]["interval"]
    assert counterexample["to"]
    assert result["context_requests"] == ["/physical_evidence"]
    assert counterexample["rule_id"] in result["missing_facts"][0]
    assert "monitor_context" not in result
    assert result["execution_authorized"] is False
    assert _validate(projection, prospective_activation=True) == result
    assert projection == before


@pytest.mark.parametrize('specification', [
    'safety_assembly_board-v1_mutex.txt',
    'safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt',
])
@pytest.mark.parametrize('later_inside', [False, True])
def test_cca_checks_later_observations_without_discarding_earlier_uncertainty(specification, later_inside):
    from cais_spade_llm.resources.recovery_geometry import _checkpoint, _part_entries

    _, projection = _correlated_motion_preview(tracking_error=True)
    scene = projection['successor_evidence']
    snapshot = deepcopy(projection['projected_snapshot'])
    snapshot['resources']['ur5e-3']['joint_positions'] = [1., 0.] if later_inside else [0., 0.]
    last = _checkpoint(snapshot, projection['geometry'], scene['resource_models'], index=2)
    projection['observations'] = [*projection['observations'][:2], last]
    _part_entries(projection['observations'])
    document = json.loads((Path(__file__).resolve().parents[1] /
        'cais_spade_llm/specification/safety' / specification).read_text())
    before = deepcopy(projection)
    result = validate_outline_evidence_safety(projection=projection, catalog=document['catalog'],
        requirement_scopes=document['requirement_scopes'], prospective_activation=True)
    assert {row['observation_index'] for row in result['rule_checks']} == {0, 1, 2}
    counterexample = result['counterexample']
    assert counterexample['definite_violation'] is later_inside
    if later_inside:
        assert result['validation_status'] == 'INFEASIBLE'
        assert counterexample['observation_index'] == 2
        assert counterexample['ap_values']['ap001'] == [True]
        assert counterexample['to'] == []
    else:
        assert result['validation_status'] == 'NEEDS_CONTEXT'
        assert counterexample['observation_index'] == 1
        assert counterexample['ap_values']['ap001'] == [False, True]
        assert counterexample['to']
    assert 'monitor_context' not in result and 'monitor_context_after' not in result
    assert result['execution_authorized'] is False
    assert projection == before


@pytest.mark.parametrize("missing", ["stationary_contract", "stationary_parts", "path_complete"])
def test_independent_event_preview_rejects_missing_interval_evidence(missing):
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )

    scene, snapshot, model, task = _event_case()
    if missing == "stationary_contract":
        scene["resources"]["ur5e-4"].pop("stationary_contract")
    elif missing == "stationary_parts":
        scene.pop("stationary_parts")
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence={})
    response = _preview_response(prepared["motion_requests"][0])
    if missing == "path_complete":
        response["path_complete"] = False
    result = finalize_outline_event_preview(prepared_preview=prepared, motion_results=[response])
    assert result["status"] == "NEEDS_CONTEXT"
    assert "event_evidence" not in result


def _custody_event_case(grasp):
    scene, snapshot, model, task = _event_case()
    part, rid = "KET4_Square_4mm", "ur5e-3"
    model["state_variables"].update(held_part={"scope": "resource", "domain": [None, part]},
        part_state={"scope": "part", "domain": ["ready", "in_gripper"]},
        part_location={"scope": "part", "domain": [rid, "Buffer For Machined parts"]})
    owner = scene["snapshot"]["resources"][rid]
    owner.update(held_part=None if grasp else part, gripper_state="open" if grasp else "closed")
    owner["grasp_transform"] = None if grasp else [0, 0, 0, 0, 0, 0, 1]
    scene["snapshot"]["parts"][part].update(current_pose=[0, 0, 0, 0, 0, 0, 1], model_name="square")
    scene["snapshot"]["resources"]["Buffer For Machined parts"] = {"contained_parts": [], "held_part": None}
    scene["jids"]["Buffer For Machined parts"] = "buffer@localhost"
    scene["geometry"]["resources"]["Buffer For Machined parts"] = {
        "component_bounds": [{"id": "buffer", "bounds": [[-4, -3]] * 3}]}
    scene["resources"]["Buffer For Machined parts"] = {"model_static": True}
    configuration = scene["resource_models"][rid]["configuration"]
    configuration["joints"][0]["axis"] = [0., 0., 1.]
    configuration["joints"].append({"name": "finger", "parent": "body", "child": "finger", "type": "prismatic",
        "xyz": [0, 0, 0], "rpy": [0, 0, 0], "axis": [0, 1, 0], "frozen_position": .1 if grasp else .02})
    configuration["components"].append({"id": "finger", "link": "finger", "bounds": [[-.01, .01]] * 3})
    scene["resource_models"][rid]["primitive_model"] = {"configuration": {"custody": {
        "kind": "gazebo_link_attacher", "preserves_part_pose_on_attachment": True,
        "gripper": {"open": .1, "closed": .02},
        "joint_positions": {"open": {"finger": .1}, "closed": {"finger": .02}},
        "joint_bounds": {"finger": [0, .12]}}}}
    scene["part_support_policy"] = {"kind": "gravity_supported_rest", "requires_no_active_goals": True,
        "future_execution_tracking": "not_established", "stationary_linear_speed_max": .001,
        "stationary_angular_speed_max": .001}
    scene["physics_evidence"] = _supported_physics("square")
    snapshot.update(held_part=owner["held_part"])
    task.update(part_name=part, expected_end_state={"held_part": part if grasp else None,
        "part_state": "in_gripper" if grasp else "ready",
        "part_location": rid if grasp else "Buffer For Machined parts"})
    model["events"] = [{"event_name": "pick" if grasp else "place", "updates": {
        "held_part": {"set": part if grasp else None}},
        "primitive_support": [{"primitive": "grasp_part" if grasp else "release_part", "params": {}}],
        "composition_contract": {"steps": [
            {"id": "grasp" if grasp else "release", "op": "grasp_part" if grasp else "release_part", "params": {}},
            {"id": "lift", "op": "move_relative", "params": {"dx": 0., "dy": 0., "dz": {"$sub": [
                {"$state": "_task_ctx", "path": ["travel_z"]}, {"$state": "_position", "path": ["z"]}]}}}]}}]
    scene["stationary_part_evidence"] = {name: {"status": "supported", "current_pose": deepcopy(row["current_pose"])}
        for name, row in scene["snapshot"]["parts"].items()}
    return scene, snapshot, model, task


def _nominal_program_case():
    scene, snapshot, model, task = _custody_event_case(False)
    part = task["part_name"]
    task["required_task_contract"] = "place"
    task["params"] = {}
    context = {"part_name": part, "model_name": "square", "travel_z": .2,
               "move_insert_mode": "", "surface_role": "support"}
    evidence = {"execution_mode": "simulation", "state": {"_task_ctx": context},
                "inputs": {"geometry": {"parts": {"model_map": {part: "square"}}}},
                "source": {"capture_revision": 2}}
    steps = [
        {"id": "insert", "op": "move_insert", "when": [{"predicate": "move_insert_required"}],
         "params": {"target_pose": {"$state": "_task_ctx", "path": ["insert_pose"]}}},
        {"id": "release", "op": "release_part", "params": {"part_name": {"$state": "_held_part"},
                                                               "model_name": {"$state": "_task_ctx", "path": ["model_name"]}}},
        {"id": "snap", "op": "snap_part_to_slot", "when": [{"predicate": "execution_mode", "args": {"mode": "simulation"}}],
         "params": {"model_name": {"$state": "_task_ctx", "path": ["model_name"]}, "slot_x": .1}},
        {"id": "lift", "op": "move_relative", "params": {"dx": 0., "dy": 0., "dz": {"$sub": [
            {"$state": "_task_ctx", "path": ["travel_z"]}, {"$state": "_position", "path": ["z"]}]}}},
    ]
    model["events"][0]["composition_contract"] = {"params": {}, "steps": steps}
    return scene, snapshot, model, task, evidence


def test_nominal_owner_program_skips_declared_insert_and_retains_snap():
    from cais_spade_llm.resources.resource_safety_preparation import _outline_nominal_program

    scene, snapshot, model, task, evidence = _nominal_program_case()
    before = deepcopy((scene, snapshot, model, task, evidence))
    result = _outline_nominal_program(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
                                     physical_evidence=scene, resource_evidence=evidence)
    assert result["custody_motion_index"] == 0
    assert len(result["motion_requests"]) == 1
    assert result["motion_requests"][0]["waypoints"][0]["position"]["z"] == .2
    assert result["nominal_snap"]["params"] == {"model_name": "square", "slot_x": .1}
    assert [row["interval_indices"] for row in result["step_evidence"]] == [[], [0], [1], [2]]
    assert result["step_evidence"][1]["state_before"]["_held_part"] == task["part_name"]
    assert result["step_evidence"][2]["state_before"]["_held_part"] is None
    assert (scene, snapshot, model, task, evidence) == before


def test_nominal_owner_program_uses_registered_place_outputs_for_later_insert(monkeypatch):
    from dataclasses import asdict
    from cais_spade_llm.resources import resource_safety_preparation as preparation
    from cais_spade_llm.resources.robot import recovery_capability
    from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry
    from cais_spade_llm.recovery_framework import workflow_execution

    scene, snapshot, model, task, evidence = _nominal_program_case()
    registry = robot_task_registry()
    programs = {}
    for name in ("place_approach", "place_insert"):
        definition = registry[name]
        programs[name] = json.loads(json.dumps(asdict(definition.program)))
        programs[name]["params"] = definition.argument_properties()
        programs[name]["required_params"] = definition.required_argument_names()
    model["events"] = [{"event_name": name, "composition_contract": program}
                       for name, program in programs.items()]
    task.update(required_task_contract="place_approach", params={"part_name": task["part_name"],
        "destination_location": "assembly_board-v1", "product_geometry": {"source": "recorded product geometry"}},
        expected_end_state={"resource_location": "assembly_board-v1"})
    evidence["controller_config"] = {"parts_tuning": {"insertion_depth_m": .0025}}
    evidence["inputs"]["geometry"]["parts"]["dimensions_m"] = {task["part_name"]: [.1, .1, .1]}
    monkeypatch.setattr(workflow_execution, "_destination_geometry", lambda *args: {"source": "recorded product geometry"})
    target = {"status": "calculated", "insert_pose": dict(zip(
        ("x", "y", "z", "qx", "qy", "qz", "qw"), [.1, .2, .3, 0., 0., 0., 1.], strict=True)),
        "target_reference": {"surface_role": "assembly_slot"}, "slot_x": .1, "slot_y": .2,
        "board_top_z": .25, "place_part_origin_z": .25, "part_height": .1,
        "target_origin_pose": {"x": .1, "y": .2, "z": .25}, "destination_location": "assembly_board-v1"}
    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **_: deepcopy(target))
    result = preparation._outline_nominal_program(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence)
    assert len(result["motion_requests"]) == 2
    assert result["motion_requests"][0]["waypoints"][0]["position"]["z"] == .3525
    assert result["motion_requests"][1]["waypoints"][0]["position"]["z"] == .3025
    effects = preparation._outline_nominal_task_effects(task=task, recovery_des_model=model,
        resource_evidence=evidence, nominal=result)
    assert effects["status"] == "supported", effects
    assert effects["state"]["_task_ctx"]["move_insert_mode"] == "simulation_direct"
    assert "move_insert_profile" not in effects["state"]["_task_ctx"]
    assert effects["state"]["_task_ctx"]["travel_z"] == .3525
    assert effects["state"]["_task_ctx"]["resolved_cartesian_positions"]["descend"]["z"] == .3025
    evidence["state"] = effects["state"]
    scene["snapshot"]["resources"]["ur5e-3"]["current_pose"] = [.1, .2, .3025, 0., 0., 0., 1.]
    task.update(required_task_contract="place_insert", params={"part_name": task["part_name"],
                                                              "destination_location": "assembly_board-v1"})
    inserted = preparation._outline_nominal_program(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence)
    assert len(inserted["motion_requests"]) == 2
    assert inserted["motion_requests"][0]["waypoints"][0]["position"]["z"] == .3
    assert inserted["custody_motion_index"] == 1
    assert inserted["nominal_snap"]["params"]["part_origin_z"] == .25
    task.update(required_task_contract="place_approach", params={"part_name": task["part_name"],
        "destination_location": "assembly_board-v1", "product_geometry": {"source": "different geometry"}})
    with pytest.raises(ValueError, match="product_geometry differs"):
        preparation._outline_nominal_program(task=task, recovery_snapshot=snapshot,
            recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence)


def test_nominal_place_calculation_rejects_unmodeled_declared_adjustment(monkeypatch):
    from cais_spade_llm.resources.resource_safety_preparation import _outline_nominal_place_targets

    with pytest.raises(ValueError, match="arguments differ"):
        _outline_nominal_place_targets(task={"part_name": "part", "expected_end_state": {"resource_location": "destination"}},
            recovery_snapshot={}, physical_evidence={}, resource_evidence={"execution_mode": "simulation", "state": {"_task_ctx": {}}},
            params={"part_name": "part", "destination_location": "destination", "pick_ctx": {},
                    "product_geometry": {}, "assembly_board_v1_aruco": None, "z_adjustment_m": .02})


def test_nominal_owner_program_replays_actual_checkpoints_and_every_step():
    from cais_spade_llm.resources.resource_safety_preparation import _outline_nominal_program
    from cais_spade_llm.resources.robot.recovery_task_projection import nominal_task_projection

    scene, snapshot, model, task, evidence = _nominal_program_case()
    kwargs = dict(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
                  physical_evidence=scene, resource_evidence=evidence)
    prepared = _outline_nominal_program(**kwargs)
    intervals = [{"custody_effects": [{"kind": "release", "part": task["part_name"]}], "interval_complete": True},
        {"nominal_snap_evidence": {"status": "supported", "params": prepared["nominal_snap"]["params"],
                                    "binding": {"source": "independently checked fixture"}, "physical_execution_verified": False}},
        {"motion_request": prepared["motion_requests"][0], "joint_trajectory": {"points": [{"positions": [.2]}]},
         "native_preview_binding": {"source": "independently checked fixture"}}]
    states = [deepcopy(scene["snapshot"]["resources"]["ur5e-3"]) for _ in range(4)]
    for row in states[1:]:
        row["held_part"] = None
    states[-1]["current_pose"][2] = .2
    replay = _outline_nominal_program(**kwargs, intervals=intervals, interval_states=states)
    witness = nominal_task_projection(task=task, recovery_des_model=model, resource_evidence=evidence,
        step_evidence=replay["step_evidence"], intervals=intervals)
    assert witness["status"] == "supported", witness
    assert [row["status"] for row in witness["steps"]] == ["skipped", "supported", "supported", "supported"]
    assert replay["state"]["_position"]["z"] == .2
    incomplete = deepcopy(intervals)
    incomplete.pop(1)
    with pytest.raises((ValueError, IndexError)):
        _outline_nominal_program(**kwargs, intervals=incomplete, interval_states=states[:-1])


def _nominal_departure_case(monkeypatch, *, context_present=True):
    from cais_spade_llm.resources import resource_safety_preparation as preparation
    from cais_spade_llm.resources.robot import recovery_capability

    scene, snapshot, model, task = _custody_event_case(False)
    rid, part = "ur5e-3", task["part_name"]
    scene["snapshot"]["resources"][rid]["current_pose"] = [0., 0., 0., 0., 0., 0., 1.]
    scene["physical_preview_contract_version"] = 7
    scene["resource_models"][rid]["static_capabilities"] = {"gripper_reach": {
        "frame": "world", "origin_pose": {"x": 0., "y": 0., "z": 0.},
        "max_xy_radius_m": 1., "tolerance_m": 0., "z_min_m": -1., "z_max_m": 2.}}
    task.update(required_task_contract="place", params={})
    program = model["events"][0]["composition_contract"]
    program["params"] = {}
    program["steps"][0]["params"] = {"part_name": {"$state": "_held_part"}}
    context = {"part_name": None, "model_name": "square", "travel_z": .2, "revision": 1}
    evidence = {"source": {"kind": "recorded", "capture": "fixture", "revision": 1},
        "execution_mode": "simulation", "controller_config": {"move_group": {"frame_id": "world"}},
        "state": {"_task_ctx": deepcopy(context)} if context_present else {}}
    previous = {"part_name": part, "travel_z": 1.5}
    old_contract = {"pickup_target_calculation": previous}
    old_id = recovery_validation_fingerprint(old_contract)
    scene["resource_models"][rid]["event_projection_contracts"] = {old_id: old_contract}
    scene["resources"][rid]["validated_pickup_target_calculation"] = {
        "source": "resource-owned pickup preview", "part_name": part,
        "physical_execution_verified": False, "binding": {"resource_jid": task["resource_jid"]},
        "contract_id": old_id, "contract_fingerprint": old_id, "calculation": previous}
    pose = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), [0., 0., 0., 0., 0., 0., 1.], strict=True))
    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **_: {
        "status": "calculated", "insert_pose": deepcopy(pose)})
    before = deepcopy((scene, snapshot, model, task, evidence))
    prepared = preparation.prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence)
    assert (scene, snapshot, model, task, evidence) == before
    return prepared, evidence


def test_nominal_departure_keeps_current_null_context_instead_of_prior_pickup(monkeypatch):
    from cais_spade_llm.resources import resource_safety_preparation as preparation

    prepared, evidence = _nominal_departure_case(monkeypatch)
    assert prepared["status"] == "ready", prepared
    progress = preparation.outline_event_preview_progress(prepared_preview=prepared, motion_results=[])
    prepared = preparation.prepare_outline_event_departure(prepared_preview=prepared, progress=progress)
    record = prepared["event_record"]
    contract = prepared["physical_evidence"]["resource_models"]["ur5e-3"]["event_projection_contracts"][record["contract_id"]]
    assert contract["placement_target"]["departure_context"] == evidence["state"]["_task_ctx"]
    assert contract["placement_target"]["departure_context"]["part_name"] is None
    assert contract["placement_departure"]["pose"]["z"] == .2
    assert prepared["motion_requests"][0]["waypoints"][0]["position"]["z"] == .2
    assert contract["placement_departure"]["context_source"] == {
        "kind": "nominal_task", "task_contract": "place",
        "context_fingerprint": recovery_validation_fingerprint(evidence["state"]["_task_ctx"])}


def test_nominal_departure_does_not_substitute_prior_pickup_for_missing_context(monkeypatch):
    prepared, _ = _nominal_departure_case(monkeypatch, context_present=False)
    assert prepared["status"] == "NEEDS_CONTEXT"
    assert prepared["reason"] == "Nominal placement departure requires its recorded task context"


@pytest.mark.parametrize("tampered", [None, "travel_z", "numeric_type", "missing_part_name", "task_contract"])
def test_cached_nominal_departure_rechecks_authoritative_context(monkeypatch, tampered):
    from cais_spade_llm.resources import resource_safety_preparation as preparation
    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record

    prepared, evidence = _nominal_departure_case(monkeypatch)
    progress = preparation.outline_event_preview_progress(prepared_preview=prepared, motion_results=[])
    prepared = preparation.prepare_outline_event_departure(prepared_preview=prepared, progress=progress)
    response = _issued_native_preview(prepared)
    response["trajectory"]["joint_trajectory"]["points"][-1]["positions"] = [.2]
    response = preview_result_record(response["request"], response)
    result = preparation.finalize_outline_event_preview(prepared_preview=prepared, motion_results=[response],
        resource_evidence=evidence)
    assert result["status"] == "prepared", result.get("geometry_evidence", {}).get("unresolved", result)
    record = result["event_evidence"][0]
    physical = deepcopy(prepared["physical_evidence"])
    if tampered:
        contracts = physical["resource_models"]["ur5e-3"]["event_projection_contracts"]
        contract = contracts[record["contract_id"]]
        context = contract["placement_target"]["departure_context"]
        if tampered == "travel_z":
            context["travel_z"] = 1.5
        elif tampered == "numeric_type":
            context["revision"] = True
        elif tampered == "missing_part_name":
            context.pop("part_name")
        source = contract["placement_target"]["departure_context_source"]
        source["context_fingerprint"] = recovery_validation_fingerprint(context)
        if tampered == "task_contract":
            source["task_contract"] = "place_approach"
        contract["placement_departure"]["context_source"] = deepcopy(source)
        identity = recovery_validation_fingerprint(contract)
        contracts[identity] = contract
        record.update(contract_id=identity, contract_fingerprint=identity)
    projected = project_recovery_outline_event(task=prepared["task"], recovery_snapshot=prepared["recovery_snapshot"],
        recovery_des_model=prepared["recovery_des_model"], evidence=physical,
        event_evidence=[record], resource_evidence=evidence)
    if tampered:
        assert projected["status"] == "needs_context"
        assert "Nominal departure differs from its authoritative task context" in projected["unresolved"]
        assert "projected_snapshot" not in projected
    else:
        assert projected["status"] == "projected", projected
        assert projected["task_contract_witness"]["status"] == "supported"


@pytest.mark.parametrize("defect", ["missing_mode", "unknown_operation", "changed_relative_motion"])
def test_nominal_owner_program_does_not_hide_unresolved_or_changed_steps(defect):
    from cais_spade_llm.resources.resource_safety_preparation import _outline_nominal_program

    scene, snapshot, model, task, evidence = _nominal_program_case()
    if defect == "missing_mode":
        evidence["state"]["_task_ctx"].pop("surface_role")
    elif defect == "unknown_operation":
        model["events"][0]["composition_contract"]["steps"][2]["op"] = "unmodeled_effect"
    else:
        model["events"][0]["composition_contract"]["steps"][-1]["params"]["dz"] = {"$state": "missing"}
    with pytest.raises(ValueError):
        _outline_nominal_program(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
                                physical_evidence=scene, resource_evidence=evidence)


def _supported_physics(model_name="square"):
    pose = [0., 0., 0., 0., 0., 0., 1.]
    return {"instance_id": "physics", "simulation_time": 2., "gravity": [0., 0., -9.81],
        "physics_engine": "ode", "physics_configuration": {"sdf":
            "<physics><ode><constraints><contact_surface_layer>0</contact_surface_layer></constraints></ode></physics>"},
        "models": {model_name: {"static": False, "pose": pose[:], "links": {"link": {
            "mass": .1, "pose": pose[:], "center_of_mass": [0., 0., 0.],
            "linear_velocity": [0., 0., 0.], "angular_velocity": [0., 0., 0.],
            "collisions": [{"id": model_name + "::collision", "pose": pose[:],
                "geometry": {"type": "box", "size": [.1, .1, .1]}, "contact_min_depth": 0.}]} }},
            "table": {"static": True, "links": {"link": {"collisions": [{"id": "table::collision",
                "pose": [0., 0., -.1, 0., 0., 0., 1.], "contact_min_depth": 0.,
                "geometry": {"type": "box", "size": [1., 1., .1]}}]}}}},
        "contacts": [{"collision1": model_name + "::collision", "collision2": "table::collision",
            "positions": [[x, y, -.05] for x, y in ((-.05, -.05), (.05, -.05), (.05, .05), (-.05, .05))],
            "normals": [[0., 0., 1.]] * 4, "depths": [0.] * 4}]}


def test_nominal_snap_waits_for_endpoint_evidence_before_departure(monkeypatch):
    from cais_spade_llm.resources import resource_safety_preparation as preparation
    from cais_spade_llm.resources.robot import recovery_capability, recovery_task_projection
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    scene, snapshot, model, task = _custody_event_case(False)
    original = deepcopy(scene)
    target = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), [0., 0., 0., 0., 0., 0., 1.]))
    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **_: {
        "status": "calculated", "insert_pose": target})
    prepared = preparation.prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene,
        resource_evidence={"state": {"_task_ctx": {"part_name": task["part_name"], "travel_z": .2}}})
    assert prepared["status"] == "ready"
    rid = prepared["resource_id"]
    contracts = prepared["physical_evidence"]["resource_models"][rid]["event_projection_contracts"]
    contract = deepcopy(contracts[prepared["event_record"]["contract_id"]])
    contract.update(version=7, nominal_snap={"params": {"model_name": "square"}, "owner_evidence": {}})
    identity = recovery_validation_fingerprint(contract)
    contracts[identity] = contract
    prepared["event_record"].update(contract_id=identity, contract_fingerprint=identity)

    def checked_snap(**kwargs):
        state = deepcopy(kwargs["state"])
        assert state["resources"][rid]["held_part"] is None
        state["parts"][task["part_name"]]["current_pose"][0] = .02
        if kwargs["endpoint_result"] is None:
            return {"status": "NEEDS_CONTEXT", "candidate_state": state, "scene_endpoint_record": {"version": 1}}
        support = deepcopy(kwargs["evidence"]["stationary_part_evidence"][task["part_name"]])
        support["current_pose"] = deepcopy(state["parts"][task["part_name"]]["current_pose"])
        return {"status": "supported", "projected_snapshot": state, "part_name": task["part_name"],
                "params": kwargs["params"], "part_support_evidence": support}

    monkeypatch.setattr(recovery_task_projection, "project_nominal_snap", checked_snap)
    first = _preview_response(prepared["motion_requests"][0])
    progress = preparation.outline_event_preview_progress(prepared_preview=prepared, motion_results=[first])
    assert progress["pending_nominal_snap"]["status"] == "NEEDS_CONTEXT"
    assert not any("nominal_snap_evidence" in row for row in progress["event_record"]["intervals"])
    second = _preview_response(prepared["motion_requests"][1], target=.2)
    with pytest.raises(ValueError, match="cannot follow an unvalidated"):
        preparation.outline_event_preview_progress(prepared_preview=prepared, motion_results=[first, second])
    prepared["nominal_snap_result"] = {"source": "test endpoint response"}
    completed = preparation.outline_event_preview_progress(prepared_preview=prepared, motion_results=[first, second])
    assert "pending_nominal_snap" not in completed
    intervals = completed["event_record"]["intervals"]
    assert "custody_effects" in intervals[1]
    assert "nominal_snap_evidence" in intervals[2]
    assert "joint_trajectory" in intervals[3]
    assert completed["physical_evidence"]["snapshot"]["resources"]["ur5e-4"] == original["snapshot"]["resources"]["ur5e-4"]
    assert scene == original


def _nominal_endpoint_rejection_case(monkeypatch):
    from cais_spade_llm.resources import resource_safety_preparation as preparation
    from cais_spade_llm.resources.robot import recovery_capability, recovery_task_projection
    from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_scene_endpoint, preview_result_record

    scene, snapshot, model, task = _custody_event_case(False)
    target = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), [0., 0., 0., 0., 0., 0., 1.]))
    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **_: {
        "status": "calculated", "insert_pose": target})
    prepared = preparation.prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene,
        resource_evidence={"state": {"_task_ctx": {"part_name": task["part_name"], "travel_z": .2}}})
    assert prepared["status"] == "ready", prepared
    rid = prepared["resource_id"]
    contracts = prepared["physical_evidence"]["resource_models"][rid]["event_projection_contracts"]
    contract = deepcopy(contracts[prepared["event_record"]["contract_id"]])
    contract.update(version=7, nominal_snap={"params": {"model_name": "square"}, "owner_evidence": {}})
    identity = recovery_validation_fingerprint(contract)
    contracts[identity] = contract
    prepared["event_record"].update(contract_id=identity, contract_fingerprint=identity)
    motion = _issued_native_preview(prepared)
    raw = deepcopy(motion["request"])
    raw["planning_scene"]["world"] = {"collision_objects": [
        {"id": name, "primitives": [{"type": 1, "dimensions": [1., 1., 1.]}]}
        for name in ("square/link/collision", "obstacle/link/collision")]}
    raw = bind_scene_endpoint(raw, {"version": 1, "status": "prepared", "resource_id": rid,
        "source": {"step_fingerprint": "fixture-snap"}, "changed_object_ids": ["square/link/collision"],
        "support_pairs": [], "command_sent": False, "execution_authorized": False})
    binding = {field: raw[field] for field in ("resource_id", "capture_fingerprint", "proposal_fingerprint", "scene_fingerprint", "model_fingerprint")}
    audit = {"version": 1, "source": "PreviewRecoveryMotion", "endpoint_only": True, "pairs_complete": True,
        "continuous_collision_verified": False, "request_binding": binding, "scene_endpoint_fingerprint": raw["scene_endpoint_fingerprint"],
        "changed_object_ids": ["square/link/collision"], "world_object_ids": ["obstacle/link/collision", "square/link/collision"],
        "support_pairs": [], "robot_collision_checked": True, "robot_collision_free": True, "robot_contacts": [],
        "checked_contacts": 1, "collision_free": False, "checked_pairs": [{"body_1": "square/link/collision",
            "body_2": "obstacle/link/collision", "shape_pairs": 1, "allowed_support_tangency": False,
            "contacts": [{"shape_1": 0, "shape_2": 0, "depth": .01, "position": [0., 0., 0.], "normal": [1., 0., 0.]}]}]}
    endpoint = preview_result_record(raw, {**binding, "status": "SCENE_ENDPOINT_REJECTED",
        "path_complete": False, "fraction": 0., "continuous_collision_verified": False,
        "sampled_collision_rejected": False, "trajectory": {}, "scene_endpoint_fingerprint": raw["scene_endpoint_fingerprint"],
        "scene_endpoint_evidence_json": json.dumps(audit)})
    prepared["nominal_snap_result"] = endpoint

    def rejected_snap(**kwargs):
        assert kwargs["state"]["resources"][rid]["held_part"] is None
        checked = preview_result_record(kwargs["endpoint_result"]["request"], kwargs["endpoint_result"])
        return {"status": "rejected", "scene_endpoint_rejected": True,
            "reason": "The calculated snapped part endpoint has unsupported contact",
            "endpoint_evidence": checked["scene_endpoint_evidence"]}

    monkeypatch.setattr(recovery_task_projection, "project_nominal_snap", rejected_snap)
    return prepared, motion


def test_endpoint_rejection_replays_without_projecting_or_claiming_motion_collision(monkeypatch):
    from cais_spade_llm.resources.resource_safety_preparation import finalize_outline_event_preview
    from cais_spade_llm.recovery_framework.outline_validation import rejected_outline_motion_projection

    prepared, motion = _nominal_endpoint_rejection_case(monkeypatch)
    before = deepcopy(prepared)
    record = finalize_outline_event_preview(prepared_preview=prepared, motion_results=[motion])
    assert record["physical_endpoint_rejected"] is True
    checked = rejected_outline_motion_projection(record=record, task=prepared["task"],
        recovery_snapshot=prepared["recovery_snapshot"], recovery_des_model=prepared["recovery_des_model"],
        physical_evidence=prepared["physical_evidence"])
    assert checked["physical_endpoint_rejected"] and checked["observations"] == []
    assert checked["endpoint_rejection"] == record["endpoint_rejection"]
    assert checked["interval_complete"] is False
    assert not {"projected_snapshot", "event_evidence", "geometry_evidence", "monitor_context"} & record.keys()
    assert prepared == before
    for defect in ("audit", "native_result", "task", "stage"):
        changed = deepcopy(record)
        if defect == "audit": changed["endpoint_rejection"]["checked_pairs"][0]["contacts"][0]["depth"] = 0.
        elif defect == "native_result": changed["nominal_snap_result"]["scene_endpoint_evidence_json"] = "{}"
        elif defect == "task": changed["task"]["event_name"] = "changed"
        else: changed["rejected_stage_index"] += 1
        with pytest.raises(ValueError):
            rejected_outline_motion_projection(record=changed, task=prepared["task"], recovery_snapshot=prepared["recovery_snapshot"],
                recovery_des_model=prepared["recovery_des_model"], physical_evidence=prepared["physical_evidence"])


@pytest.mark.parametrize("nominal", [False, True])
def test_workflow_retains_endpoint_rejection_without_changing_prefix(tmp_path, monkeypatch, nominal):
    from cais_spade_llm.resources.resource_safety_preparation import finalize_outline_event_preview
    from cais_spade_llm.recovery_framework.outline_validation import rejected_outline_motion_projection
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_validated_event

    preview, motion = _nominal_endpoint_rejection_case(monkeypatch)
    record = finalize_outline_event_preview(prepared_preview=preview, motion_results=[motion])
    task, scene = preview["task"], deepcopy(preview["physical_evidence"])
    scene["resource_models"][preview["resource_id"]]["event_projection_contracts"] = {}
    session = {"turn_index": 2, "accepted_outline_prefix": [{"event_name": "retained"}],
        "outline_monitor_context": {"original": {"state": 3}}, "projected_physical_evidence": deepcopy(scene),
        "pending_outline_validation": {"kind": "nominal" if nominal else "event", "task": deepcopy(task),
            "response": {"next_transition": deepcopy(task)}, "recovery_snapshot": deepcopy(preview["recovery_snapshot"])}}
    if nominal:
        session["pending_outline_validation"]["projected_physical_evidence"] = deepcopy(scene)
    prepared = {"multi_turn_session_state": session, "recovery_resources": {task["resource_jid"]: {
        "recovery_des_model": preview["recovery_des_model"], "recovery_snapshot": preview["recovery_snapshot"]}}}
    recovery = SimpleNamespace(_stopped=False, _generation=1, _prepared=prepared,
        offline=True, _offline_cancel=threading.Event(), _observations={},
        runtime=SimpleNamespace(source_observation_path=None),
        agent=SimpleNamespace(process_planner=object()), _state={"artifact_directory": str(tmp_path)})
    recovery._publish = lambda **values: recovery._state.update(values)
    recovery._offline_work = PartSlippageRecovery._offline_work.__get__(recovery)
    recovery._offline_preview_observations = PartSlippageRecovery._offline_preview_observations.__get__(recovery)
    from cais_spade_llm.recovery_framework.recovery_observations import observation_fingerprint

    original_physical = {"snapshot": deepcopy(scene["snapshot"])}
    original_physical["evidence_fingerprint"] = observation_fingerprint(original_physical)
    saved_scene = {"status": "captured", "physical_evidence_fingerprint": original_physical["evidence_fingerprint"],
        "planning_scene": deepcopy(preview["nominal_snap_result"]["request"]["planning_scene"]),
        "start_state": deepcopy(preview["nominal_snap_result"]["request"]["start_state"]),
        "scene_fingerprint": "fixture-scene", "model_fingerprint": "fixture-model",
        "robot_description": "recorded", "robot_description_semantic": "recorded"}
    configuration = {"robots": {"owner": {"group_name": "arm"}}, "planning_scene_mounting_contacts": []}
    recovery._observations = {"physical_evidence": original_physical, "planning_scene_capture": deepcopy(saved_scene),
        "inputs": {"scene": deepcopy(configuration)}}
    preview["physical_evidence"].update(planning_scene_capture=saved_scene, planning_scene_configuration=configuration)
    called = []

    async def retry(*, prepared, session, planner):
        stored = session["projected_physical_evidence"]["rejected_event_evidence"][0]
        checked = rejected_outline_motion_projection(record=stored, task=task, recovery_snapshot=preview["recovery_snapshot"],
            recovery_des_model=preview["recovery_des_model"], physical_evidence=session["projected_physical_evidence"])
        assert checked["physical_endpoint_rejected"]
        assert session["projected_physical_evidence"]["planning_scene_capture"] == saved_scene
        if nominal:
            assert session["pending_outline_validation"]["projected_physical_evidence"]["planning_scene_configuration"] == configuration
        called.append(checked)
        return "need_revision", {"turn_index": 2}

    monkeypatch.setattr(multi_turn_validated_event, "retry_pending_event", retry)
    before = deepcopy(prepared)
    for field in ("planning_scene_capture", "planning_scene_configuration"):
        changed = deepcopy(preview["physical_evidence"])
        changed[field]["changed"] = True
        with pytest.raises(ValueError, match="immutable observations"):
            asyncio.run(PartSlippageRecovery.retry_outline_validation(recovery, {**record, "physical_evidence": changed}))
        assert recovery._prepared == before
    asyncio.run(PartSlippageRecovery.retry_outline_validation(recovery, {**record, "physical_evidence": preview["physical_evidence"]}))
    current = recovery._prepared["multi_turn_session_state"]
    assert len(called) == 1 and recovery._state["status"] == "outline_ready"
    for key in ("accepted_outline_prefix", "outline_monitor_context"):
        assert current[key] == before["multi_turn_session_state"][key]
    assert current["projected_physical_evidence"]["snapshot"] == scene["snapshot"]


def _recorded_target_capabilities(base_y):
    return {"gripper_reach": {"frame": "world", "origin_pose": {"x": 0., "y": base_y, "z": 1.021},
        "max_xy_radius_m": .7, "tolerance_m": .01, "z_min_m": .85, "z_max_m": 1.6}}


@pytest.mark.parametrize("base_y,expected", [(-.5, "INFEASIBLE"), (.5, "FEASIBLE")])
def test_recorded_gear_target_checks_owner_declaration_without_changing_limits(base_y, expected):
    from cais_spade_llm.resources.robot.recovery_capability import check_outline_motion_capabilities

    # Capture 4862656369764af1b78a86063afe6602's measured gear_small position.
    target = {"x": -7.91016079905615e-05, "y": .2999985103004382, "z": 1.0249998347058105}
    capabilities = _recorded_target_capabilities(base_y)
    motions = [{"mode": "cartesian", "waypoints": [{"position": target}]}]
    original = deepcopy((capabilities, motions))
    result = check_outline_motion_capabilities(motion_requests=motions,
        static_capabilities=capabilities, frame="world", resource_jid="recovery-resource-4@localhost")
    assert result["feasibility_status"] == expected
    assert result["checks"][0]["checked_pose"] == target
    assert result["command_sent"] is False and result["execution_authorized"] is False
    assert (capabilities, motions) == original


@pytest.mark.parametrize("defect,expected", [(None, "FEASIBLE"), ("intermediate_xy", "INFEASIBLE"),
    ("z", "INFEASIBLE"), ("missing_frame", "NEEDS_CONTEXT"), ("conflicting_frame", "NEEDS_CONTEXT"),
    ("null", "NEEDS_CONTEXT"), ("missing", "NEEDS_CONTEXT"), ("nonfinite", "NEEDS_CONTEXT")])
def test_outline_cartesian_checks_each_calculated_target_and_preserves_missing_values(defect, expected):
    from cais_spade_llm.resources.robot.recovery_capability import check_outline_motion_capabilities

    capabilities = _recorded_target_capabilities(.5)
    poses = [{"position": {"x": 0., "y": .3, "z": 1.3}}, {"position": {"x": 0., "y": .3, "z": 1.1}}]
    frame = "world"
    if defect == "intermediate_xy":
        poses[0]["position"]["x"] = 1.
    elif defect == "z":
        poses[0]["position"]["z"] = 2.
    elif defect == "missing_frame":
        frame = None
    elif defect == "conflicting_frame":
        capabilities["gripper_reach"]["frame"] = "base"
    elif defect == "null":
        poses[0]["position"]["z"] = None
    elif defect == "missing":
        poses[0]["position"].pop("z")
    elif defect == "nonfinite":
        poses[0]["position"]["z"] = float("inf")
    result = check_outline_motion_capabilities(motion_requests=[{"mode": "cartesian", "waypoints": poses}],
        static_capabilities=capabilities, frame=frame, resource_jid="ur5e-3@localhost")
    assert result["feasibility_status"] == expected
    if defect == "intermediate_xy":
        assert result["checks"][0]["target_index"] == 0
    assert result["execution_authorized"] is False


@pytest.mark.parametrize("base_y,status", [(-.5, "rejected"), (.5, "ready")])
def test_owner_preview_checks_calculated_pickup_and_departure_before_native(base_y, status, monkeypatch):
    from cais_spade_llm.resources import resource_safety_preparation as preparation
    from cais_spade_llm.resources.robot import recovery_capability

    scene, snapshot, model, task = _custody_event_case(True)
    scene["physical_preview_contract_version"] = 4
    capabilities = _recorded_target_capabilities(base_y)
    scene["resource_models"]["ur5e-3"]["static_capabilities"] = deepcopy(capabilities)
    target = {"x": -7.91016079905615e-05, "y": .2999985103004382, "z": 1.2449998347058105,
              "qx": 0., "qy": 0., "qz": 0., "qw": 1.}
    monkeypatch.setattr(recovery_capability, "calculate_recovery_pick_target", lambda **_: {
        "status": "calculated", "part_name": task["part_name"], "target_pose": target, "approach_pose": target,
        "travel_z": 1.5, "source": {"kind": "calculated", "calculation": "recorded fixture geometry"}})
    evidence = {"resource_type": "robot", "source": {"kind": "recorded", "capture": "fixture", "revision": 1},
                "controller_config": {"move_group": {"frame_id": "world"}}}
    original = deepcopy((scene, snapshot, model, task, evidence))
    prepared = preparation.prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence, static_capabilities=capabilities)
    assert prepared["status"] == status, prepared
    assert (scene, snapshot, model, task, evidence) == original
    assert prepared["command_sent"] is False and prepared["execution_authorized"] is False
    if status == "rejected":
        assert prepared["physical_capability_rejected"] is True
        assert prepared["capability_result"]["feasibility_status"] == "INFEASIBLE"
        assert prepared["resource_evidence_fingerprint"] == recovery_validation_fingerprint(evidence)
        assert prepared["event_record_binding"] == outline_event_binding(task, snapshot, model, scene)
        assert not set(prepared) & {"geometry_evidence", "event_evidence", "physical_evidence"}
        from cais_spade_llm.recovery_framework import outline_gazebo_evidence, recovery_motion_preview

        def forbidden(**_):
            pytest.fail("Declared capability rejection must precede live or native preview calls")

        monkeypatch.setattr(recovery_motion_preview, "capture_preview_scene", forbidden)
        monkeypatch.setattr(recovery_motion_preview, "preview_recovery_motion", forbidden)
        result = asyncio.run(outline_gazebo_evidence.prepare_owner_event_evidence(
            SimpleNamespace(agent_name="ur5e-3", static_capabilities=capabilities, execution_mode="simulation"), task=task,
            recovery_snapshot=snapshot, recovery_des_model=model, physical_evidence=scene,
            resource_evidence=evidence, observations={}))
        assert result == prepared
    else:
        contract = prepared["physical_evidence"]["resource_models"]["ur5e-3"]["event_projection_contracts"][
            prepared["event_record"]["contract_id"]]
        audit = preparation.validate_outline_motion_capabilities(contract=contract,
            model=scene["resource_models"]["ur5e-3"], resource_jid=task["resource_jid"])
        assert audit["motion_requests"] == prepared["motion_requests"]
        assert len(audit["result"]["checks"]) == 3
        assert prepared["motion_requests"][1]["waypoints"][0]["position"]["z"] == 1.5


@pytest.mark.parametrize("defect", [None, "missing_output", "closed_gripper", "unsupported_operation"])
def test_pick_arrival_uses_declared_waypoints_for_new_event_identifiers(defect):
    from dataclasses import asdict
    from cais_spade_llm.resources.resource_safety_preparation import _outline_pick_arrival
    from cais_spade_llm.resources.robot.robot_task_recovery import robot_recovery_des_descriptor
    from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry

    snapshot = {"current_state": "idle", "current_location": "home", "held_part": None}
    model = robot_recovery_des_descriptor(resource_jid="robot@localhost", snapshot=snapshot,
        task_names=["pick_approach"])
    model["events"][0]["composition_contract"]["steps"] = json.loads(json.dumps(
        asdict(robot_task_registry()["pick_approach"].program)))["steps"]
    task = {"event_name": "new_approach_identifier", "resource_jid": "robot@localhost", "part_name": "part"}
    declarations = [{"event_name": "pick_approach", "parameter_bindings": {
        "part_name": "part", "origin_resource_location": "recorded_station"}}]
    approach = dict(x=.1, y=.2, z=1.5, qx=0., qy=0., qz=0., qw=1.)
    target = {"approach_pose": approach, "target_pose": {**approach, "z": 1.2}}
    owner = {"gripper_state": "open"}
    if defect == "missing_output":
        target.pop("target_pose")
    elif defect == "closed_gripper":
        owner["gripper_state"] = "closed"
    elif defect == "unsupported_operation":
        model["events"][0]["composition_contract"]["steps"].append({"id": "unsupported", "op": "unrecorded_action"})
    def calculate():
        return _outline_pick_arrival(task=task, declarations=declarations, model=model,
            target=target, snapshot=snapshot,
            resource_evidence={"resource_type": "robot", "execution_mode": "simulation"}, owner=owner)
    if defect is None:
        assert calculate() == [approach, {**approach, "z": 1.2}]
    else:
        with pytest.raises(ValueError):
            calculate()


@pytest.mark.parametrize("defect", ["missing_capabilities", "conflicting_capabilities", "missing_source"])
@pytest.mark.parametrize("version", [4, 5])
def test_version_four_preparation_reports_missing_or_conflicting_owner_declarations(defect, version):
    from cais_spade_llm.resources.resource_safety_preparation import prepare_outline_event_preview_request

    scene, snapshot, model, task = _event_case()
    scene["physical_preview_contract_version"] = version
    capabilities = _recorded_target_capabilities(.5)
    scene["resource_models"]["ur5e-3"]["static_capabilities"] = deepcopy(capabilities)
    evidence = {"source": {"kind": "recorded", "capture": "fixture", "revision": 1}}
    if defect == "missing_capabilities":
        scene["resource_models"]["ur5e-3"].pop("static_capabilities")
    elif defect == "conflicting_capabilities":
        capabilities["gripper_reach"]["max_xy_radius_m"] = 3.
    else:
        evidence.pop("source")
    result = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence, static_capabilities=capabilities)
    assert result["status"] == "NEEDS_CONTEXT", result
    assert result["missing_facts"]
    assert result["command_sent"] is False and result["execution_authorized"] is False


@pytest.mark.parametrize("supported_x", [.001, 2.])
@pytest.mark.parametrize("version", [4, 5])
def test_departure_rechecks_actual_supported_endpoint_and_preserves_peer_scene(supported_x, monkeypatch, version):
    from cais_spade_llm.resources import resource_safety_preparation as preparation
    from cais_spade_llm.resources.robot import recovery_capability

    scene, snapshot, model, task = _custody_event_case(False)
    scene["physical_preview_contract_version"] = version
    scene["resource_models"]["ur5e-3"]["static_capabilities"] = _recorded_target_capabilities(.5)
    target = {"x": 0., "y": .3, "z": 1.2, "qx": 0., "qy": 0., "qz": 0., "qw": 1.}
    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **_: {
        "status": "calculated", "insert_pose": target})
    evidence = {"source": {"kind": "recorded", "capture": "fixture", "revision": 1},
        "state": {"_task_ctx": {"part_name": task["part_name"], "travel_z": 1.5}},
        "controller_config": {"move_group": {"frame_id": "world"}}}
    prepared = preparation.prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence)
    assert prepared["status"] == "ready", prepared
    progress = {"physical_evidence": deepcopy(scene)}
    progress["physical_evidence"]["snapshot"]["resources"]["ur5e-3"]["current_pose"] = [
        supported_x, .3, 1.2, 0., 0., 0., 1.]
    original = deepcopy((prepared, progress))
    if supported_x == 2.:
        with pytest.raises(ValueError, match="unsupported departure target"):
            preparation.prepare_outline_event_departure(prepared_preview=prepared, progress=progress)
    else:
        updated = preparation.prepare_outline_event_departure(prepared_preview=prepared, progress=progress)
        record = updated["event_record"]
        contract = updated["physical_evidence"]["resource_models"]["ur5e-3"]["event_projection_contracts"][record["contract_id"]]
        audit = preparation.validate_outline_motion_capabilities(contract=contract,
            model=scene["resource_models"]["ur5e-3"], resource_jid=task["resource_jid"])
        assert audit["result"]["checks"][-1]["checked_pose"] == {"x": supported_x, "y": .3, "z": 1.5}
        assert updated["physical_evidence"]["snapshot"]["resources"]["ur5e-4"] == scene["snapshot"]["resources"]["ur5e-4"]
        assert updated["motion_requests"][0] == prepared["motion_requests"][0]
    assert (prepared, progress) == original


@pytest.mark.parametrize("defect", [None, "missing_audit", "capabilities", "audit_result", "audit_targets", "interval_targets"])
@pytest.mark.parametrize("version", [4, 5])
def test_version_four_cached_event_rechecks_capabilities_and_exact_motion_requests(defect, version):
    from cais_spade_llm.resources.resource_safety_preparation import prepare_outline_event_preview_request, finalize_outline_event_preview

    scene, snapshot, model, task = _event_case()
    scene["physical_preview_contract_version"] = version
    scene["resource_models"]["ur5e-3"]["static_capabilities"] = _recorded_target_capabilities(-.5)
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene,
        resource_evidence={"source": {"kind": "recorded", "capture": "fixture", "revision": 1}})
    response = _issued_native_preview(prepared)
    completed = finalize_outline_event_preview(prepared_preview=prepared, motion_results=[response])
    assert completed["status"] == "prepared", completed
    evidence, record = prepared["physical_evidence"], completed["event_evidence"][0]
    contract = evidence["resource_models"]["ur5e-3"]["event_projection_contracts"][record["contract_id"]]
    audit = contract["motion_capability_evidence"]
    assert audit["result"]["checks"] == []  # Named joints are not Cartesian envelope claims.
    if defect == "missing_audit":
        contract.pop("motion_capability_evidence")
    elif defect == "capabilities":
        audit["static_capabilities"]["gripper_reach"]["max_xy_radius_m"] = 3.
    elif defect == "audit_result":
        audit["result"]["checks"] = [{"feasibility_status": "FEASIBLE"}]
    elif defect == "audit_targets":
        audit["motion_requests"][0]["joint_target"]["position"] = [.1]
    elif defect == "interval_targets":
        record["intervals"][0]["motion_request"]["joint_target"]["position"] = [.1]
    record["contract_fingerprint"] = recovery_validation_fingerprint(contract)
    result = project_recovery_outline_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        evidence=evidence, event_evidence=[record])
    assert result["status"] == ("projected" if defect is None else "needs_context"), result


@pytest.mark.parametrize("defect", [None, "missing_contacts", "moving", "edge_com", "dynamic_support", "slope", "no_policy"])
def test_gravity_supported_rest_requires_recorded_geometry_and_explicit_conditional_model(defect):
    from cais_spade_llm.resources.recovery_geometry import gravity_supported_part

    scene, _, _, _ = _custody_event_case(False)
    physics, policy = scene["physics_evidence"], scene["part_support_policy"]
    if defect == "missing_contacts":
        physics["contacts"] = []
    elif defect == "moving":
        physics["models"]["square"]["links"]["link"]["linear_velocity"] = [.1, 0., 0.]
    elif defect == "edge_com":
        physics["models"]["square"]["links"]["link"]["center_of_mass"] = [.05, 0., 0.]
    elif defect == "dynamic_support":
        physics["models"]["table"]["static"] = False
    elif defect == "slope":
        physics["contacts"][0]["normals"] = [[.1, 0., .9]] * 4
    elif defect == "no_policy":
        policy = {}
    result = gravity_supported_part(model_name="square", physics=physics, policy=policy, owners_idle=True)
    assert result["status"] == ("supported" if defect is None else "NEEDS_CONTEXT"), result
    assert result["physical_execution_verified"] is False
    if defect is None:
        assert result["model_assumptions"]


@pytest.mark.parametrize("defect", [None, "enabled", "missing_enabled", "nonzero_velocity", "unknown_engine",
    "model_joint", "attachment", "unknown_attachments", "dynamic_contact", "owner_active"])
def test_disabled_body_interval_evidence_requires_exact_recorded_engine_state(defect):
    from cais_spade_llm.resources.recovery_geometry import disabled_part_observation

    physics = _supported_physics()
    physics.update(physics_engine="ode", attachment_complete=True, attachments=[])
    physics["models"]["square"]["joint_count"] = 0
    link = physics["models"]["square"]["links"]["link"]
    link["enabled"] = False
    if defect == "enabled":
        link["enabled"] = True
    elif defect == "missing_enabled":
        link.pop("enabled")
    elif defect == "nonzero_velocity":
        link["angular_velocity"] = [0., 0., 1e-12]
    elif defect == "unknown_engine":
        physics.pop("physics_engine")
    elif defect == "model_joint":
        physics["models"]["square"]["joint_count"] = 1
    elif defect == "attachment":
        physics["attachments"] = [{"model1": "robot", "model2": "square"}]
    elif defect == "unknown_attachments":
        physics.pop("attachment_complete")
    elif defect == "dynamic_contact":
        physics["models"]["table"]["static"] = False
    original = deepcopy(physics)
    result = disabled_part_observation(model_name="square", physics=physics, owners_idle=defect != "owner_active")
    assert result["status"] == ("supported" if defect is None else "NEEDS_CONTEXT"), result
    assert result["physical_execution_verified"] is False
    assert physics == original
    if defect is None:
        assert result["enabled"] is False
        assert "wake" in result["model_assumptions"][-1]


def test_disabled_body_capture_preserves_engine_provenance_without_live_calls():
    from cais_spade_llm.recovery_framework.recovery_observations import _stationary_part_observation

    physics = _supported_physics()
    physics.update(physics_engine="ode", attachment_complete=True, attachments=[])
    physics["models"]["square"]["joint_count"] = 0
    physics["models"]["square"]["links"]["link"].update(enabled=False, linear_speed=0., angular_speed=0.)
    part = {"name": "square", "pose": deepcopy(physics["models"]["square"]["pose"])}
    evidence = {"snapshot": {"resources": {"table": {"held_part": None}},
        "parts": {"square": {"contained_by": None}}},
        "resources": {"table": {"model_static": True, "idle": True}}}
    result = _stationary_part_observation("square", part, physics, evidence, {})
    assert result["status"] == "supported"
    assert result["kind"] == "Gazebo ODE disabled body"
    assert result["current_pose"] == part["pose"]
    assert result["source"] == {"service": "GETRECOVERYSTATE", "instance_id": "physics", "simulation_time": 2.}
    physics["models"]["square"]["links"]["link"]["enabled"] = True
    assert _stationary_part_observation("square", part, physics, evidence, {})["status"] == "NEEDS_CONTEXT"


def _fixed_attachment_physics():
    physics = _supported_physics()
    parent_pose = [1., 2., 3., 0., 0., 1., 0.]
    part_pose = [.8, 1.9, 3.3, 0., 0., 1., 0.]
    physics.update(attachment_complete=True, attachment_evidence_version=1, attachment_revision=4, contacts=[])
    physics['models']['table'].update(pose=parent_pose[:], joint_count=1)
    physics['models']['table']['links']['link'].update(pose=parent_pose[:], linear_velocity=[0.,0.,0.],
        angular_velocity=[0.,0.,0.], linear_speed=0., angular_speed=0.)
    physics['models']['square'].update(pose=part_pose[:], joint_count=0)
    physics['models']['square']['links']['link'].update(pose=part_pose[:], enabled=True,
        linear_speed=0., angular_speed=0.)
    physics['attachments'] = [{'model1':'table', 'link1':'link', 'model2':'square', 'link2':'link',
        'joint_evidence':{'status':'observed', 'joint_name':'observed_joint',
            'joint_scoped_name':'table::observed_joint', 'fixed_joint':True,
            'parent':{'model':'table', 'link':'link', 'scoped_name':'table::link', 'pose':parent_pose[:]},
            'child':{'model':'square', 'link':'link', 'scoped_name':'square::link', 'pose':part_pose[:]},
            'parent_to_child':[.2,.1,.3,0.,0.,0.,1.]}}]
    return physics


@pytest.mark.parametrize('enabled', [False, True])
def test_fixed_attachment_to_static_endpoint_uses_live_constraint_not_sampled_rest(enabled):
    from cais_spade_llm.resources.recovery_geometry import fixed_attached_part_observation

    physics = _fixed_attachment_physics()
    physics['models']['square']['links']['link']['enabled'] = enabled
    before = deepcopy(physics)
    result = fixed_attached_part_observation(model_name='square',
        current_pose=physics['models']['square']['pose'], physics=physics, owners_idle=True)
    assert result['status'] == 'supported', result
    assert result['source']['attachment_revision'] == 4
    assert result['source']['physics_fingerprint'] == recovery_validation_fingerprint(physics)
    assert result['static_endpoint']['model'] == 'table'
    assert result['static_endpoint_to_part'] == pytest.approx([.2,.1,.3,0.,0.,0.,1.])
    assert result['physical_execution_verified'] is False
    assert result['model_assumptions'] and physics == before


@pytest.mark.parametrize('topology', ['reversed', 'static_hub'])
def test_fixed_attachment_preserves_endpoint_direction_and_static_hub_constraints(topology):
    from cais_spade_llm.resources.recovery_geometry import fixed_attached_part_observation

    physics = _fixed_attachment_physics()
    row = physics['attachments'][0]
    native = row['joint_evidence']
    if topology == 'reversed':
        row['model1'], row['model2'] = row['model2'], row['model1']
        native['parent'], native['child'] = native['child'], native['parent']
        native['joint_scoped_name'] = 'square::observed_joint'
        native['parent_to_child'] = [-.2,-.1,-.3,0.,0.,0.,1.]
        physics['models']['square']['joint_count'] = 1
        physics['models']['table']['joint_count'] = 0
    else:
        physics['models']['another_part'] = deepcopy(physics['models']['square'])
        sibling = deepcopy(row)
        sibling['model2'] = 'another_part'
        sibling['joint_evidence']['child'].update(model='another_part', scoped_name='another_part::link')
        sibling['joint_evidence'].update(joint_name='another_joint', joint_scoped_name='table::another_joint')
        physics['attachments'].append(sibling)
        physics['models']['table']['joint_count'] = 2
    result = fixed_attached_part_observation(model_name='square',
        current_pose=physics['models']['square']['pose'], physics=physics, owners_idle=True)
    assert result['status'] == 'supported', result
    assert result['attachment'] == row
    assert result['static_endpoint_to_part'] == pytest.approx([.2,.1,.3,0.,0.,0.,1.])


@pytest.mark.parametrize('defect', ['legacy', 'incomplete', 'dead_joint', 'movable_joint',
    'joint_identity', 'link_identity', 'endpoint_pose', 'relative_pose', 'part_pose',
    'moving_parent', 'dynamic_parent', 'second_attachment', 'self_attachment', 'missing_link',
    'extra_part_joint', 'owner_active', 'boolean_pose'])
def test_fixed_attachment_missing_or_conflicting_evidence_does_not_fall_back_to_zero_velocity(defect):
    from cais_spade_llm.resources.recovery_geometry import fixed_attached_part_observation

    physics = _fixed_attachment_physics()
    row = physics['attachments'][0]
    native = row['joint_evidence']
    pose = deepcopy(physics['models']['square']['pose'])
    if defect == 'legacy': physics.pop('attachment_evidence_version'); row.pop('joint_evidence')
    elif defect == 'incomplete': physics['attachment_complete'] = False
    elif defect == 'dead_joint': native['status'] = 'NEEDS_CONTEXT'
    elif defect == 'movable_joint': native['fixed_joint'] = False
    elif defect == 'joint_identity': native['joint_scoped_name'] = 'another::observed_joint'
    elif defect == 'link_identity': native['child']['link'] = 'different'
    elif defect == 'endpoint_pose': native['child']['pose'][0] += .01
    elif defect == 'relative_pose': native['parent_to_child'][0] += .01
    elif defect == 'part_pose': pose[0] += .01
    elif defect == 'moving_parent': physics['models']['table']['links']['link']['linear_velocity'][0] = .01
    elif defect == 'dynamic_parent': physics['models']['table']['static'] = False
    elif defect == 'second_attachment': physics['attachments'].append(deepcopy(row))
    elif defect == 'self_attachment': row['model1'] = row['model2']
    elif defect == 'missing_link': physics['models']['table']['links'] = {}
    elif defect == 'extra_part_joint': physics['models']['square']['joint_count'] = 1
    elif defect == 'boolean_pose': pose[3] = False
    before = deepcopy(physics)
    result = fixed_attached_part_observation(model_name='square', current_pose=pose,
        physics=physics, owners_idle=defect != 'owner_active')
    assert result['status'] == 'NEEDS_CONTEXT', result
    assert result['physical_execution_verified'] is False and physics == before


@pytest.mark.parametrize('defect', [None, 'held', 'source', 'revision', 'pose'])
def test_fixed_attachment_capture_preserves_same_snapshot_and_never_hides_robot_custody(defect):
    from cais_spade_llm.recovery_framework.recovery_observations import _stationary_part_observation

    physics = _fixed_attachment_physics()
    part = {'name':'square', 'pose':deepcopy(physics['models']['square']['pose']),
        'instance_id':physics['instance_id'], 'simulation_time':physics['simulation_time']}
    evidence = {'snapshot':{'resources':{'table':{'held_part':None}}, 'parts':{'square':{'contained_by':None}}},
        'resources':{'table':{'model_static':True, 'idle':True}}}
    if defect == 'held': evidence['snapshot']['resources']['table']['held_part'] = 'square'
    elif defect == 'source': part['instance_id'] = 'another'
    elif defect == 'revision': physics['attachment_revision'] = None
    elif defect == 'pose': part['pose'][0] += .1
    original = deepcopy((physics, part, evidence))
    result = _stationary_part_observation('square', part, physics, evidence, {})
    assert result['status'] == ('supported' if defect is None else 'NEEDS_CONTEXT'), result
    if defect is None:
        assert result['kind'] == 'recorded fixed attachment to static endpoint'
        assert result['source']['physics_fingerprint'] == recovery_validation_fingerprint(physics)
    assert (physics, part, evidence) == original


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_free_part_model_uses_supported_owner_hold_without_relabelling_motion(disabled, active):
    from cais_spade_llm.recovery_framework.recovery_observations import _stationary_part_observation

    _, rid, state, model, physical = _commanded_hold_case()
    physical["controller_hold_observation"]["has_active_goal"] = active
    physics = _supported_physics()
    physics.update(physics_engine="ode", attachment_complete=True, attachments=[])
    physics["models"]["square"]["joint_count"] = 0
    physics["models"]["square"]["links"]["link"].update(
        enabled=not disabled, linear_speed=0., angular_speed=0.)
    part = {"name": "square", "pose": deepcopy(physics["models"]["square"]["pose"])}
    evidence = {"snapshot": {"resources": {rid: state}, "parts": {"square": {"contained_by": None}}},
        "resources": {rid: physical}, "resource_models": {rid: model}}
    scene, _, _, _ = _custody_event_case(False)
    original = deepcopy(evidence)
    result = _stationary_part_observation("square", part, physics, evidence,
                                          {"part_support": scene["part_support_policy"]})
    assert result["status"] == ("NEEDS_CONTEXT" if active else "supported"), result
    assert evidence == original
    assert physical["idle"] is False
    if not active:
        assert result["owner_evidence"][rid]["observed_idle"] is False
        assert result["owner_evidence"][rid]["physical_execution_verified"] is False


def test_projected_deposit_uses_exact_collision_shapes_and_declared_contact_margin():
    from cais_spade_llm.resources.recovery_geometry import projected_part_support

    scene, _, _, task = _custody_event_case(False)
    original = deepcopy(scene)
    result = projected_part_support(part_name=task["part_name"], state=scene["snapshot"], evidence=scene)
    assert result["status"] == "supported", result
    assert result["source"]["kind"] == "projected"
    assert scene == original
    scene["snapshot"]["parts"][task["part_name"]]["current_pose"][2] += .02
    result = projected_part_support(part_name=task["part_name"], state=scene["snapshot"], evidence=scene)
    assert result["status"] == "NEEDS_CONTEXT"


def test_projected_support_keeps_original_pose_through_complete_json_roundtrip():
    from cais_spade_llm.resources.recovery_geometry import projected_part_support

    scene, _, _, task = _custody_event_case(False)
    part = scene["snapshot"]["parts"][task["part_name"]]
    part["current_pose"] = [-.031999999258299206, .014484720440157564, 0., 0., 0., 0., 1.]
    original = deepcopy(scene)
    result = projected_part_support(part_name=task["part_name"], state=scene["snapshot"], evidence=scene)
    assert result["status"] == "supported", result
    saved = json.loads(json.dumps(result, allow_nan=False))
    assert saved == result
    assert saved["current_pose"] == part["current_pose"]
    assert saved["source"]["part_pose"] == part["current_pose"]
    assert saved["physics_evidence"]["models"][part["model_name"]]["pose"] == part["current_pose"]
    replay_scene = json.loads(json.dumps(scene, allow_nan=False))
    assert projected_part_support(part_name=task["part_name"],
        state=replay_scene["snapshot"], evidence=replay_scene) == saved
    assert scene == original


def test_projected_support_rejects_nonrecordable_pose_without_rounding():
    from fractions import Fraction
    from cais_spade_llm.resources.recovery_geometry import projected_part_support

    scene, _, _, task = _custody_event_case(False)
    scene["snapshot"]["parts"][task["part_name"]]["current_pose"][0] = Fraction(1, 3)
    original = deepcopy(scene)
    result = projected_part_support(part_name=task["part_name"], state=scene["snapshot"], evidence=scene)
    assert result["status"] == "NEEDS_CONTEXT"
    assert "finite number" in result["reason"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert scene == original


@pytest.mark.parametrize("tilt", [0., .00002, .2])
def test_support_place_target_derives_contact_without_changing_observations(tilt):
    import math
    from cais_spade_llm.resources.robot.recovery_capability import _support_place_pose

    scene, _, _, task = _custody_event_case(False)
    scene["physics_evidence"]["models"]["square"]["joint_count"] = 0
    part, destination = task["part_name"], "Buffer For Machined parts"
    scene["snapshot"]["parts"][part]["current_pose"] = [0., 0., .2, math.sin(tilt / 2), 0., 0., math.cos(tilt / 2)]
    scene["resources"][destination]["geometry_source"] = {"model": "table"}
    target = {"target_origin_pose": {"x": .1, "y": .2, "z": -.05}, "support_surface_height": -.05}
    original = deepcopy((scene, target))
    result = _support_place_pose(part=part, destination=destination, target=target, physical_evidence=scene)
    assert result["support_evidence"]["status"] == "supported"
    assert result["part_pose"][:3] == pytest.approx([.1, .2, 0.], abs=1e-12)
    assert result["part_pose"][3:] == pytest.approx([0., 0., 0., 1.], abs=1e-12)
    assert result["support_collision_id"] == "table::collision"
    assert result["part_collision_id"] == "square::collision"
    assert result["source"]["kind"] == "calculated"
    assert result["source"]["starting_part_pose"] == scene["snapshot"]["parts"][part]["current_pose"]
    assert result["settling_modeled"] is False and result["command_sent"] is False
    assert "support_surface_height" not in result and "support_surface_height_source" not in result
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert (scene, target) == original


@pytest.mark.parametrize("missing", ["height", "shape", "gravity", "ambiguous", "footprint", "static", "center", "joint_count", "joint"])
def test_support_place_target_rejects_unsupported_or_unstable_contact(missing):
    import math
    from cais_spade_llm.resources.robot.recovery_capability import _support_place_pose

    scene, _, _, task = _custody_event_case(False)
    scene["physics_evidence"]["models"]["square"]["joint_count"] = 0
    part, destination = task["part_name"], "Buffer For Machined parts"
    scene["resources"][destination]["geometry_source"] = {"model": "table"}
    target = {"target_origin_pose": {"x": .1, "y": .2, "z": -.05}, "support_surface_height": -.05}
    if missing == "height":
        target["support_surface_height"] = None
    elif missing == "shape":
        scene["physics_evidence"]["models"]["square"]["links"]["link"]["collisions"][0]["geometry"] = {"type": "mesh"}
    elif missing == "gravity":
        scene["physics_evidence"]["gravity"] = [1., 0., -9.81]
    elif missing == "ambiguous":
        scene["snapshot"]["parts"][part]["current_pose"][3:] = [math.sin(math.pi / 8), 0., 0., math.cos(math.pi / 8)]
    elif missing == "footprint":
        target["target_origin_pose"]["x"] = .49
    elif missing == "static":
        scene["physics_evidence"]["models"]["table"]["static"] = False
    elif missing == "center":
        scene["physics_evidence"]["models"]["square"]["links"]["link"]["center_of_mass"] = [.2, 0., 0.]
    elif missing == "joint_count":
        scene["physics_evidence"]["models"]["square"].pop("joint_count")
    elif missing == "joint":
        scene["physics_evidence"]["models"]["square"]["joint_count"] = 1
    before = deepcopy((scene, target))
    with pytest.raises(ValueError):
        _support_place_pose(part=part, destination=destination, target=target, physical_evidence=scene)
    assert (scene, target) == before


@pytest.mark.parametrize("defect", [None, "multiple", "conflicting_height", "null_height",
                                  "unsupported_geometry", "missing_geometry", "missing_receiver"])
def test_support_place_height_derivation_uses_one_exact_captured_face(defect):
    from cais_spade_llm.resources.robot.recovery_capability import _support_place_pose

    scene, _, _, task = _custody_event_case(False)
    part, destination = task["part_name"], "Exit"
    scene["physics_evidence"]["models"]["square"]["joint_count"] = 0
    # Exact Exit collision dimensions/poses from capture_f923020bf9ce496eb443826291f8f2c2.
    collisions = [
        {"id": "Exit::link::surface", "pose": [.5, .58, 1.04, 0., 0., 0., 1.],
         "geometry": {"type": "box", "size": [.4, .4, .05]}},
        {"id": "Exit::link::rear_edge", "pose": [.5, .77, 1.085, 0., 0., 0., 1.],
         "geometry": {"type": "box", "size": [.4, .02, .04]}},
        {"id": "Exit::link::side_edge_1", "pose": [.31, .58, 1.085, 0., 0., 0., 1.],
         "geometry": {"type": "box", "size": [.02, .38, .04]}},
        {"id": "Exit::link::side_edge_2", "pose": [.69, .58, 1.085, 0., 0., 0., 1.],
         "geometry": {"type": "box", "size": [.02, .38, .04]}},
    ]
    for collision in collisions:
        collision["contact_min_depth"] = 0.
    scene["resources"][destination] = {"geometry_source": {"model": "Exit", "service": "GETRECOVERYSTATE"}}
    scene["physics_evidence"]["models"][destination] = {"static": True, "links": {"link": {"collisions": collisions}}}
    target = {"target_origin_pose": {"x": .5, "y": .58, "z": 1.075}}
    if defect == "multiple":
        collisions.append({**deepcopy(collisions[0]), "id": "other::surface", "pose": [.5, .58, 1.2, 0., 0., 0., 1.]})
    elif defect == "conflicting_height":
        target["support_surface_height"] = 1.075
    elif defect == "null_height":
        target["support_surface_height"] = None
    elif defect == "unsupported_geometry":
        collisions[-1]["geometry"] = {"type": "mesh"}
    elif defect == "missing_geometry":
        collisions[-1].pop("geometry")
    elif defect == "missing_receiver":
        scene["physics_evidence"]["models"].pop(destination)
    before = deepcopy((scene, target))
    if defect is not None:
        with pytest.raises((ValueError, KeyError)):
            _support_place_pose(part=part, destination=destination, target=target, physical_evidence=scene)
    else:
        result = _support_place_pose(part=part, destination=destination, target=target, physical_evidence=scene)
        assert result["support_evidence"]["status"] == "supported"
        assert result["support_surface_height"] == 1.065
        assert result["support_collision_id"] == "Exit::link::surface"
        assert result["support_surface_height_source"] == {
            "kind": "calculated", "calculation": "uniquely covering recorded horizontal box face",
            "collision_id": "Exit::link::surface", "geometry_source": scene["resources"][destination]["geometry_source"],
            "receiver_model_fingerprint": recovery_validation_fingerprint(scene["physics_evidence"]["models"][destination]),
            "destination_fingerprint": recovery_validation_fingerprint(target)}
        explicit = _support_place_pose(part=part, destination=destination,
            target={**deepcopy(target), "support_surface_height": 1.065}, physical_evidence=scene)
        assert result["part_pose"] == explicit["part_pose"]
        assert result["support_evidence"] == explicit["support_evidence"]
        assert result["command_sent"] is False and result["settling_modeled"] is False
    assert (scene, target) == before


def test_projected_mesh_support_uses_saved_vertices_and_never_loads_live_assets(monkeypatch):
    from cais_spade_llm.resources.recovery_geometry import projected_part_support

    scene, _, _, task = _custody_event_case(False)
    collision = scene["physics_evidence"]["models"]["square"]["links"]["link"]["collisions"][0]
    collision["geometry"] = {"type": "mesh", "uri": "file:///recorded/part.stl", "scale": [.001] * 3}
    vertices = [[x, y, z] for x in (-50., 50.) for y in (-50., 50.) for z in (-50., 50.)]
    scene["physics_meshes"] = {collision["geometry"]["uri"]: {"uri": collision["geometry"]["uri"],
        "vertices": vertices, "vertex_count": len(vertices), "source_sha256": "captured_hash",
        "source_kind": "recorded_collision_mesh"}}
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("Offline geometry must not read current files"))
    result = projected_part_support(part_name=task["part_name"], state=scene["snapshot"], evidence=scene)
    assert result["status"] == "supported", result
    scene["physics_meshes"] = {}
    result = projected_part_support(part_name=task["part_name"], state=scene["snapshot"], evidence=scene)
    assert result["status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("depths,offset,supported", [
    ((.01, .002, .005), -.001, True),
    ((.01, .002, .005), -.003, False),
    ((.01, .005, .002), -.003, False),
    ((0., .005, .002), -.0001, False),
    ((.01, .005, .002), .0001, False),
])
def test_projected_contact_uses_recorded_minimum_and_never_bridges_air_gap(depths, offset, supported):
    from cais_spade_llm.resources.recovery_geometry import projected_part_support

    scene, _, _, task = _custody_event_case(False)
    physics = scene["physics_evidence"]
    physics["models"]["square"]["links"]["link"]["collisions"][0]["contact_min_depth"] = depths[0]
    physics["models"]["table"]["links"]["link"]["collisions"][0]["contact_min_depth"] = depths[1]
    physics["physics_configuration"]["sdf"] = (
        f"<physics><ode><constraints><contact_surface_layer>{depths[2]}</contact_surface_layer>"
        "</constraints></ode></physics>")
    scene["snapshot"]["parts"][task["part_name"]]["current_pose"][2] += offset
    original = deepcopy(scene)
    result = projected_part_support(part_name=task["part_name"], state=scene["snapshot"], evidence=scene)
    assert result["status"] == ("supported" if supported else "NEEDS_CONTEXT"), result
    assert scene == original
    if supported:
        assert result["source"]["contact_margins"] == [{"collision1": "square::collision", "collision2": "table::collision",
            "contact_min_depth1": depths[0], "contact_min_depth2": depths[1],
            "contact_surface_layer": depths[2], "penetration_allowance": min(depths)}]
        assert result["physics_evidence"]["contacts"][-1]["depths"] == pytest.approx([-offset] * 4)


@pytest.mark.parametrize("missing", ["part", "support", "world", "conflict", "invalid"])
def test_projected_contact_never_defaults_missing_physics_values(missing):
    from cais_spade_llm.resources.recovery_geometry import projected_part_support

    scene, _, _, task = _custody_event_case(False)
    physics = scene["physics_evidence"]
    if missing in {"part", "support"}:
        name = "square" if missing == "part" else "table"
        physics["models"][name]["links"]["link"]["collisions"][0].pop("contact_min_depth")
    elif missing == "world":
        physics.pop("physics_configuration")
    elif missing == "conflict":
        physics["contact_surface_layer"] = .001
    else:
        physics["models"]["square"]["links"]["link"]["collisions"][0]["contact_min_depth"] = True
    result = projected_part_support(part_name=task["part_name"], state=scene["snapshot"], evidence=scene)
    assert result["status"] == "NEEDS_CONTEXT" and "contact_" in result["reason"], result


def test_capture_freezes_native_mesh_uri_with_complete_vertices_and_source_hash():
    from cais_spade_llm.recovery_framework.recovery_observations import _freeze_collision_mesh

    mesh = _freeze_collision_mesh("model://gear_small/meshes/Gear_Small.STL")
    assert mesh["vertex_count"] == len(mesh["vertices"]) > 0
    assert mesh["triangle_count"] == len(mesh["triangles"]) > 0
    assert all(len(triangle) == 3 and all(0 <= index < mesh["vertex_count"] for index in triangle)
               for triangle in mesh["triangles"])
    assert len(mesh["source_sha256"]) == 64
    assert mesh["source_kind"] == "recorded_collision_mesh"
    with pytest.raises(ValueError, match="workspace asset"):
        _freeze_collision_mesh("file:///tmp/not-a-recorded-collision.stl")


def test_supported_buffer_occupancy_uses_existing_runtime_product_state_meaning(monkeypatch):
    from cais_spade_llm.resources import resource_safety_preparation as preparation
    from cais_spade_llm.resources.robot import recovery_capability

    scene, snapshot, model, task = _custody_event_case(False)
    part, rid, receiver = task["part_name"], "ur5e-3", "Buffer For Machined parts"
    scene["resources"][receiver].update(idle=True, geometry_source={"model": "table"},
        stationary_contract={"kind": "static_body_and_idle_containment"})
    scene["environment_models"] = {rid: {"state_variables": {"held_part": {"reference": "part_name"}}},
        receiver: {"state_variables": {"zone_4_part": {"reference": "part_name"}},
                   "assignments": {"zones": [{"zone": 4, "pose": [0., 0., 0., 0., 0., 0.]}]}}}
    scene["environment_valuation"] = {rid: {"held_part": part}, receiver: {"zone_4_part": None}}
    scene["product_name"] = "assembly_board-v1"
    target = {"x": 0., "y": 0., "z": 0., "qx": 0., "qy": 0., "qz": 0., "qw": 1.}
    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **_: {
        "status": "calculated", "insert_pose": target})
    prepared = preparation.prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene,
        resource_evidence={"state": {"_task_ctx": {"part_name": part, "travel_z": .2}}})
    prepared["custody_operation"]["destination_location"] = receiver
    responses = [_preview_response(prepared["motion_requests"][0])]
    progress = preparation.outline_event_preview_progress(prepared_preview=prepared, motion_results=responses)
    prepared = preparation.prepare_outline_event_departure(prepared_preview=prepared, progress=progress)
    responses.append(_preview_response(prepared["motion_requests"][1], target=.2))
    result = preparation.finalize_outline_event_preview(prepared_preview=prepared,
        motion_results=responses)
    assert result["status"] == "prepared", result
    projection = result["geometry_evidence"]
    assert projection["part_updates"]["part_state"] == "ready"
    assert projection["successor_evidence"]["environment_valuation"][receiver]["zone_4_part"] == part
    assert projection["product_custody_evidence"]["kind"] == "projected"
    assert projection["product_completion_evidence"] is None
    assert projection["deferred_part_fields"] == []
    assert scene["environment_valuation"][receiver]["zone_4_part"] is None


def test_product_completion_requires_declared_effect_target_and_prior_requirements():
    from cais_spade_llm.resources.recovery_geometry import _owner_product_completion

    scene, _, _, task = _custody_event_case(False)
    part, rid, target = task["part_name"], "ur5e-3", "Square_Plate/Square_Hole_1"
    scene["snapshot"]["resources"][rid]["held_part"] = None
    scene["geometry"]["parts"][part]["target"] = target
    event = {"event_name": "place_insert", "function_name": "place_insert", "event_id": 3,
        "updates": {"part_location": {"set": "assembly_board-v1"}},
        "parameter_bindings": {"resource_id": {"equals": rid}, "destination_location": {"equals": "assembly_board-v1"},
            "part_name": {"reference": "part_name"}, "target": {"type": "string"}},
        "product_effects": {"state": "assembled", "target": {"set_from_param": "target"},
            "processCompleted": [{"process": "assembly", "target": {"set_from_param": "target"}}]}}
    scene["environment_models"] = {rid: {"events": [event]}}
    scene["requirements"] = {part: [{"state": "assembled", "target": target}]}
    scene["product_name"] = "assembly_board-v1"
    contract = {"owner_declarations": [{"event_name": "place_insert"}], "placement_target": {
        "status": "calculated", "destination_location": "assembly_board-v1", "target": target,
        "target_reference": {"surface_role": "assembly_slot", "target_point": "inserted_part_origin"},
        "target_origin_pose": {"x": 0., "y": 0., "z": 0.}}}
    result = _owner_product_completion(task=task, state=scene["snapshot"], geometry=scene["geometry"],
        evidence=scene, contract=contract, rid=rid, support={part: {"status": "supported"}})
    assert result["kind"] == "predicted"
    assert result["product_effects"][part]["processCompleted"] == [{"process": "assembly", "target": target}]
    scene["requirements"][part].insert(0, {"process": "trim", "result": "square"})
    with pytest.raises(ValueError, match="Preceding product requirements"):
        _owner_product_completion(task=task, state=scene["snapshot"], geometry=scene["geometry"],
            evidence=scene, contract=contract, rid=rid, support={part: {"status": "supported"}})
    contract["placement_target"]["target"] = "another target"
    assert _owner_product_completion(task=task, state=scene["snapshot"], geometry=scene["geometry"],
        evidence=scene, contract=contract, rid=rid, support={part: {"status": "supported"}}) is None


def test_robot_nominal_contract_preserves_required_arguments_and_optional_null_defaults():
    from cais_spade_llm.resources.robot.robot_task_recovery import robot_recovery_des_descriptor

    model = robot_recovery_des_descriptor(resource_jid="ur5e-3@localhost",
        snapshot={"current_state": "idle", "current_location": "home", "held_part": None},
        task_names=["place_insert"])
    params = model["events"][0]["composition_contract"]["params"]
    assert params["destination_location"] == {"type": "string", "required": True}
    assert params["part_name"] == {"type": "string", "required": True}
    assert params["product_geometry"] == {"type": "object", "required": False, "default": None}


@pytest.mark.parametrize("grasp", [False, True])
@pytest.mark.parametrize("clock_version", [None, 1])
def test_declared_custody_preview_preserves_part_obligations_and_complete_bounds(grasp, clock_version, monkeypatch):
    from cais_spade_llm.resources import resource_safety_preparation as preparation
    from cais_spade_llm.resources.robot import recovery_capability

    scene, snapshot, model, task = _custody_event_case(grasp)
    if clock_version is not None:
        scene["owner_projection_clock_version"] = clock_version
    if grasp:
        support_scene, rid, _ = _support_departure_case()
        for field in ("stationary_part_evidence", "scene_object_bindings", "support_collision_object_bindings"):
            scene[field] = deepcopy(support_scene[field])
        for name, part in scene["snapshot"]["parts"].items():
            scene["stationary_part_evidence"].setdefault(name, {
                "status": "supported", "current_pose": deepcopy(part["current_pose"])})
        scene["resource_models"][rid]["configuration"]["joints"][0]["axis"] = [0., 0., 1.]
        model["events"][0]["composition_contract"] = {"steps": [
            {"id": "grasp", "op": "grasp_part", "params": {}},
            {"id": "lift", "op": "move_relative", "params": {"dx": 0., "dy": 0., "dz": .2}}]}
    target = {"x": 0, "y": 0, "z": 0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}
    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **_: {
        "status": "calculated", "insert_pose": target})
    monkeypatch.setattr(recovery_capability, "calculate_recovery_pick_target", lambda **_: {
        "status": "calculated", "part_name": task["part_name"], "target_pose": target, "approach_pose": target,
        "travel_z": .2, "source": {"kind": "calculated", "calculation": "recorded fixture geometry"}})
    prepared = preparation.prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene,
        resource_evidence={"state": {"_task_ctx": {"part_name": task["part_name"], "travel_z": .2}}})
    assert prepared["status"] == "ready", prepared
    # The receiver is a proposed part location; the owner station has no effect
    # on the physically established part location.
    if not grasp:
        prepared["custody_operation"]["destination_location"] = "Buffer For Machined parts"
    responses = [_preview_response(prepared["motion_requests"][0])]
    progress = preparation.outline_event_preview_progress(prepared_preview=prepared, motion_results=responses)
    prepared = preparation.prepare_outline_event_departure(prepared_preview=prepared, progress=progress)
    if grasp:
        from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure

        assert progress["physical_evidence"]["snapshot"]["resources"][rid]["held_part"] == task["part_name"]
        assert progress["physical_evidence"]["snapshot"]["parts"][task["part_name"]]["current_pose"][2] == 0
        support = prepare_support_departure(resource_id=rid,
            state=progress["physical_evidence"]["snapshot"], evidence=progress["physical_evidence"])
        assert support["status"] == "prepared", support
        request = deepcopy(prepared["motion_requests"][1])
        request["support_departure_json"] = json.dumps(support)
        request["support_departure_fingerprint"] = hashlib.sha256(request["support_departure_json"].encode()).hexdigest()
        responses.append({**_preview_response(request, target=.2),
            "support_departure_evidence": _support_departure_audit(support)})
    else:
        responses.append(_preview_response(prepared["motion_requests"][1], target=.2))
    result = preparation.finalize_outline_event_preview(prepared_preview=prepared, motion_results=responses)
    assert result["status"] == "prepared", result
    projection = result["geometry_evidence"]
    if clock_version == 1:
        from fractions import Fraction

        assert projection["projection_clock_version"] == 1
        intervals = [row for row in projection["observations"] if "continuous_interval" in row]
        assert [row["continuous_interval"] for row in intervals] == [["0", "1"], ["1", "1"], ["1", "2"]]
        assert Fraction(projection["observations"][-1]["time_exact"]) == 2
        overlay = [row for row in projection["observations"]
                   if row.get("owner_safety_evidence", {}).get("physical_state_updates")]
        assert overlay
        for point in overlay:
            assert point["owner_safety_evidence"]["physical_state_updates"]["ur5e-3"]["held_part"] == task["expected_end_state"]["held_part"]
        assert projection["owner_interval_contract"]["preview_fingerprint"] == recovery_validation_fingerprint(result["event_evidence"][0])
    else:
        assert "projection_clock_version" not in projection
    assert projection["interval_complete"] is True
    assert projection["resource_updates"]["held_part"] == task["expected_end_state"]["held_part"]
    assert projection["part_updates"]["part_location"] == task["expected_end_state"]["part_location"]
    assert projection["deferred_part_fields"] == ["part_state"]
    assert "part_state" not in projection["part_updates"]
    assert _validate(projection, all_rules=True, prospective_activation=True)["is_safe"]
    saved = json.loads(json.dumps(result, allow_nan=False))
    assert saved == result
    saved_scene = json.loads(json.dumps(prepared["physical_evidence"], allow_nan=False))
    replayed = project_recovery_outline_event(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, evidence=saved_scene, event_evidence=saved["event_evidence"])
    assert replayed == projection
    if grasp:
        intervals = saved["event_evidence"][0]["intervals"]
        assert len(intervals) == 3
        assert "joint_trajectory" in intervals[0]
        assert intervals[1]["custody_effects"][0]["kind"] == "grasp"
        assert intervals[2]["support_departure_evidence"]["continuous_support_departure_verified"]
        assert projection["projected_snapshot"]["parts"][task["part_name"]]["current_pose"][2] == pytest.approx(.2, abs=1e-13)
        for missing in ("support_departure", "support_departure_evidence", "native_support_departure_evidence"):
            incomplete = deepcopy(saved["event_evidence"])
            incomplete[0]["intervals"][-1].pop(missing)
            rejected = project_recovery_outline_event(task=task, recovery_snapshot=snapshot,
                recovery_des_model=model, evidence=saved_scene, event_evidence=incomplete)
            assert rejected["status"] == "needs_context", rejected
        retained = projection["successor_evidence"]["resources"][rid]["validated_pickup_target_calculation"]
        assert retained["part_name"] == task["part_name"]
        assert retained["calculation"]["travel_z"] == .2
        other_scene = deepcopy(projection["successor_evidence"])
        other_scene["resource_models"]["ur5e-4"]["geometry_unchanged_events"] = ["stay"]
        other_scene["snapshot"]["resources"]["ur5e-4"]["resource_state"] = "failed"
        other_task = {"resource_jid": "ur5e-4@localhost", "event_name": "other event", "outline_id": "RECOVERY_OTHER",
            "expected_end_state": {"resource_state": "idle"}, "params": {}}
        other_model = {"state_variables": {"resource_state": {"scope": "resource", "domain": ["failed", "idle"]}},
            "events": [{"event_name": "stay", "updates": {"resource_state": {"set": "idle"}},
                        "primitive_support": [{"primitive": "stay", "params": {}}]}]}
        other_prepared = preparation.prepare_outline_event_preview_request(task=other_task,
            recovery_snapshot=other_scene["snapshot"]["resources"]["ur5e-4"], recovery_des_model=other_model,
            physical_evidence=other_scene, resource_evidence={})
        assert other_prepared["status"] == "ready", other_prepared
        other_result = preparation.finalize_outline_event_preview(prepared_preview=other_prepared, motion_results=[])
        assert other_result["status"] == "prepared", other_result.get("reason", other_result)
        next_scene = other_result["geometry_evidence"]["successor_evidence"]
        assert next_scene["resources"][rid]["validated_pickup_target_calculation"] == retained
        assert next_scene["provenance"]["binding"]["resource_jid"] == other_task["resource_jid"]
        _, _, release_model, release_task = _custody_event_case(False)
        release_prepared = preparation.prepare_outline_event_preview_request(task=release_task,
            recovery_snapshot=projection["projected_resource_snapshot"], recovery_des_model=release_model,
            physical_evidence=next_scene, resource_evidence={})
        assert release_prepared["status"] == "ready", release_prepared
        release_record = release_prepared["event_record"]
        release_contract = release_prepared["physical_evidence"]["resource_models"][rid]["event_projection_contracts"][release_record["contract_id"]]
        assert release_contract["placement_target"]["departure_context"] == {"part_name": task["part_name"], "travel_z": .2}
        assert release_contract["placement_target"]["departure_context_source"]["binding"] == retained["binding"]
        for defect in ("part_name", "contract_fingerprint", "binding"):
            wrong = deepcopy(next_scene)
            wrong["resources"][rid]["validated_pickup_target_calculation"][defect] = (
                {"resource_jid": "ur5e-4@localhost"} if defect == "binding" else "different")
            unavailable = preparation.prepare_outline_event_preview_request(task=release_task,
                recovery_snapshot=projection["projected_resource_snapshot"], recovery_des_model=release_model,
                physical_evidence=wrong, resource_evidence={})
            assert unavailable["status"] == "NEEDS_CONTEXT", unavailable
    if not grasp:
        support = saved["event_evidence"][0]["intervals"][1]["part_support_evidence"]
        assert support["current_pose"] == projection["projected_snapshot"]["parts"][task["part_name"]]["current_pose"]
        assert projection["successor_evidence"]["stationary_part_evidence"][task["part_name"]] == support
        intervals = saved["event_evidence"][0]["intervals"]
        assert len(intervals) == 3 and intervals[1]["custody_effects"][0]["kind"] == "release"
        assert intervals[2]["joint_trajectory"]["points"][-1]["positions"] == [.2]
        assert task["part_name"] in intervals[2]["stationary_parts"]
        assert projection["projected_snapshot"]["parts"][task["part_name"]]["current_pose"][2] == 0
        assert "validated_pickup_target_calculation" not in projection["successor_evidence"]["resources"]["ur5e-3"]
        for defect in ("missing_motion", "missing_declaration", "wrong_part", "wrong_delta"):
            bad_scene, bad_records = deepcopy(saved_scene), deepcopy(saved["event_evidence"])
            record = bad_records[0]
            contract = bad_scene["resource_models"]["ur5e-3"]["event_projection_contracts"][record["contract_id"]]
            if defect == "missing_motion":
                record["intervals"].pop()
            elif defect == "missing_declaration":
                contract.pop("placement_departure")
            elif defect == "wrong_part":
                contract["placement_target"]["departure_context"]["part_name"] = "gear_small"
            else:
                contract["placement_departure"]["params"]["dz"] += .01
            record["contract_fingerprint"] = recovery_validation_fingerprint(contract)
            rejected = project_recovery_outline_event(task=task, recovery_snapshot=snapshot,
                recovery_des_model=model, evidence=bad_scene, event_evidence=bad_records)
            assert rejected["status"] == "needs_context", rejected


@pytest.mark.parametrize("complete_srdf", [True, False])
@pytest.mark.parametrize("recorded_mounting", [False, True])
def test_explicit_owner_preview_loads_named_pose_from_captured_srdf_without_execution(complete_srdf, recorded_mounting, monkeypatch):
    from unittest.mock import AsyncMock, Mock
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence, recovery_motion_preview

    scene, snapshot, model, task = _event_case()
    rid = "ur5e-3"
    owner_state = scene["snapshot"]["resources"][rid]
    owner_state.update(joint_names=["slide", "other"], joint_positions=[0, 0])
    resource_model = scene["resource_models"][rid]
    resource_model["named_joint_poses"] = {}
    configuration = resource_model["configuration"]
    configuration["joint_names"] = ["slide", "other"]
    configuration["joints"].append({"name": "other", "parent": "body", "child": "tool", "type": "prismatic",
        "xyz": [0, 0, 0], "rpy": [0, 0, 0], "axis": [0, 1, 0]})
    configuration["reference_link"] = "tool"
    configuration["components"].append({"id": "tool", "link": "tool", "bounds": [[-.01, .01]] * 3})
    srdf = '<robot name="recorded"><group_state name="home" group="owner_group"><joint name="other" value="0.3"/>'
    if complete_srdf:
        srdf += '<joint name="slide" value="0.2"/>'
    srdf += '</group_state></robot>'
    capture = {"robot_description": "recorded-urdf", "robot_description_semantic": srdf, "model_fingerprint": "recorded-urdf-srdf",
               "planning_scene": {"is_diff": False, "robot_state": {"is_diff": False,
                   "joint_state": {"name": ["slide", "other"], "position": [0, 0]}}}}
    settings = {"action_name": "/preview", "planning_scene_service": "/get_scene",
        "preview_parameter_service": "/preview/parameters", "max_translation_step": .01,
        "max_rotation_step": .1, "max_joint_step": .1, "velocity_scaling": .2,
        "acceleration_scaling": .2, "timeout_seconds": 10.}
    facts = {"execution_mode": "simulation", "resource_type": "robot", "controller_config": {"move_group": {
        "group_name": "owner_group", "ee_link": "tool", "frame_id": "world"}},
        "inputs": {"scene": {"safety_preparation": {"outline_preview": settings},
                             "planning_scene_mounting_contacts": {"not_from_observations": True}}}}
    observations = {"physical_evidence": scene, "inputs": {"scene": {"robots": [{"resource_id": rid}]}}}
    declarations = {"fixed_joints": ["saved_joint"], "world_contacts": []}
    if recorded_mounting:
        observations["inputs"]["scene"]["planning_scene_mounting_contacts"] = declarations
    mounting = Mock(side_effect=lambda native, **kwargs: {
        "planning_scene": {**deepcopy(native), "allowed_collision_matrix": {"recorded_mounting": True}},
        "evidence": {"declarations_source": "saved observations"},
    })
    captured = AsyncMock(return_value=deepcopy(capture))
    projected = Mock(side_effect=lambda native, evidence: {
        "planning_scene": deepcopy(native["planning_scene"]), "start_state": deepcopy(native["planning_scene"]["robot_state"])})

    async def preview(request, **kwargs):
        assert request["joint_target"] == {"name": ["slide", "other"], "position": [.2, .3]}
        assert (request["group_name"], request["link_name"], request["frame_id"]) == ("owner_group", "tool", "world")
        assert ("allowed_collision_matrix" in request["planning_scene"]) == recorded_mounting
        return recovery_motion_preview.preview_result_record(request, {
                "status": "PREVIEW_COMPLETE", "path_complete": True, "fraction": 1.0, "continuous_collision_verified": False,
                **{field: request[field] for field in ("resource_id", "capture_fingerprint", "proposal_fingerprint",
                                                       "scene_fingerprint", "model_fingerprint")},
                "trajectory": {"joint_trajectory": {"joint_names": ["slide", "other"], "points": [
                    {"positions": values, "velocities": [], "accelerations": [], "effort": [],
                     "time_from_start": {"sec": index, "nanosec": 0}}
                    for index, values in enumerate(([0, 0], [.2, .3]))]}}})

    preview_call = AsyncMock(side_effect=preview)
    monkeypatch.setattr(recovery_motion_preview, "capture_preview_scene", captured)
    monkeypatch.setattr(recovery_motion_preview, "build_scene_object_bindings", lambda *args, **kwargs: {})
    monkeypatch.setattr(recovery_motion_preview, "project_preview_scene", projected)
    monkeypatch.setattr(recovery_motion_preview, "preview_scene_fingerprint", lambda _: "isolated-scene")
    monkeypatch.setattr(recovery_motion_preview, "preview_recovery_motion", preview_call)
    monkeypatch.setattr(recovery_motion_preview, "apply_declared_mounting_contacts", mounting)
    monkeypatch.setattr(outline_gazebo_evidence, "_scene_owner_records", lambda *args: ({}, {}))
    forbidden = Mock(side_effect=AssertionError("Event preview cannot execute a command"))
    owner = SimpleNamespace(agent_name=rid, execution_mode="simulation", execute=forbidden)
    original = deepcopy(scene)
    call = outline_gazebo_evidence.prepare_owner_event_evidence(owner, task=task,
        recovery_snapshot=snapshot, recovery_des_model=model, physical_evidence=scene,
        resource_evidence=facts, observations=observations)
    if complete_srdf:
        result = asyncio.run(call)
        assert result["status"] == "prepared", result.get("reason", result)
        assert result["geometry_evidence"]["projected_snapshot"]["resources"][rid]["joint_positions"] == [.2, .3]
        preview_call.assert_awaited_once()
        projected.assert_called_once()
        if recorded_mounting:
            mounting.assert_called_once()
            assert mounting.call_args.kwargs["declarations"] == declarations
            assert mounting.call_args.kwargs["robot_description"] == "recorded-urdf"
            assert result["mounting_contact_evidence"] == {"declarations_source": "saved observations"}
        else:
            mounting.assert_not_called()
            assert "mounting_contact_evidence" not in result
    else:
        with pytest.raises(ValueError, match="SRDF named pose does not match"):
            asyncio.run(call)
        preview_call.assert_not_awaited()
        mounting.assert_not_called()
    captured.assert_awaited_once()
    forbidden.assert_not_called()
    assert scene == original


def test_named_pose_projects_recorded_joints_without_claiming_motion_interval():
    case = _case()
    case["resource_models"]["ur5e-3"]["named_joint_poses"]["home"]["positions"] = [1]
    before = deepcopy(case)
    result = project_recovery_geometry(**case)
    assert result["status"] == "projected", result
    assert result["observations"][-1]["region_occupancy"]["assembly_board-v1"]["ur5e-3"] is True
    assert result["interval_complete"] is False
    checked = _validate(result, prospective_activation=True)
    assert checked["validation_status"] == "NEEDS_CONTEXT"
    assert checked["requires_projection"] is True
    assert checked["context_requests"] == []
    assert case == before


def test_station_name_and_tcp_do_not_replace_missing_link_geometry():
    case = _case()
    case["resource_models"].pop("ur5e-4")
    result = project_recovery_geometry(**case)
    assert result["status"] == "needs_context"
    assert result["context_requests"] == ["/physical_evidence"]
    assert "component bounds" in result["unresolved"][0]


def test_interval_crossing_rejects_unsafe_alternatives_between_clear_endpoints():
    case = _case()
    case["resource_models"]["ur5e-4"]["configuration"]["root_xyz"] = [1, 0, 0]
    _motion(case, 2)
    projection = project_recovery_geometry(**case)
    assert projection["status"] == "projected", projection
    assert projection["observations"][0]["region_occupancy"]["assembly_board-v1"]["ur5e-3"] is False
    assert projection["observations"][-1]["region_occupancy"]["assembly_board-v1"]["ur5e-3"] is False
    checked = _validate(projection, prospective_activation=True)
    assert not checked["is_safe"]
    assert checked["validation_status"] == "NEEDS_CONTEXT"
    assert checked["counterexample"]["possible_violation"] is True


def test_joint_witness_requires_declared_stationary_peers_and_parts():
    case = _case()
    _motion(case, 0.2)
    case["step_evidence"][0].pop("stationary_parts")
    result = project_recovery_geometry(**case)
    assert result["status"] == "needs_context"
    assert result["requires_projection"] is True


def test_rule_local_labels_and_monitor_history_remain_separate():
    case = _case()
    _motion(case, 0.2)
    projection = project_recovery_geometry(**case)
    checked = _validate(projection, all_rules=True, prospective_activation=True)
    assert checked["is_safe"], checked
    assert len(checked["states_after"]) == 2
    assert all(set(row["ap_values"]) == {"ap001", "ap002"} for row in checked["rule_checks"])
    assert checked["execution_authorized"] is False
    assert checked["nominal_continuation_verified"] is False
    case["snapshot"] = projection["projected_snapshot"]
    _motion(case, 0.3, start=0.2)
    following = project_recovery_geometry(**case)
    continuation = _validate(following, all_rules=True, monitor_context=checked["monitor_context"])
    assert continuation["is_safe"], continuation
    assert len(continuation["monitor_context"]["observations"]) > len(checked["monitor_context"]["observations"])
    wrong = deepcopy(checked["monitor_context"])
    wrong["states"][next(iter(wrong["states"]))] = ["unverified"]
    rejected = _validate(following, all_rules=True, monitor_context=wrong)
    assert rejected["validation_status"] == "NEEDS_CONTEXT"
    assert "replayed" in rejected["reason"]


def test_missing_history_does_not_silently_reset_selected_rule():
    case = _case()
    _motion(case, 0.2)
    checked = _validate(project_recovery_geometry(**case))
    assert checked["validation_status"] == "NEEDS_CONTEXT"
    assert "prospective activation" in checked["reason"]


def test_symbolic_completed_does_not_replace_complete_process_ledger():
    case = _case()
    _motion(case, 0.2)
    case["snapshot"]["parts"]["gear_small"].update(state="assembled", processCompleted_complete=False)
    checked = _validate(project_recovery_geometry(**case), all_rules=True, prospective_activation=True)
    assert checked["validation_status"] == "NEEDS_CONTEXT"
    assert "explicitly complete" in checked["reason"]


def test_attachment_geometry_is_in_resource_occupancy_until_declared_release():
    case = _case()
    owner = case["snapshot"]["resources"]["ur5e-3"]
    owner.update(held_part="gear_small", grasp_transform=[1, 0, 0, 0, 0, 0, 1])
    case["snapshot"]["parts"]["gear_small"]["current_pose"] = [1, 0, 0, 0, 0, 0, 1]
    evidence = case["step_evidence"][0]
    evidence.pop("named_pose")
    evidence.update(geometry_unchanged=True, custody_effects=[{
        "kind": "release", "part": "gear_small", "pose": [1, 0, 0, 0, 0, 0, 1], "contained_by": None}])
    result = project_recovery_geometry(**case)
    assert result["status"] == "projected", result
    assert result["observations"][0]["region_occupancy"]["assembly_board-v1"]["ur5e-3"] is True
    last = result["observations"][-1]
    assert last["region_occupancy"]["assembly_board-v1"]["ur5e-3"] is False
    assert last["part_region_occupancy"]["assembly_board-v1"]["gear_small"] is True


def test_adapter_uses_exact_captured_jid_and_retains_projected_successor():
    case = _case()
    evidence = {key: deepcopy(case[key]) for key in ("snapshot", "geometry", "resource_models", "step_evidence")}
    evidence.update(resources=case["snapshot"]["resources"], parts=case["snapshot"]["parts"],
                    jids={"ur5e-3": "recovery-resource-3@localhost", "ur5e-4": "recovery-resource-4@localhost"})
    result = project_outline_evidence({"resource_jid": "recovery-resource-3@localhost"}, case["primitive_support"], evidence)
    assert result["status"] == "projected", result
    assert result["successor_evidence"]["snapshot"] == result["projected_snapshot"]
    assert result["successor_evidence"]["step_evidence"] == {}
    assert project_outline_evidence({"resource_jid": "ur5e-3"}, case["primitive_support"], evidence)["status"] == "needs_context"
    evidence["snapshot"]["resources"].pop("ur5e-4")
    evidence["geometry"]["resources"].pop("ur5e-4")
    rejected = project_outline_evidence({"resource_jid": "recovery-resource-3@localhost"}, case["primitive_support"], evidence)
    assert rejected["status"] == "needs_context"
    assert "every captured resource identity" in rejected["unresolved"][0]
    assert rejected["context_requests"] == ["/physical_evidence"]


def test_declared_tool_joint_projection_retains_model_and_defers_its_interval():
    case = _case()
    config = case["resource_models"]["ur5e-3"]["configuration"]
    config["joints"].append({"name": "finger", "parent": "body", "child": "finger_link",
                             "type": "prismatic", "xyz": [0, 0, 0], "rpy": [0, 0, 0],
                             "axis": [0, 1, 0], "frozen_position": 0})
    config["components"].append({"id": "finger_box", "link": "finger_link", "bounds": [[-0.01, 0.01]] * 3})
    after = deepcopy(config)
    after["joints"][-1]["frozen_position"] = 0.1
    evidence = case["step_evidence"][0]
    evidence.pop("named_pose")
    evidence.update(configuration_after=after, interval_complete=False)
    result = project_recovery_geometry(**case)
    assert result["status"] == "projected", result
    assert result["resource_models"]["ur5e-3"]["configuration"] == after
    assert config["joints"][-1]["frozen_position"] == 0
    assert result["interval_complete"] is False
    after["root_xyz"] = [10, 0, 0]
    rejected = project_recovery_geometry(**case)
    assert rejected["status"] == "needs_context"
    assert "cannot change" in rejected["unresolved"][0]


def test_part_may_reenter_during_interval_even_when_initially_inside():
    case = _case()
    owner = case["snapshot"]["resources"]["ur5e-3"]
    owner.update(held_part="KET4_Square_4mm", grasp_transform=[1, 0, 0, 0, 0, 0, 1])
    case["snapshot"]["parts"]["KET4_Square_4mm"]["current_pose"] = [1, 0, 0, 0, 0, 0, 1]
    _motion(case, 0)
    points = case["step_evidence"][0]["joint_trajectory"]["points"]
    points[0]["velocities"], points[1]["velocities"] = [4], [-4]
    result = project_recovery_geometry(**case)
    assert result["status"] == "projected", result
    interval = next(row for row in result["observations"] if row["phase"] == "between")
    assert interval["part_entry_possibilities"]["assembly_board-v1"]["KET4_Square_4mm"] == [False, True]
    checked = _validate(result, all_rules=True, prospective_activation=True)
    assert checked["validation_status"] == "NEEDS_CONTEXT"


def test_declared_primitive_completion_advances_precedence_with_predicted_evidence():
    from cais_spade_llm.resources.recovery_transition_derivation import validate_recovery_transition

    case = _case()
    case["snapshot"]["parts"]["KET4_Square_4mm"]["current_pose"] = [0, 0, 0, 0, 0, 0, 1]
    completed = [{"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}]
    declaration = {"name": "record_verified_completion", "params": {}, "required_params": [],
                   "preconditions": {"processCompleted": {"equals": []}},
                   "effects": {"processCompleted": {"set": completed}}}
    model = {"state_variables": {"processCompleted": {"scope": "part", "domain": [[], completed]}},
             "primitive_catalog": [declaration], "events": []}
    task = {"event_name": "proposed_completion", "resource_jid": "ur5e-3@localhost", "part_name": "gear_small",
            "expected_start_state": {"processCompleted": []}, "expected_end_state": {"processCompleted": completed}}
    validated = validate_recovery_transition(task=task, recovery_des_model=model,
                                             recovery_snapshot=case["snapshot"]["resources"]["ur5e-3"],
                                             part_context=case["snapshot"]["parts"]["gear_small"])
    assert validated["allowed"], validated
    support = validated["primitive_support"]
    support["part_name"] = task["part_name"]
    case["primitive_support"] = support
    case["step_evidence"] = [{**support["primitives"][0], "validated": True,
                              "source": "declared_test_owner", "geometry_unchanged": True}]
    projected = project_recovery_geometry(**case)
    assert projected["status"] == "projected", projected
    part = projected["projected_snapshot"]["parts"]["gear_small"]
    assert part["processCompleted"] == completed
    assert part["product_effect_evidence"][0]["kind"] == "predicted"
    assert part["product_effect_evidence"][0]["declaration"]["effects"] == declaration["effects"]
    first = _validate(projected, all_rules=True, prospective_activation=True)
    assert first["is_safe"], first
    following = _case()
    following["snapshot"] = projected["projected_snapshot"]
    command = {"primitive": "grasp_part", "params": {"part_name": "KET4_Square_4mm"}}
    following["primitive_support"] = {"semantics": "primitive_transition", "primitives": [command], "trace": [command]}
    following["step_evidence"] = [{**command, "validated": True, "source": "declared_test_owner", "geometry_unchanged": True,
                                   "custody_effects": [{"kind": "grasp", "part": "KET4_Square_4mm",
                                                       "pose": [0, 0, 0, 0, 0, 0, 1],
                                                       "grasp_transform": [0, 0, 0, 0, 0, 0, 1]}]}]
    grasp = project_recovery_geometry(**following)
    assert grasp["status"] == "projected", grasp
    after_grasp = _validate(grasp, all_rules=True, monitor_context=first["monitor_context"])
    assert after_grasp["is_safe"], after_grasp
    following = _case()
    following["snapshot"] = grasp["projected_snapshot"]
    _motion(following, 1)
    moving = project_recovery_geometry(**following)
    second = _validate(moving, all_rules=True, monitor_context=after_grasp["monitor_context"])
    assert second["is_safe"], second
    assert second["physical_execution_verified"] is False
    broken = deepcopy(case)
    broken["primitive_support"]["trace"][0]["declared_effects"] = {}
    assert project_recovery_geometry(**broken)["status"] == "needs_context"


@pytest.mark.parametrize("field", ["joint_names", "joint_positions"])
def test_missing_joint_facts_are_context_not_assumed_defaults(field):
    case = _case()
    case["snapshot"]["resources"]["ur5e-3"].pop(field)
    result = project_recovery_geometry(**case)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["execution_authorized"] is False


def _retrieval_case():
    case = _case()
    evidence = {key: deepcopy(case[key]) for key in ("snapshot", "geometry", "resource_models")}
    evidence.update(jids={"ur5e-3": "ur5e-3@localhost", "ur5e-4": "ur5e-4@localhost"},
                    resources={"ur5e-3": {"launch_id": "run", "observed_monotonic": 1}},
                    provenance={"captured_at": "first", "captured_monotonic": 1}, unresolved=[])
    evidence["geometry"]["parts"]["gear_small"]["source"] = {"model": "gear_small", "instance_id": "run", "simulation_time": 1}
    prepared = {"physical_evidence": deepcopy(evidence), "source_context_revision": 4,
                "recovery_resources": {}, "offline_recovery": False}
    session = {"projected_physical_evidence": deepcopy(evidence), "observation_count": 0,
               "max_observations": 10, "accepted_outline_prefix": []}
    return evidence, prepared, session


def test_timestamp_only_capture_does_not_retry_validator_and_retains_originals(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import (
        validate_event,
    )
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
        recovery_validation_fingerprint,
    )
    from cais_spade_llm.recovery_framework import recovery_observations

    evidence, prepared, session = _retrieval_case()
    original = deepcopy(prepared)
    fresh = deepcopy(evidence)
    fresh["provenance"].update(captured_at="second", captured_monotonic=2)
    fresh["resources"]["ur5e-3"]["observed_monotonic"] = 2
    fresh["geometry"]["parts"]["gear_small"]["source"]["simulation_time"] = 2
    fresh["snapshot"]["parts"]["gear_small"]["processCompleted_evidence"]["checkpoint"] = "later"
    calls = []

    def capture(_runtime):
        calls.append("capture")
        return deepcopy(fresh)

    async def validate(*, resource_jid, payload):
        calls.append("validate")
        return {"validator_jid": resource_jid, "results": [{"allowed": False,
            "feasibility_status": "NEEDS_CONTEXT", "context_requests": ["/physical_evidence"],
            "candidate_fingerprint": recovery_validation_fingerprint(payload["candidates"][0]["task"])}]}

    async def recorded_capture(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(recovery_observations, "capture_recovery_physical_evidence", capture)
    monkeypatch.setattr(asyncio, "to_thread", recorded_capture)
    actor = SimpleNamespace(environment_runtime=SimpleNamespace(context=SimpleNamespace(revision=4)),
                            request_recovery_outline_physical_validation=validate)
    turn = {}
    result, _ = asyncio.run(asyncio.wait_for(validate_event(task={"resource_jid": "ur5e-3@localhost", "outline_id": "candidate"},
        prepared=prepared, session=session, planner=SimpleNamespace(product_agent=actor), turn=turn), timeout=3))
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert calls == ["validate", "capture"]
    assert session["observation_count"] == 1
    assert prepared == original
    assert turn["validation_context_exchanges"][0]["returned_facts"]["/physical_evidence"]["value"] == fresh


def test_retrieval_progress_detects_changed_fact_and_complete_evidence():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import (
        _physical_fact_fingerprint,
    )

    evidence, _, _ = _retrieval_case()
    before = _physical_fact_fingerprint(evidence)
    changed = deepcopy(evidence)
    changed["snapshot"]["resources"]["ur5e-3"]["joint_positions"] = [0.2]
    assert _physical_fact_fingerprint(changed) != before
    changed = deepcopy(evidence)
    changed["snapshot"]["parts"]["gear_small"]["processCompleted_evidence"]["complete"] = False
    assert _physical_fact_fingerprint(changed) != before


def test_nominal_branches_share_the_same_fresh_observation_budget(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_nominal_evidence as nominal,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_validated_event as validated,
    )
    from cais_spade_llm.recovery_framework import recovery_observations

    evidence, prepared, session = _retrieval_case()
    session.update(max_observations=1, symbolic_resources={"ur5e-3@localhost": {"held_part": None}}, symbolic_parts={})
    prepared.update(llm_input={"goal_conditions": [{"entity_kind": "resource", "entity": "ur5e-3@localhost",
                                                   "field": "held_part", "expected": None}]},
                    saved_pending_nodes=[{"id": name, "resource_jid": name} for name in ("first", "second")])
    calls = []

    def capture(_runtime):
        calls.append("capture")
        changed = deepcopy(evidence)
        changed["snapshot"]["resources"]["ur5e-3"]["joint_positions"] = [0.2]
        return changed

    async def validate(*, task, session, prepared, planner, turn):
        await validated._retrieve(prepared, session, planner, ["/physical_evidence"], turn)
        return {"allowed": False, "reason": "NEEDS_CONTEXT: missing owner model"}, {}

    async def recorded_capture(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(recovery_observations, "capture_recovery_physical_evidence", capture)
    monkeypatch.setattr(asyncio, "to_thread", recorded_capture)
    monkeypatch.setattr(validated, "validate_event", validate)
    monkeypatch.setattr(nominal, "_nominal_task", lambda node, *_: ({"outline_id": node["id"]}, ""))
    actor = SimpleNamespace(environment_runtime=SimpleNamespace(context=SimpleNamespace(revision=4)))
    result = asyncio.run(nominal.validate_nominal_continuation(prepared=prepared, session=session,
                                                             planner=SimpleNamespace(product_agent=actor)))
    assert result["complete"] is False
    assert len(result["failures"]) == 2
    assert calls == ["capture"]
    assert session["observation_count"] == 1


def test_outline_checkpoint_reuses_one_fk_with_exact_endpoint_evidence(monkeypatch):
    from cais_spade_llm.resources import recovery_geometry as geometry
    from cais_spade_llm.resources.continuous_motion import ContinuousMotion

    case = _case()
    _motion(case, 0.)
    motion = ContinuousMotion(case["step_evidence"][0]["joint_trajectory"],
                              case["resource_models"]["ur5e-3"]["configuration"])
    args = (case["snapshot"], case["geometry"], case["resource_models"])
    expected = geometry._checkpoint(*args, index=1)
    expected["phase"] = "between"
    original, calls = motion.transforms, []

    def transforms(start, end):
        calls.append((start, end))
        return original(start, end)

    monkeypatch.setattr(motion, "transforms", transforms)
    actual = geometry._checkpoint(*args, index=1, moving=("ur5e-3", motion, 0, motion.duration))
    assert actual == expected
    assert len(calls) == 1


def test_outline_motion_geometry_is_local_and_does_not_accumulate_carried_parts(monkeypatch):
    from cais_spade_llm.resources import recovery_geometry as geometry
    from cais_spade_llm.resources.continuous_motion import ContinuousMotion

    case = _case()
    _motion(case, 1.)
    case["snapshot"]["resources"]["ur5e-3"].update(
        held_part="KET4_Square_4mm", grasp_transform=[.2, 0, 0, 0, 0, 0, 1])
    motion = ContinuousMotion(case["step_evidence"][0]["joint_trajectory"],
                              case["resource_models"]["ur5e-3"]["configuration"])
    args = (case["snapshot"], case["geometry"], case["resource_models"])
    kwargs = {"index": 1, "moving": ("ur5e-3", motion, 0, motion.duration)}
    before, calls, original = deepcopy(case), [], geometry._resource_boxes

    def boxes(*values):
        calls.append(None)
        return original(*values)

    monkeypatch.setattr(geometry, "_resource_boxes", boxes)
    expected = geometry._motion_checkpoint(*args, **kwargs)
    assert len(calls) == 1
    assert len(expected["continuous_cells"]) > 1
    actual = geometry._motion_checkpoint(*args, **kwargs)
    assert len(calls) == 2
    assert actual == expected
    actual["resources"]["ur5e-3"]["joint_positions"][0] = 42
    actual["continuous_cells"][0]["occupancy_possibilities"]["assembly_board-v1"]["ur5e-3"].append("changed")
    assert geometry._motion_checkpoint(*args, **kwargs) == expected
    assert len(calls) == 3
    assert case == before


@pytest.mark.parametrize("changed", ["joint_positions", "configuration"])
def test_outline_motion_rebuilds_peer_geometry_for_each_interval(changed):
    from cais_spade_llm.resources import recovery_geometry as geometry
    from cais_spade_llm.resources.continuous_motion import ContinuousMotion

    case = _case()
    _motion(case, 1.)
    motion = ContinuousMotion(case["step_evidence"][0]["joint_trajectory"],
                              case["resource_models"]["ur5e-3"]["configuration"])
    args = (case["snapshot"], case["geometry"], case["resource_models"])
    kwargs = {"index": 1, "moving": ("ur5e-3", motion, 0, motion.duration)}
    first = geometry._motion_checkpoint(*args, **kwargs)
    assert first["region_occupancy"]["assembly_board-v1"]["ur5e-4"] is False
    if changed == "joint_positions":
        case["snapshot"]["resources"]["ur5e-4"]["joint_positions"] = [-2.]
    else:
        case["resource_models"]["ur5e-4"]["configuration"]["root_xyz"][0] = 1.
    second = geometry._motion_checkpoint(*args, **kwargs)
    assert second["region_occupancy"]["assembly_board-v1"]["ur5e-4"] is True
    assert first["region_occupancy"]["assembly_board-v1"]["ur5e-4"] is False


@pytest.mark.parametrize("mode,status", [("clear", "FEASIBLE"), ("crosses", "INFEASIBLE"),
                                        ("tracking_error", "NEEDS_CONTEXT")])
def test_outline_local_geometry_reuse_preserves_full_projection_and_cca(mode, status, monkeypatch):
    from cais_spade_llm.resources import recovery_geometry as geometry

    options = {"crosses": mode == "crosses", "tracking_error": mode == "tracking_error"}
    _, actual = _correlated_motion_preview(**options)
    checkpoint = geometry._checkpoint

    def without_reuse(*args, **kwargs):
        kwargs.pop("_geometry_cache", None)
        return checkpoint(*args, **kwargs)

    monkeypatch.setattr(geometry, "_checkpoint", without_reuse)
    _, expected = _correlated_motion_preview(**options)
    assert actual == expected
    actual_result = _validate(actual, all_rules=True, prospective_activation=True)
    assert actual_result == _validate(expected, all_rules=True, prospective_activation=True)
    assert actual_result["validation_status"] == status


def _owner_projection_cache_case(monkeypatch):
    from cais_spade_llm.resources import recovery_geometry as geometry
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )
    from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryResource

    scene, snapshot, model, task = _event_case()
    scene['physical_preview_contract_version'] = 6
    scene['resource_models']['ur5e-3']['static_capabilities'] = {}
    evidence = {'source': {'kind': 'recorded', 'capture': 'fixture', 'revision': 1}}
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence)
    owner = SavedRecoveryResource(task['resource_jid'], {'recovery_snapshot': snapshot,
        'recovery_des_model': model}, {'jids': scene['jids']})
    count = []
    original = geometry.project_recovery_outline_event

    def project(**kwargs):
        count.append(1)
        return original(**kwargs)

    monkeypatch.setattr(geometry, 'project_recovery_outline_event', project)
    result = finalize_outline_event_preview(prepared_preview=prepared,
        motion_results=[_issued_native_preview(prepared)], owner_projector=owner.project_recovery_outline_event)
    assert result['status'] == 'prepared', result
    args = {'task': task, 'recovery_snapshot': snapshot, 'recovery_des_model': model,
            'physical_input': {'physical_evidence': result['physical_evidence']}}
    assert owner._outline_event_projection_cache is not None
    return owner, args, result['geometry_evidence'], count, original


def test_owner_projection_cache_reuses_only_local_result_and_keeps_cca_independent(monkeypatch):
    owner, args, expected, count, original = _owner_projection_cache_case(monkeypatch)
    assert len(count) == 1
    first = owner.project_recovery_outline_event(**deepcopy(args))
    assert first == expected and len(count) == 1
    first['projected_snapshot']['resources']['ur5e-3']['current_pose'][0] = 123.
    second = owner.project_recovery_outline_event(**deepcopy(args))
    assert second == expected and len(count) == 1
    assert second == original(task=args['task'], recovery_snapshot=args['recovery_snapshot'],
        recovery_des_model=args['recovery_des_model'], evidence=args['physical_input']['physical_evidence'])
    # Each candidate still independently evaluates both selected CCA monitors.
    assert _validate(second, all_rules=True, prospective_activation=True) == _validate(
        expected, all_rules=True, prospective_activation=True)
    explicit = deepcopy(args)
    explicit['physical_input']['outline_event_evidence'] = deepcopy(
        explicit['physical_input']['physical_evidence']['event_evidence'])
    assert owner.project_recovery_outline_event(**explicit) == expected and len(count) == 1


@pytest.mark.parametrize('changed', [
    'proposal', 'snapshot', 'model', 'peer_configuration', 'geometry', 'capability', 'policy',
    'source_revision', 'native_evidence', 'event_evidence', 'missing_versus_null', 'resource_evidence',
])
def test_owner_projection_cache_requires_every_current_input(changed, monkeypatch):
    owner, args, _, count, _ = _owner_projection_cache_case(monkeypatch)
    altered = deepcopy(args)
    scene = altered['physical_input']['physical_evidence']
    if changed == 'proposal':
        altered['task']['event_name'] = 'another_proposed_event'
    elif changed == 'snapshot':
        altered['recovery_snapshot']['current_pose'][0] += .1
    elif changed == 'model':
        altered['recovery_des_model']['events'][0]['updates']['resource_state']['set'] = 'failed'
    elif changed == 'peer_configuration':
        scene['snapshot']['resources']['ur5e-4']['joint_positions'][0] += .1
    elif changed == 'geometry':
        scene['geometry']['regions']['assembly_board-v1']['bounds'][0][0] += .01
    elif changed == 'capability':
        scene['resource_models']['ur5e-3']['static_capabilities']['recorded_revision'] = 2
    elif changed == 'policy':
        scene['physical_preview_contract_version'] = 5
    elif changed == 'source_revision':
        scene['source_revision'] = 2
    elif changed == 'native_evidence':
        scene['event_evidence'][0]['intervals'][0]['native_preview_binding']['response_fingerprint'] = 'changed'
    elif changed == 'event_evidence':
        altered['physical_input']['outline_event_evidence'] = []
    elif changed == 'resource_evidence':
        altered['physical_input']['resource_evidence'] = {'execution_mode': 'simulation'}
    else:
        altered['recovery_snapshot']['recorded_null'] = None
    owner.project_recovery_outline_event(**altered)
    assert len(count) == 2
    owner.project_recovery_outline_event(**deepcopy(args))
    assert len(count) == 3  # A single entry cannot retain the prior input as a second cache slot.


def test_owner_projection_cache_does_not_reuse_failures_or_supplied_geometry(monkeypatch):
    owner, args, valid, count, _ = _owner_projection_cache_case(monkeypatch)
    altered = deepcopy(args)
    altered['physical_input']['outline_event_evidence'] = []
    altered['physical_input']['geometry_evidence'] = valid
    for expected_count in (2, 3):
        result = owner.project_recovery_outline_event(**altered)
        assert result['status'] == 'needs_context'
        assert len(count) == expected_count
        assert owner._outline_event_projection_cache is None


def test_owner_projection_cache_does_not_survive_projector_replacement(monkeypatch):
    from cais_spade_llm.resources import recovery_geometry as geometry

    owner, args, expected, count, original = _owner_projection_cache_case(monkeypatch)
    calls = []
    def replacement(**kwargs):
        calls.append(1)
        return original(**kwargs)
    monkeypatch.setattr(geometry, 'project_recovery_outline_event', replacement)
    assert owner.project_recovery_outline_event(**args) == expected
    assert calls == [1] and count == [1]


def _nominal_owner_authority_case(*, projected=False):
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )

    scene, snapshot, model, task = _event_case()
    scene['physical_preview_contract_version'] = 7
    scene['resource_models']['ur5e-3']['static_capabilities'] = {}
    model['events'][0]['composition_contract'] = {'params': {}, 'steps': [
        {'id': 'home', 'op': 'move_to_named_pose', 'params': {'pose_name': 'home'}}]}
    task['required_task_contract'] = model['events'][0]['event_name']
    evidence = {'source': {'kind': 'recorded', 'capture': 'fixture', 'revision': 1},
        'execution_mode': 'simulation', 'state': {'_task_ctx': {'model_name': 'square'}}}
    if projected:
        scene['provenance'] = {'kind': 'projected', 'contract_fingerprint': 'accepted prior event'}
        scene['snapshot']['resources']['ur5e-3']['task_ctx.model_name'] = 'accepted projected model'
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence)
    assert prepared['status'] == 'ready', prepared
    result = finalize_outline_event_preview(prepared_preview=prepared,
        motion_results=[_issued_native_preview(prepared)], resource_evidence=evidence)
    assert result['status'] == 'prepared', result
    return {'task': task, 'recovery_snapshot': snapshot, 'recovery_des_model': model,
        'evidence': prepared['physical_evidence'], 'event_evidence': result['event_evidence'],
        'resource_evidence': evidence}, result['geometry_evidence']


@pytest.mark.parametrize('projected', [False, True])
def test_nominal_owner_context_comes_from_recorded_facts_and_accepted_projection(projected):
    kwargs, expected = _nominal_owner_authority_case(projected=projected)
    before = deepcopy(kwargs)
    result = project_recovery_outline_event(**kwargs)
    assert result == expected
    assert result['task_contract_witness']['status'] == 'supported'
    assert kwargs == before


@pytest.mark.parametrize('changed', ['execution_mode', 'model_name', 'missing', 'null'])
def test_rehashed_nominal_owner_context_cannot_change_source_guard_facts(changed):
    kwargs, _ = _nominal_owner_authority_case(projected=True)
    record = kwargs['event_evidence'][0]
    contracts = kwargs['evidence']['resource_models']['ur5e-3']['event_projection_contracts']
    contract = contracts[record['contract_id']]
    if changed == 'execution_mode':
        contract['nominal_owner_evidence']['execution_mode'] = 'hardware'
    elif changed == 'model_name':
        contract['nominal_owner_evidence']['state']['_task_ctx']['model_name'] = None
    elif changed == 'missing':
        kwargs.pop('resource_evidence')
    else:
        kwargs['resource_evidence'] = None
    identity = recovery_validation_fingerprint(contract)
    contracts[identity] = contract
    record.update(contract_id=identity, contract_fingerprint=identity)
    result = project_recovery_outline_event(**kwargs)
    assert result['status'] == 'needs_context'
    assert any('authoritative resource_evidence' in reason for reason in result['unresolved'])
    assert 'projected_snapshot' not in result


def test_nominal_owner_projection_cache_rechecks_authoritative_context(monkeypatch):
    from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryResource

    kwargs, expected = _nominal_owner_authority_case()
    owner = SavedRecoveryResource(kwargs['task']['resource_jid'], {}, {'jids': kwargs['evidence']['jids']})
    arguments = {key: kwargs[key] for key in ('task', 'recovery_snapshot', 'recovery_des_model')}
    arguments['physical_input'] = {'physical_evidence': kwargs['evidence'],
        'outline_event_evidence': kwargs['event_evidence'], 'resource_evidence': kwargs['resource_evidence']}
    assert owner.project_recovery_outline_event(**arguments) == expected
    assert owner._outline_event_projection_cache is not None
    arguments['physical_input']['resource_evidence'] = deepcopy(kwargs['resource_evidence'])
    arguments['physical_input']['resource_evidence']['execution_mode'] = 'hardware'
    result = owner.project_recovery_outline_event(**arguments)
    assert result['status'] == 'needs_context'
    assert owner._outline_event_projection_cache is None


@pytest.mark.parametrize("target,value", [("current_state", "failed"),
    ("held_part", "gear_small"), ("gripper_state", "closed")])
def test_nominal_projection_rejects_conflicting_declared_bookkeeping(target, value):
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )

    scene, snapshot, model, task = _event_case()
    scene["physical_preview_contract_version"] = 7
    scene["resource_models"]["ur5e-3"]["static_capabilities"] = {}
    scene["snapshot"]["resources"]["ur5e-3"]["gripper_state"] = "open"
    model["events"][0]["composition_contract"] = {"params": {}, "steps": [
        {"id": "home", "op": "move_to_named_pose", "params": {"pose_name": "home"}}],
        "effects": [{"target": target, "action": "set", "value": value}]}
    task["required_task_contract"] = model["events"][0]["event_name"]
    evidence = {"source": {"kind": "recorded", "capture": "fixture", "revision": 1},
        "execution_mode": "simulation", "state": {"_task_ctx": {}}}
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence=evidence)
    assert prepared["status"] == "ready", prepared
    result = finalize_outline_event_preview(prepared_preview=prepared,
        motion_results=[_issued_native_preview(prepared)], resource_evidence=evidence)
    assert result["status"] == "NEEDS_CONTEXT", result
    assert any("Nominal owner effect _" + target + " conflicts with supported" in reason
               for reason in result["geometry_evidence"]["unresolved"])
    assert "projected_snapshot" not in result["geometry_evidence"]


def _continuous_mating_rejection_case():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_mating_contact, preview_result_record
    from cais_spade_llm.resources.recovery_mating_contact import prepare_mating_contact
    from cais_spade_llm.resources.resource_safety_preparation import prepare_outline_event_preview_request

    scene, snapshot, model, task = _event_case()
    rid, part = 'ur5e-3', 'gear_small'
    scene['physical_preview_contract_version'] = 3
    scene['resource_models'][rid]['configuration']['joints'][0]['axis'] = [0., 0., 1.]
    state = scene['snapshot']
    state['resources'][rid].update(held_part=part, grasp_transform=[0., 0., 0., 0., 0., 0., 1.])
    state['parts'][part].update(current_pose=[0., 0., 0., 0., 0., 0., 1.], model_name=part)
    snapshot['held_part'] = part
    policy = {'target_collision_object': 'Gear_Plate/Gear_Shaft_1/collision', 'axis_tolerance_m': .0005,
        'max_contact_depth_m': .0005, 'target_yaw_rad': 0., 'yaw_tolerance_rad': .005,
        'ignore_tooth_contact_with': [], 'retain_fixture_attachment': True}
    owner = {'execution_mode': 'simulation', 'source': {'capture': 'fixture'}, 'controller_config': {
        'payload_collision': {'enabled': True}, 'move_group': {'ee_link': 'body', 'frame_id': 'world'},
        'services': {'attach': '/ATTACHLINK', 'detach': '/DETACHLINK'}}, 'inputs': {'geometry': {
            'assembly_board': {'model_name': 'assembly_board_v1', 'component_attachment_model': 'assembly_board_v1',
                'mating_contact_by_part': {part: policy}, 'center': {'x': 0., 'y': 0.},
                'slots': {part: [0., 0.]}, 'target_origin_z_m': {part: 0.}},
            'parts': {'model_map': {part: part}, 'assembly_target_map': {part: 'Gear_Plate/Gear_Shaft_1'}}}}}
    scene['physics_evidence'] = {'instance_id': 'fixture', 'simulation_time': 1., 'models': {
        part: {'joint_count': 0, 'links': {'link': {'collisions': [
            {'id': 'gear_small::link::collision', 'geometry': {'type': 'mesh'}}]}}},
        'Gear_Plate': {'static': True, 'links': {'Gear_Shaft_1': {'collisions': [
            {'id': 'Gear_Plate::Gear_Shaft_1::collision', 'geometry': {'type': 'cylinder'}}]}}}}}
    scene['scene_object_bindings'] = {part: {'objects': [{'object_id': 'gear_small/link/collision',
        'recorded_geometry': {'collision_id': 'gear_small::link::collision'}}]}}
    scene['support_collision_object_bindings'] = {'Gear_Plate::Gear_Shaft_1::collision': {
        'object_id': 'Gear_Plate/Gear_Shaft_1/collision', 'source': 'captured SDF'}}
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence=owner)
    assert prepared['status'] == 'ready', prepared
    contract = prepared['physical_evidence']['resource_models'][rid]['event_projection_contracts'][prepared['event_record']['contract_id']]
    contract['mating_owner_evidence'] = deepcopy(owner)
    identity = recovery_validation_fingerprint(contract)
    prepared['physical_evidence']['resource_models'][rid]['event_projection_contracts'] = {identity: contract}
    prepared['event_record'].update(contract_id=identity, contract_fingerprint=identity)
    prepared['motion_requests'] = [{'mode': 'cartesian', 'waypoints': [{'position': {'x': 0., 'y': 0., 'z': 0.},
        'orientation': {'x': 0., 'y': 0., 'z': 0., 'w': 1.}}]}]
    response = _issued_native_preview(prepared)
    mating = prepare_mating_contact(resource_id=rid, part_name=part, state=state, evidence=prepared['physical_evidence'],
        owner_evidence=owner, target_origin_pose={'x': 0., 'y': 0., 'z': 0.}, target_tool_pose=[0., 0., 0., 0., 0., 0., 1.])
    assert mating['status'] == 'prepared', mating
    request = bind_mating_contact(response['request'], mating)
    response['request'] = request
    trajectory = response['trajectory']['joint_trajectory']
    trajectory['points'][0]['positions'] = trajectory['points'][1]['positions'] = [0.]
    trajectory['points'][0]['velocities'] = [-.1]
    trajectory['points'][1]['velocities'] = [.1]
    binding = {field: request[field] for field in ('resource_id', 'capture_fingerprint', 'proposal_fingerprint',
        'scene_fingerprint', 'model_fingerprint')}
    binding.update(support_departure_fingerprint='', mating_contact_fingerprint=request['mating_contact_fingerprint'])
    audit = {'version': 1, 'source': 'PreviewRecoveryMotion', 'request_binding': binding,
        'pairs_complete': True, 'continuous_collision_verified': False, 'samples': [{
            'state_index': i, 'phase': 'trajectory', 'joint_state': {'name': ['slide'], 'position': [0.]},
            'part_pose': [0., 0., 0., 0., 0., 0., 1.], 'checked_contacts': 0, 'contacts': []} for i in range(2)]}
    response.update(mating_contact_fingerprint=request['mating_contact_fingerprint'],
        mating_contact_evidence_json=json.dumps(audit), sampled_collision_rejected=False)
    prepared['native_requests'] = [request]
    return prepared, preview_result_record(request, response)


def test_continuous_mating_counterexample_replays_without_native_collision_or_state_commit():
    from cais_spade_llm.resources.resource_safety_preparation import outline_event_preview_progress, finalize_outline_event_preview
    from cais_spade_llm.recovery_framework.outline_validation import rejected_outline_motion_projection

    prepared, response = _continuous_mating_rejection_case()
    before = deepcopy((prepared, response))
    progress = outline_event_preview_progress(prepared_preview=prepared, motion_results=[response])
    result = finalize_outline_event_preview(prepared_preview=prepared, motion_results=[response])
    assert result == progress
    assert result['status'] == 'rejected' and result['continuous_mating_rejected']
    assert result['rejected_motion']['status'] == 'PREVIEW_COMPLETE'
    assert result['rejected_motion']['sampled_collision_rejected'] is False
    assert result['mating_rejection']['counterexample']['violated_conditions']
    checked = rejected_outline_motion_projection(record=result, task=prepared['task'],
        recovery_snapshot=prepared['recovery_snapshot'], recovery_des_model=prepared['recovery_des_model'],
        physical_evidence=prepared['physical_evidence'])
    assert checked['status'] == 'rejected' and checked['mating_rejection'] == result['mating_rejection']
    assert checked['observations'] == [] and checked['interval_complete'] is False
    assert not {'event_evidence', 'geometry_evidence', 'projected_snapshot', 'monitor_context'} & result.keys()
    assert 'physical_evidence' not in result['mating_replay_preparation']
    assert (prepared, response) == before


@pytest.mark.parametrize('defect', ['counterexample', 'task', 'native_request', 'native_result', 'stage', 'prefix', 'model'])
def test_continuous_mating_rejection_rejects_forged_or_stale_saved_evidence(defect):
    from cais_spade_llm.resources.resource_safety_preparation import finalize_outline_event_preview
    from cais_spade_llm.recovery_framework.outline_validation import rejected_outline_motion_projection

    prepared, response = _continuous_mating_rejection_case()
    record = finalize_outline_event_preview(prepared_preview=prepared, motion_results=[response])
    assert record['status'] == 'rejected', record
    scene = deepcopy(prepared['physical_evidence'])
    if defect == 'counterexample': record['mating_rejection']['counterexample']['time_from_start'] += .1
    elif defect == 'task': record['task']['event_name'] = 'changed'
    elif defect == 'native_request': record['native_requests'][0]['scene_fingerprint'] = 'changed'
    elif defect == 'native_result': record['motion_results'][0]['joint_trajectory']['points'][0]['positions'] = [.1]
    elif defect == 'stage': record['rejected_stage_index'] += 1
    elif defect == 'prefix': record['mating_replay_preparation']['event_record']['intervals'] = [{'geometry_unchanged': True}]
    else: scene['resource_models']['ur5e-3']['configuration']['joints'][0]['axis'] = [1., 0., 0.]
    with pytest.raises(ValueError):
        rejected_outline_motion_projection(record=record, task=prepared['task'],
            recovery_snapshot=prepared['recovery_snapshot'], recovery_des_model=prepared['recovery_des_model'],
            physical_evidence=scene)


def test_continuous_mating_ambiguous_bounds_remain_needs_context(monkeypatch):
    from cais_spade_llm.resources import recovery_mating_contact
    from cais_spade_llm.resources.resource_safety_preparation import finalize_outline_event_preview

    prepared, response = _continuous_mating_rejection_case()
    monkeypatch.setattr(recovery_mating_contact, 'verify_mating_contact', lambda **_: {
        'status': 'NEEDS_CONTEXT', 'reason': 'Continuous mating corridor remains unresolved'})
    result = finalize_outline_event_preview(prepared_preview=prepared, motion_results=[response])
    assert result['status'] == 'NEEDS_CONTEXT'
    assert result['reason'] == 'Continuous mating corridor remains unresolved'
    assert not {'physical_motion_rejected', 'continuous_mating_rejected', 'event_evidence', 'geometry_evidence'} & result.keys()


@pytest.mark.parametrize('nominal', [False, True])
def test_workflow_retains_continuous_mating_rejection_and_pending_scene_without_llm(tmp_path, monkeypatch, nominal):
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery
    from cais_spade_llm.resources.resource_safety_preparation import finalize_outline_event_preview
    from cais_spade_llm.recovery_framework.outline_validation import rejected_outline_motion_projection
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_validated_event

    preview, response = _continuous_mating_rejection_case()
    record = finalize_outline_event_preview(prepared_preview=preview, motion_results=[response])
    task = preview['task']
    source_scene = deepcopy(preview['physical_evidence'])
    source_scene['resource_models']['ur5e-3']['event_projection_contracts'] = {}
    session = {'turn_index': 2, 'accepted_outline_prefix': [{'event_name': 'retained'}],
        'outline_monitor_context': {'original': {'state': 3}}, 'projected_physical_evidence': deepcopy(source_scene),
        'pending_outline_validation': {'kind': 'nominal' if nominal else 'event', 'task': deepcopy(task),
            'response': {'next_transition': deepcopy(task)}, 'recovery_snapshot': deepcopy(preview['recovery_snapshot'])}}
    if nominal:
        session['pending_outline_validation']['projected_physical_evidence'] = deepcopy(source_scene)
    prepared = {'multi_turn_session_state': session, 'recovery_resources': {task['resource_jid']: {
        'recovery_des_model': preview['recovery_des_model'], 'recovery_snapshot': preview['recovery_snapshot']}}}
    recovery = SimpleNamespace(_stopped=False, _generation=1, _prepared=prepared,
        offline=True, _offline_cancel=threading.Event(), _observations={},
        runtime=SimpleNamespace(source_observation_path=None),
        agent=SimpleNamespace(process_planner=object()), _state={'artifact_directory': str(tmp_path)})
    recovery._publish = lambda **values: recovery._state.update(values)
    recovery._offline_work = PartSlippageRecovery._offline_work.__get__(recovery)
    recovery._offline_preview_observations = PartSlippageRecovery._offline_preview_observations.__get__(recovery)
    called = []

    async def retry(*, prepared, session, planner):
        stored = session['projected_physical_evidence']['rejected_event_evidence'][0]
        checked = rejected_outline_motion_projection(record=stored, task=task,
            recovery_snapshot=preview['recovery_snapshot'], recovery_des_model=preview['recovery_des_model'],
            physical_evidence=session['projected_physical_evidence'])
        assert checked['continuous_mating_rejected']
        called.append(deepcopy(stored))
        return 'need_revision', {'turn_index': 2}

    monkeypatch.setattr(multi_turn_validated_event, 'retry_pending_event', retry)
    original = deepcopy(prepared)
    asyncio.run(PartSlippageRecovery.retry_outline_validation(recovery, {**record,
        'physical_evidence': preview['physical_evidence']}))
    current = recovery._prepared['multi_turn_session_state']
    assert len(called) == 1 and recovery._state['status'] == 'outline_ready'
    assert called[0]['mating_rejection'] == record['mating_rejection']
    for key in ('accepted_outline_prefix', 'outline_monitor_context'):
        assert current[key] == original['multi_turn_session_state'][key]
    assert current['projected_physical_evidence']['snapshot'] == source_scene['snapshot']
    if nominal:
        assert current['pending_outline_validation']['projected_physical_evidence'] == source_scene
    saved = json.loads(next((tmp_path/'recovery_outline'/'turns').glob('*.json')).read_text())
    assert saved['request_sent'] is False and saved['validation_only'] is True


def test_progress_scene_remains_detached_after_reusing_its_owned_copy():
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, outline_event_preview_progress,
    )

    scene, snapshot, model, task = _event_case()
    scene["retained_records"] = [{"capture": "original"}]
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence={})
    responses = [_preview_response(prepared["motion_requests"][0])]
    before = deepcopy((prepared, responses, scene))
    result = outline_event_preview_progress(prepared_preview=prepared, motion_results=responses)
    assert (prepared, responses, scene) == before
    physical = result["physical_evidence"]
    assert list(physical) == list(prepared["physical_evidence"])
    physical["snapshot"]["resources"]["ur5e-4"]["held_part"] = "gear_small"
    physical["resource_models"]["ur5e-3"]["configuration"]["joint_names"].append("changed")
    physical["geometry"]["frame"] = "changed"
    physical["retained_records"][0]["capture"] = "changed"
    assert (prepared, responses, scene) == before


def test_successor_preserves_order_and_isolation_without_copying_replaced_step_evidence():
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )

    class DiscardedEvidence(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("Replaced primitive step evidence must not be copied into the successor")

    scene, snapshot, model, task = _event_case()
    scene["step_evidence"] = {"old": ["preserved source"]}
    scene["retained_records"] = [{"capture": "original"}]
    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=scene, resource_evidence={})
    result = finalize_outline_event_preview(prepared_preview=prepared,
        motion_results=[_preview_response(prepared["motion_requests"][0])])
    assert result["status"] == "prepared", result
    evidence = deepcopy(prepared["physical_evidence"])
    expected = project_recovery_outline_event(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, evidence=evidence, event_evidence=result["event_evidence"])
    original = deepcopy(evidence)
    archive = DiscardedEvidence(evidence["step_evidence"])
    evidence["step_evidence"] = archive
    actual = project_recovery_outline_event(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, evidence=evidence, event_evidence=result["event_evidence"])
    assert actual["status"] == "projected", actual
    assert json.dumps(actual) == json.dumps(expected)
    successor = actual["successor_evidence"]
    assert list(successor) == list(expected["successor_evidence"])
    assert successor["step_evidence"] == {}
    successor["snapshot"]["resources"]["ur5e-4"]["held_part"] = "gear_small"
    successor["resource_models"]["ur5e-3"]["configuration"]["joint_names"].append("changed")
    successor["geometry"]["frame"] = "changed"
    successor["retained_records"][0]["capture"] = "changed"
    assert evidence == original and evidence["step_evidence"] is archive


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("defect", ["none", "mapping", "string", "missing", "entry_none", "entry_scalar",
    "missing_endpoint", "empty_endpoint", "nonstring_endpoint", "unknown_model", "unknown_link",
    "invalid_model", "invalid_link", "missing_complete", "false_complete"])
def test_stationary_capture_rejects_malformed_graph_before_rest_fallback(disabled, defect, monkeypatch):
    from cais_spade_llm.recovery_framework.recovery_observations import _stationary_part_observation
    from cais_spade_llm.resources import recovery_geometry

    physics = _supported_physics()
    physics.update(attachment_complete=True, attachments=[])
    physics["models"]["square"]["joint_count"] = 0
    physics["models"]["square"]["links"]["link"].update(
        enabled=not disabled, linear_speed=0., angular_speed=0.)
    row = {"model1":"table", "link1":"link", "model2":"square", "link2":"link"}
    if defect == "none": physics["attachments"] = None
    elif defect == "mapping": physics["attachments"] = {"row":row}
    elif defect == "string": physics["attachments"] = "unknown"
    elif defect == "missing": physics.pop("attachments")
    elif defect == "entry_none": physics["attachments"] = [None]
    elif defect == "entry_scalar": physics["attachments"] = [7]
    elif defect == "missing_complete": physics.pop("attachment_complete")
    elif defect == "false_complete": physics["attachment_complete"] = False
    else:
        physics["attachments"] = [row]
        if defect == "missing_endpoint": row.pop("model2")
        elif defect == "empty_endpoint": row["model2"] = ""
        elif defect == "nonstring_endpoint": row["model2"] = []
        elif defect == "unknown_model": row["model1"] = "missing"
        elif defect == "unknown_link": row["link1"] = "missing"
        elif defect == "invalid_model": physics["models"]["table"] = None
        elif defect == "invalid_link": physics["models"]["table"]["links"]["link"] = None
    part = {"name":"square", "pose":deepcopy(physics["models"]["square"]["pose"])}
    evidence = {"snapshot":{"resources":{"table":{"held_part":None}}, "parts":{"square":{"contained_by":None}}},
        "resources":{"table":{"model_static":True, "idle":True}}}
    def forbidden(**kwargs):
        pytest.fail("Malformed attachment graph reached a support fallback")
    for name in ("fixed_attached_part_observation", "disabled_part_observation", "gravity_supported_part"):
        monkeypatch.setattr(recovery_geometry, name, forbidden)
    before = deepcopy((physics, part, evidence))
    result = _stationary_part_observation("square", part, physics, evidence, {})
    assert result["status"] == "NEEDS_CONTEXT"
    assert "attachment graph" in result["reason"]
    assert (physics, part, evidence) == before


@pytest.mark.parametrize("disabled", [False, True])
def test_stationary_capture_accepts_complete_empty_attachment_graph(disabled):
    from cais_spade_llm.recovery_framework.recovery_observations import _stationary_part_observation

    physics = _supported_physics()
    physics.update(attachment_complete=True, attachments=[])
    physics["models"]["square"]["joint_count"] = 0
    physics["models"]["square"]["links"]["link"].update(
        enabled=not disabled, linear_speed=0., angular_speed=0.)
    part = {"name":"square", "pose":deepcopy(physics["models"]["square"]["pose"])}
    evidence = {"snapshot":{"resources":{"table":{"held_part":None}}, "parts":{"square":{"contained_by":None}}},
        "resources":{"table":{"model_static":True, "idle":True}}}
    configuration = {"part_support":{"kind":"gravity_supported_rest", "requires_no_active_goals":True,
        "future_execution_tracking":"not_established", "stationary_linear_speed_max":.001,
        "stationary_angular_speed_max":.001}}
    result = _stationary_part_observation("square", part, physics, evidence, configuration)
    assert result["status"] == "supported", result
    assert result["kind"] == ("Gazebo ODE disabled body" if disabled else "gravity_supported_rest")
