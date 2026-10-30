from __future__ import annotations

"""Recorded recovery geometry, hold, support, and capture evidence without live calls."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from cais_spade_llm.resources.recovery_geometry import (
    outline_event_binding,
    project_recovery_geometry,
)
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint


@pytest.mark.parametrize('resource_id', ['ur5e-1', 'ur5e-2'])
def test_recovery_uses_the_nominal_handling_robot_access_calculation(resource_id):
    from cais_spade_llm.recovery_framework import SCENE_PATH
    from cais_spade_llm.resources.robot.target_calculations import handling_robot_access_poses
    from cais_spade_llm.resources.robot.recovery_capability import calculate_recovery_pick_target

    machine = next(row for row in json.loads(SCENE_PATH.read_text())['machines']
                   if row['handling_robot_access']['resource_id'] == resource_id)
    access = machine['handling_robot_access']
    pose = [*machine['workholding_pose'][:3], 0., 1., 0., 0.]
    part = 'recorded-part'
    evidence = {'resource_id': resource_id, 'execution_mode': 'simulation', 'frame': 'world', 'observed_parts': {part: {
        'current_pose': pose, 'location': machine['resource_id']}},
        'inputs': {'scene': {'machines': [machine]}}, 'pick_geometry': {part: {'part_height_m': .05}}}
    snapshot = {'current_pose': dict(zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), pose, strict=True))}
    before = deepcopy((evidence, snapshot))
    result = calculate_recovery_pick_target(task={'resource_jid': resource_id+'@localhost', 'part_name': part},
        recovery_snapshot=snapshot, resource_evidence=evidence)
    expected = handling_robot_access_poses(access, access['tcp_offset_z_m'], tuple(pose[:2]),
                                          access.get('grasp_orientation_xyzw'))
    assert result['status'] == 'calculated', result
    assert (result['approach_pose'], result['target_pose']) == expected
    assert result['access_retreat_pose'] == expected[0]
    assert (evidence, snapshot) == before


def _kmr_joint_physical():
    names = [f"joint_a{i}" for i in range(1, 8)]
    config = {"root": "KMR_base_link", "reference_link": "tool", "root_xyz": [0., 0., 0.],
        "root_quaternion": [0., 0., 0., 1.], "joint_names": names, "interpolation": "splines",
        "joints": [{"name": name, "parent": "KMR_base_link" if i == 0 else "link"+str(i),
                    "child": "tool" if i == 6 else "link"+str(i+1), "type": "revolute", "axis": [0., 0., 1.],
                    "xyz": [0., 0., .1], "rpy": [0., 0., 0.], "position_bounds": [-3., 3.]}
                   for i, name in enumerate(names)],
        "components": [{"id": "tool", "link": "tool", "bounds": [[-.02, .02]]*3}]}
    owner = {"joint_names": names, "joint_positions": [0.]*7, "held_part": None,
             "current_pose": [0., 0., .7, 0., 0., 0., 1.]}
    return {"snapshot": {"frame": "world", "resources": {"KMR": owner, "M1": {"part_name": None}}, "parts": {}},
        "resource_models": {"KMR": {"configuration": config}, "M1": {}},
        "resources": {"KMR": {}, "M1": {"model_static": True}},
        "geometry": {"frame": "world", "regions": {}, "parts": {}, "resources": {
            "KMR": {}, "M1": {"stationary_only": True, "component_bounds": [{"id": "body", "bounds": [[4., 5.]]*3}]}}}}


@pytest.mark.parametrize('defect', [None, 'missing_limits', 'no_witness', 'wrong_witness'])
def test_robot_joint_calculation_requires_bounded_independently_verified_endpoint(monkeypatch, defect):
    from cais_spade_llm.resources.robot.recovery_capability import _cartesian_joint_witness

    physical = _kmr_joint_physical()
    physical['jids'] = {'KMR': 'owner@localhost'}
    state = physical['snapshot']['resources']['KMR']
    pose = dict(zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), state['current_pose'], strict=True))
    if defect == 'missing_limits':
        physical['resource_models']['KMR']['configuration']['joints'][0].pop('position_bounds')
    elif defect in {'no_witness', 'wrong_witness'}:
        monkeypatch.setattr('cais_spade_llm.recovery_framework.kmr_recovery._calculate_cartesian_joints',
            lambda *args: None if defect == 'no_witness' else {'joint_positions': [.1]*7})
    before = deepcopy(physical)
    result = _cartesian_joint_witness(pose=pose, resource_jid='owner@localhost',
        recovery_snapshot=state, resource_evidence={}, physical_evidence=physical)
    assert result['feasibility_status'] == ('FEASIBLE' if defect is None else 'NEEDS_CONTEXT')
    assert physical == before and result['evidence']['execution_authorized'] is False


def test_candidate_joint_calculation_cache_rechecks_evidence_and_changed_geometry(monkeypatch):
    import scipy.optimize
    from cais_spade_llm.recovery_framework.kmr_recovery import _cartesian_joint_calculation_cache
    from cais_spade_llm.resources.robot.recovery_capability import _cartesian_joint_witness

    solver = scipy.optimize.least_squares
    calls = []

    def solve(*args, **kwargs):
        calls.append(True)
        return solver(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, 'least_squares', solve)
    physical = _kmr_joint_physical()
    physical['jids'] = {'KMR': 'owner@localhost'}
    state = physical['snapshot']['resources']['KMR']
    pose = dict(zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), state['current_pose'], strict=True))

    def check():
        return _cartesian_joint_witness(pose=pose, resource_jid='owner@localhost',
            recovery_snapshot=state, resource_evidence={}, physical_evidence=physical)

    token = _cartesian_joint_calculation_cache.set({})
    try:
        first = check()
        assert first['feasibility_status'] == 'FEASIBLE'
        count = len(calls)
        state['resource_state'] = 'changed'
        again = check()
        assert again['feasibility_status'] == 'FEASIBLE' and len(calls) == count
        assert first['evidence']['starting_snapshot_fingerprint'] != again['evidence']['starting_snapshot_fingerprint']
        first['evidence']['joint_positions'][0] = 999
        assert check()['evidence']['joint_positions'][0] != 999
        physical['resource_models']['KMR']['configuration']['root_xyz'][0] += .02
        changed = check()
        # The recorded pose is at the arm's full extension, so the moved base puts it beyond the whole chain:
        # proven unreachable without a cached answer or a search.
        assert changed['feasibility_status'] == 'INFEASIBLE' and len(calls) == count
        physical['resource_models']['KMR']['configuration']['root_xyz'][0] -= .02
    finally:
        _cartesian_joint_calculation_cache.reset(token)
    assert _cartesian_joint_calculation_cache.get() is None
    count = len(calls)
    assert check()['feasibility_status'] == 'FEASIBLE'
    assert len(calls) > count


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


def test_endpoint_hold_encloses_numeric_roundoff_without_relaxing_legacy_limits():
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence

    _, rid, state, model, physical = _commanded_hold_case()
    configuration = model['configuration']
    parent = configuration['joints'][-1]['child']
    configuration['joints'].append({'name': 'passive', 'parent': parent, 'child': 'passive_link',
        'type': 'prismatic', 'axis': [0., 0., 1.], 'xyz': [0., 0., 0.], 'rpy': [0., 0., 0.],
        'position_bounds': [.015, .11], 'frozen_position': .11000000000000007})
    original = deepcopy(model)
    inputs = dict(resource_id=rid, state=state, model=model, physical=physical)
    assert commanded_hold_evidence(**inputs)['status'] == 'NEEDS_CONTEXT'
    bounded = commanded_hold_evidence(**inputs, include_observation_roundoff=True)
    assert bounded['status'] == 'supported', bounded.get('reason')
    assert bounded['joint_intervals']['passive'][1] > .11000000000000007
    assert bounded['observation_roundoff']['passive']['declared_bounds'] == [.015, .11]
    assert model == original
    configuration['joints'][-1]['frozen_position'] = .11001
    assert commanded_hold_evidence(**inputs, include_observation_roundoff=True)['status'] == 'NEEDS_CONTEXT'


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


@pytest.mark.parametrize('change', [None, 'owner_active', 'overlapping_active', 'unknown_active',
    'unknown_joints', 'description', 'mobile_root', 'missing_owner', 'duplicate_child'])
def test_fixed_root_hold_uses_bound_joint_ownership_without_requiring_unrelated_navigation_idle(change):
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence

    _, rid, state, model, physical = _commanded_hold_case()
    configuration = model['configuration']
    configuration['root'] = 'owner_base'
    description = ('<robot name="recorded"><link name="world"/><link name="owner_base"/>'
        '<joint name="mount" type="fixed"><parent link="world"/><child link="owner_base"/></joint>'
        '<joint name="slide" type="prismatic"><parent link="owner_base"/><child link="body"/></joint>'
        '<joint name="other_slide" type="prismatic"><parent link="world"/><child link="other_body"/></joint></robot>')
    controller = physical['controller_hold_observation']
    controller['controller'] = '/owner'
    activity = {'owner': deepcopy(controller),
        'other': {'joint_names': ['other_slide'], 'holding': False, 'has_active_goal': True, 'has_pending_goal': False},
        'base': {'navigation_enabled': True, 'holding': False, 'has_active_goal': True}}
    physical['controller_hold_activity'] = activity
    if change == 'owner_active':
        controller['has_active_goal'] = True
    elif change == 'overlapping_active':
        activity['other']['joint_names'] = ['slide']
    elif change == 'unknown_active':
        activity['unknown'] = {'has_active_goal': True}
    elif change == 'unknown_joints':
        activity['other']['joint_names'] = ['not_in_description']
    elif change == 'mobile_root':
        description = description.replace('name="mount" type="fixed"', 'name="mount" type="prismatic"')
    elif change == 'missing_owner':
        activity.pop('owner')
    elif change == 'duplicate_child':
        description = description.replace('child link="other_body"', 'child link="owner_base"')
    configuration['description_sha256'] = hashlib.sha256(description.encode()).hexdigest()
    if change == 'description':
        description += ' '
    original = deepcopy(physical)
    legacy = commanded_hold_evidence(resource_id=rid, state=state, model=model, physical=physical)
    result = commanded_hold_evidence(resource_id=rid, state=state, model=model, physical=physical,
                                     robot_description=description)
    assert legacy['status'] == 'NEEDS_CONTEXT'
    assert result['status'] == ('supported' if change is None else 'NEEDS_CONTEXT')
    assert physical == original
    if change is None:
        assert result['controller_activity_scope']['checked_controllers'] == ['owner']
        assert set(result['controller_activity_scope']['excluded_controllers']) == {'base', 'other'}
        assert result['observed_idle'] is False
        assert result['physical_execution_verified'] is False


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


def test_nominal_place_calculation_rejects_unmodeled_declared_adjustment(monkeypatch):
    from cais_spade_llm.resources.resource_safety_preparation import _outline_nominal_place_targets

    with pytest.raises(ValueError, match="arguments differ"):
        _outline_nominal_place_targets(task={"part_name": "part", "expected_end_state": {"resource_location": "destination"}},
            recovery_snapshot={}, physical_evidence={}, resource_evidence={"execution_mode": "simulation", "state": {"_task_ctx": {}}},
            params={"part_name": "part", "destination_location": "destination", "pick_ctx": {},
                    "product_geometry": {}, "assembly_board_v1_aruco": None, "z_adjustment_m": .02})


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


@pytest.mark.parametrize("upright", [True, False])
def test_support_place_target_rests_an_upright_cylinder_on_its_circular_face(upright):
    import math
    from cais_spade_llm.resources.robot.recovery_capability import _support_place_pose

    scene, _, _, task = _custody_event_case(False)
    scene["physics_evidence"]["models"]["square"]["joint_count"] = 0
    scene["physics_evidence"]["models"]["square"]["links"]["link"]["collisions"][0]["geometry"] = {
        "type": "cylinder", "radius": .02, "length": .1}
    part, destination = task["part_name"], "Buffer For Machined parts"
    # Lying on its side, the cylinder's axis is horizontal.
    tilt = 0. if upright else math.pi / 2
    scene["snapshot"]["parts"][part]["current_pose"] = [0., 0., .2, math.sin(tilt / 2), 0., 0., math.cos(tilt / 2)]
    scene["resources"][destination]["geometry_source"] = {"model": "table"}
    target = {"target_origin_pose": {"x": .1, "y": .2, "z": -.05}, "support_surface_height": -.05}
    if not upright:
        with pytest.raises(ValueError, match="upright"):
            _support_place_pose(part=part, destination=destination, target=target, physical_evidence=scene)
        return
    result = _support_place_pose(part=part, destination=destination, target=target, physical_evidence=scene)
    assert result["support_evidence"]["status"] == "supported"
    assert result["part_pose"][:3] == pytest.approx([.1, .2, 0.], abs=1e-12)


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


def test_robot_nominal_contract_preserves_required_arguments_and_optional_null_defaults():
    from cais_spade_llm.resources.robot.robot_task_recovery import robot_recovery_des_descriptor

    model = robot_recovery_des_descriptor(resource_jid="ur5e-3@localhost",
        snapshot={"current_state": "idle", "current_location": "home", "held_part": None},
        task_names=["place_insert"])
    params = model["events"][0]["composition_contract"]["params"]
    assert params["destination_location"] == {"type": "string", "required": True}
    assert params["part_name"] == {"type": "string", "required": True}
    assert params["product_geometry"] == {"type": "object", "required": False, "default": None}


def test_named_pose_projects_recorded_joints_without_claiming_motion_interval():
    case = _case()
    case["resource_models"]["ur5e-3"]["named_joint_poses"]["home"]["positions"] = [1]
    before = deepcopy(case)
    result = project_recovery_geometry(**case)
    assert result["status"] == "projected", result
    assert result["observations"][-1]["region_occupancy"]["assembly_board-v1"]["ur5e-3"] is True
    assert result["interval_complete"] is False
    assert case == before


def test_station_name_and_tcp_do_not_replace_missing_link_geometry():
    case = _case()
    case["resource_models"].pop("ur5e-4")
    result = project_recovery_geometry(**case)
    assert result["status"] == "needs_context"
    assert result["context_requests"] == ["/physical_evidence"]
    assert "component bounds" in result["unresolved"][0]


def test_joint_witness_requires_declared_stationary_peers_and_parts():
    case = _case()
    _motion(case, 0.2)
    case["step_evidence"][0].pop("stationary_parts")
    result = project_recovery_geometry(**case)
    assert result["status"] == "needs_context"
    assert result["requires_projection"] is True


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


@pytest.mark.parametrize("field", ["joint_names", "joint_positions"])
def test_missing_joint_facts_are_context_not_assumed_defaults(field):
    case = _case()
    case["snapshot"]["resources"]["ur5e-3"].pop(field)
    result = project_recovery_geometry(**case)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["execution_authorized"] is False


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
