from __future__ import annotations

"""Request/provenance and isolation checks for the planning-only preview client."""

import asyncio
import copy
import hashlib
import json
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from cais_spade_llm.recovery_framework.recovery_motion_preview import (
    bind_support_departure,
    capture_preview_scene,
    named_pose_joint_target,
    preview_recovery_motion,
    preview_result_record,
    validate_preview_request,
)


def support_departure_record():
    return {"status": "prepared", "version": 1, "resource_id": "ur5e-3", "part_name": "part",
        "command_sent": False, "execution_authorized": False, "pairs": [{
            "part_object_id": "part/link/collision", "support_object_id": "table/link/collision",
            "part_collision_id": "part::link::collision", "support_collision_id": "table::link::collision",
            "plane_point": [0., 0., 1.], "plane_normal": [0., 0., 1.],
            "initial_penetration_bound": 0.001, "source_contact_fingerprint": "contact",
            "geometry_fingerprint": "geometry"}]}


def test_support_departure_binding_preserves_exact_input_and_rejects_stale_requests():
    original, contact = request(), support_departure_record()
    saved = copy.deepcopy((original, contact))
    prepared = bind_support_departure(original, contact)
    assert hashlib.sha256(prepared["support_departure_json"].encode()).hexdigest() == prepared["support_departure_fingerprint"]
    assert (original, contact) == saved
    for field in ("capture_fingerprint", "proposal_fingerprint", "scene_fingerprint", "model_fingerprint"):
        changed = copy.deepcopy(prepared)
        changed[field] = "changed"
        with pytest.raises(ValueError, match="stale request bindings"):
            validate_preview_request(changed)
    prepared["support_departure_json"] += " "
    with pytest.raises(ValueError, match="fingerprint"):
        validate_preview_request(prepared)


@pytest.mark.parametrize("defect", ["empty", "duplicate", "normal", "missing_source", "negative_depth", "execution"])
def test_support_departure_requires_bounded_exact_nonexecuting_contact_evidence(defect):
    evidence = support_departure_record()
    if defect == "empty":
        evidence["pairs"] = []
    elif defect == "duplicate":
        evidence["pairs"].append(copy.deepcopy(evidence["pairs"][0]))
    elif defect == "normal":
        evidence["pairs"][0]["plane_normal"] = [0., 0., 2.]
    elif defect == "missing_source":
        evidence["pairs"][0].pop("source_contact_fingerprint")
    elif defect == "negative_depth":
        evidence["pairs"][0]["initial_penetration_bound"] = -1.
    else:
        evidence["execution_authorized"] = True
    with pytest.raises(ValueError):
        bind_support_departure(request(), evidence)


def test_support_departure_result_keeps_samples_separate_and_requires_separation():
    prepared = bind_support_departure(request(), support_departure_record())
    response = result()
    response["support_departure_fingerprint"] = prepared["support_departure_fingerprint"]
    sampled = {"pairs_complete": True, "continuous_collision_verified": False,
        "samples": [{"state_index": 0}], "pairs": [{"part_object_id": "part/link/collision",
            "support_object_id": "table/link/collision", "endpoint_separated": True}]}
    response["support_departure_evidence_json"] = json.dumps(sampled)
    record = preview_result_record(prepared, response)
    assert record["support_departure_evidence"] == sampled
    assert record["coverage"]["continuous_collision_verified"] is False
    assert record["execution_authorized"] is False
    sampled["pairs"][0]["endpoint_separated"] = False
    response["support_departure_evidence_json"] = json.dumps(sampled)
    with pytest.raises(ValueError, match="separation evidence"):
        preview_result_record(prepared, response)


def request() -> dict:
    state = {"is_diff": False, "joint_state": {"name": ["joint"], "position": [0.0]}}
    return {
        "planning_scene": {"is_diff": False, "robot_state": state},
        "start_state": copy.deepcopy(state),
        "group_name": "arm", "link_name": "tool", "frame_id": "world",
        "mode": "joint_target", "waypoints": [],
        "joint_target": {"name": ["joint"], "position": [0.1]},
        "resource_id": "ur5e-3", "capture_fingerprint": "capture",
        "proposal_fingerprint": "proposal", "scene_fingerprint": "scene",
        "model_fingerprint": "model", "max_translation_step": 0.01,
        "max_rotation_step": 0.01, "max_joint_step": 0.01,
        "velocity_scaling": 0.1, "acceleration_scaling": 0.1, "timeout_seconds": 10.0,
    }


def base_request():
    record = request()
    record.update(mode='base_trajectory', base_link='base', joint_target={})
    record['base_trajectory'] = {'joint_names': ['joint'], 'points': [
        {'positions': [0.], 'velocities': [.1], 'accelerations': [], 'effort': [],
         'time_from_start': {'sec': 0, 'nanosec': 0}},
        {'positions': [.6], 'velocities': [0.], 'accelerations': [], 'effort': [],
         'time_from_start': {'sec': 3, 'nanosec': 0}}]}
    return record


def test_base_preview_preserves_exact_motion_and_does_not_claim_continuous_or_execution_approval():
    original = base_request()
    before = copy.deepcopy(original)
    checked = validate_preview_request(original)
    response = result()
    response['trajectory']['joint_trajectory'] = copy.deepcopy(original['base_trajectory'])
    captured = preview_result_record(checked, response)
    assert captured['joint_trajectory'] == original['base_trajectory']
    assert captured['coverage'] == {'path_complete': True, 'continuous_collision_verified': False, 'cca_validated': False}
    assert not captured['execution_authorized'] and not captured['command_sent']
    assert original == before
    response['trajectory']['joint_trajectory']['points'][-1]['time_from_start']['sec'] = 4
    with pytest.raises(ValueError, match='timing'):
        preview_result_record(checked, response)


@pytest.mark.parametrize('omitted_from', ['request', 'response', 'both'])
def test_base_preview_accepts_ros_empty_array_defaults_without_mutating_evidence(omitted_from):
    original = base_request()
    response = result()
    response['trajectory']['joint_trajectory'] = copy.deepcopy(original['base_trajectory'])
    for side, trajectory in (('request', original['base_trajectory']),
                             ('response', response['trajectory']['joint_trajectory'])):
        if omitted_from in (side, 'both'):
            for point in trajectory['points']:
                point.pop('accelerations')
                point.pop('effort')
    before = copy.deepcopy((original, response))
    captured = preview_result_record(validate_preview_request(original), response)
    assert captured['joint_trajectory'] == response['trajectory']['joint_trajectory']
    assert (original, response) == before


@pytest.mark.parametrize('defect', ['velocity', 'acceleration', 'effort', 'frame', 'stamp'])
def test_base_preview_rejects_changed_derivatives_and_time_frame(defect):
    original = base_request()
    response = result()
    trajectory = response['trajectory']['joint_trajectory'] = copy.deepcopy(original['base_trajectory'])
    if defect in ('frame', 'stamp'):
        trajectory['header'] = {'frame_id': 'another_frame' if defect == 'frame' else '',
                                'stamp': {'sec': 1 if defect == 'stamp' else 0, 'nanosec': 0}}
    else:
        field = {'velocity': 'velocities', 'acceleration': 'accelerations', 'effort': 'effort'}[defect]
        trajectory['points'][-1][field] = [.1]
    with pytest.raises(ValueError):
        preview_result_record(validate_preview_request(original), response)


@pytest.mark.parametrize('defect', ['unknown_joint', 'start', 'velocity', 'nan', 'boolean', 'time',
                                  'other_mode', 'other_motion', 'base_link', 'acceleration'])
def test_base_preview_requires_complete_unchanged_owner_inputs(defect):
    record = base_request()
    point = record['base_trajectory']['points'][0]
    if defect == 'unknown_joint':
        record['base_trajectory']['joint_names'] = ['missing']
    elif defect == 'start':
        point['positions'][0] = .2
    elif defect == 'velocity':
        point['velocities'] = []
    elif defect == 'nan':
        point['velocities'][0] = float('nan')
    elif defect == 'boolean':
        point['positions'][0] = False
    elif defect == 'time':
        point['time_from_start']['sec'] = 1
    elif defect == 'other_mode':
        record['mode'] = 'scene_endpoint'
    elif defect == 'other_motion':
        record['joint_target'] = {'name': ['joint'], 'position': [.1]}
    elif defect == 'base_link':
        record['base_link'] = ''
    else:
        point['accelerations'] = [0.]
    with pytest.raises(ValueError):
        validate_preview_request(record)


def result() -> dict:
    return {
        "resource_id": "ur5e-3", "capture_fingerprint": "capture",
        "proposal_fingerprint": "proposal", "scene_fingerprint": "scene",
        "model_fingerprint": "model", "status": "PREVIEW_COMPLETE",
        "path_complete": True, "fraction": 1.0, "continuous_collision_verified": False,
        "trajectory": {"joint_trajectory": {"joint_names": ["joint"], "points": [
            {"positions": [0.0]}, {"positions": [0.1]},
        ]}},
    }


def scene_endpoint_pair():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_scene_endpoint

    original = request()
    original["planning_scene"]["world"] = {"collision_objects": [
        {"id": name, "primitives": [{"type": 1, "dimensions": [1., 1., 1.]}]}
        for name in ("part/link/collision", "support/link/collision")]}
    evidence = {"version": 1, "status": "prepared", "resource_id": original["resource_id"],
        "command_sent": False, "execution_authorized": False, "source": {"step_fingerprint": "declared-step"},
        "changed_object_ids": ["part/link/collision"], "support_pairs": [{
            "part_object_id": "part/link/collision", "support_object_id": "support/link/collision",
            "plane_point": [0., 0., .5], "plane_normal": [0., 0., 1.]}]}
    prepared = bind_scene_endpoint(original, evidence)
    binding = {key: prepared[key] for key in ("resource_id", "capture_fingerprint", "proposal_fingerprint", "scene_fingerprint", "model_fingerprint")}
    audit = {"version": 1, "source": "PreviewRecoveryMotion", "endpoint_only": True,
        "pairs_complete": True, "continuous_collision_verified": False, "request_binding": binding,
        "scene_endpoint_fingerprint": prepared["scene_endpoint_fingerprint"], "changed_object_ids": evidence["changed_object_ids"],
        "world_object_ids": ["part/link/collision", "support/link/collision"],
        "robot_collision_checked": True, "robot_collision_free": True, "robot_contacts": [], "checked_contacts": 1,
        "support_pairs": [{**evidence["support_pairs"][0], "minimum_signed_distance": 0.}],
        "checked_pairs": [{"body_1": "part/link/collision", "body_2": "support/link/collision", "shape_pairs": 1,
            "allowed_support_tangency": True, "contacts": [{"shape_1": 0, "shape_2": 0, "depth": 0.,
                "position": [0., 0., .5], "normal": [0., 0., 1.]}]}], "collision_free": True}
    response = {**result(), "status": "SCENE_ENDPOINT_COMPLETE", "path_complete": False, "fraction": 0.,
        "sampled_collision_rejected": False, "trajectory": {},
        "scene_endpoint_fingerprint": prepared["scene_endpoint_fingerprint"],
        "scene_endpoint_evidence_json": json.dumps(audit)}
    return original, evidence, prepared, response


def test_scene_endpoint_binding_is_detached_and_never_claims_motion():
    original, evidence, prepared, response = scene_endpoint_pair()
    assert original["mode"] == "joint_target" and "preview_binding" not in evidence
    checked = preview_result_record(prepared, response)
    assert checked["scene_endpoint_evidence"]["collision_free"] is True
    assert checked["path_complete"] is False
    assert checked["command_sent"] is False
    assert checked["execution_authorized"] is False
    for field in ("resource_id", "scene_fingerprint", "proposal_fingerprint", "model_fingerprint"):
        with pytest.raises(ValueError, match="stale request bindings"):
            validate_preview_request({**prepared, field: "different"})


@pytest.mark.parametrize("defect", ["missing_pair", "missing_shape", "missing_world", "stale", "depth", "distance",
                                  "motion", "parsed_only", "continuous", "changed_object", "robot_collision"])
def test_scene_endpoint_rejects_incomplete_or_forged_native_audits(defect):
    _, _, prepared, response = scene_endpoint_pair()
    audit = json.loads(response["scene_endpoint_evidence_json"])
    if defect == "missing_pair":
        audit["checked_pairs"] = []
    elif defect == "missing_shape":
        audit["checked_pairs"][0]["shape_pairs"] = 0
    elif defect == "missing_world":
        audit["world_object_ids"] = []
    elif defect == "stale":
        audit["request_binding"]["scene_fingerprint"] = "old"
    elif defect == "depth":
        audit["checked_pairs"][0]["contacts"][0]["depth"] = .000001
    elif defect == "distance":
        audit["support_pairs"][0]["minimum_signed_distance"] = -.000001
    elif defect == "motion":
        response["path_complete"] = True
    elif defect == "continuous":
        audit["continuous_collision_verified"] = True
    elif defect == "changed_object":
        audit["changed_object_ids"] = ["other"]
    elif defect == "robot_collision":
        audit["robot_collision_free"] = False
    response["scene_endpoint_evidence_json"] = "" if defect == "parsed_only" else json.dumps(audit)
    response["scene_endpoint_evidence"] = audit
    with pytest.raises(ValueError):
        preview_result_record(prepared, response)


def test_scene_endpoint_retains_complete_factual_rejection_separately_from_motion():
    _, _, prepared, response = scene_endpoint_pair()
    audit = json.loads(response["scene_endpoint_evidence_json"])
    audit["support_pairs"][0]["minimum_signed_distance"] = -.000001
    audit["checked_pairs"][0]["contacts"][0]["depth"] = .000001
    audit["collision_free"] = False
    response.update(status="SCENE_ENDPOINT_REJECTED", scene_endpoint_evidence_json=json.dumps(audit))
    parsed = preview_result_record(prepared, response)
    assert parsed["scene_endpoint_evidence"]["collision_free"] is False
    assert parsed["sampled_collision_rejected"] is False


def test_scene_endpoint_missing_native_capture_cannot_reuse_parsed_audit():
    _, _, prepared, response = scene_endpoint_pair()
    response.update(status="NEEDS_CONTEXT", scene_endpoint_evidence=json.loads(response["scene_endpoint_evidence_json"]),
                    scene_endpoint_evidence_json="")
    assert "scene_endpoint_evidence" not in preview_result_record(prepared, response)


def route_calculation():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import prepare_cartesian_route_search

    original, response = cartesian_failure()
    capabilities = {"gripper_reach": {"frame": "world", "origin_pose": {"x": 0., "y": 0., "z": 0.},
        "max_xy_radius_m": 1., "z_min_m": -1., "z_max_m": 1., "tolerance_m": 0.}}
    prepared = prepare_cartesian_route_search(original, capabilities)
    intermediate = copy.deepcopy(original["waypoints"][0])
    intermediate["position"].update(x=.05, y=.05)
    candidate = {"version": 1, "solved": True,
        "route_search": json.loads(prepared["route_search_json"]),
        "request_binding": {key: prepared[key] for key in (
            "resource_id", "capture_fingerprint", "proposal_fingerprint", "scene_fingerprint", "model_fingerprint")},
        "original_waypoints": copy.deepcopy(original["waypoints"]),
        "derived_waypoints": [intermediate, *copy.deepcopy(original["waypoints"])],
        "mandatory_waypoint_indices": [1], "source_state": copy.deepcopy(original["start_state"]),
        "edges": [{"ik_queries": [{"seed": [0.], "success": True}]}],
        "command_sent": False, "execution_authorized": False}
    response.pop("cartesian_diagnostics_json")
    response.update(status="ROUTE_CANDIDATE", fraction=0., route_candidate_json=json.dumps(candidate))
    return prepared, response, capabilities


def test_calculated_route_is_not_motion_approval_and_archives_raw_once(tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import recovery_motion_preview as preview

    monkeypatch.setattr(preview, "_ROUTE_RECORDS", tmp_path)
    monkeypatch.setattr(preview, "_new_node", lambda: pytest.fail("Offline route evidence called ROS"))
    prepared, response, _ = route_calculation()
    original = copy.deepcopy((prepared, response))
    checked = preview_result_record(prepared, response)
    assert not checked["coverage"]["path_complete"] and not checked["execution_authorized"]
    assert checked["joint_trajectory"] == {}
    assert "edges" not in checked["route_candidate"]
    proof = preview.archive_cartesian_route(prepared, response)
    assert preview.read_cartesian_route(proof) == prepared
    assert '"edges":' not in json.dumps(proof)
    artifact = next(tmp_path.glob("*/calculation.json"))
    assert json.loads(artifact.read_text())["result"]["route_candidate_json"] == response["route_candidate_json"]
    assert (prepared, response) == original
    artifact.write_text(artifact.read_text() + " ")
    with pytest.raises(ValueError, match="artifact changed"):
        preview.read_cartesian_route(proof)


@pytest.mark.parametrize("defect", ["mandatory_endpoint", "mandatory_order", "orientation", "reach", "start",
    "scene", "settings", "trajectory", "approved", "execution", "missing_raw", "unexpected"])
def test_calculated_route_rejects_changed_source_targets_or_approval(defect):
    prepared, response, _ = route_calculation()
    candidate = json.loads(response["route_candidate_json"])
    if defect == "mandatory_endpoint":
        candidate["derived_waypoints"][-1]["position"]["x"] += .1
    elif defect == "mandatory_order":
        candidate["mandatory_waypoint_indices"] = [0]
    elif defect == "orientation":
        candidate["derived_waypoints"][0]["orientation"] = {"x": 0., "y": 0., "z": 1., "w": 0.}
    elif defect == "reach":
        candidate["derived_waypoints"][0]["position"]["x"] = 2.
    elif defect == "start":
        candidate["source_state"]["joint_state"]["position"] = [.1]
    elif defect == "scene":
        candidate["request_binding"]["scene_fingerprint"] = "changed"
    elif defect == "settings":
        candidate["route_search"]["maximum_edges"] = 129
    elif defect == "trajectory":
        response["trajectory"] = result()["trajectory"]
    elif defect == "approved":
        response.update(status="PREVIEW_COMPLETE", path_complete=True, fraction=1.)
    elif defect == "execution":
        candidate["execution_authorized"] = True
    elif defect == "unexpected":
        prepared.pop("route_search_json")
    response["route_candidate_json"] = "" if defect == "missing_raw" else json.dumps(candidate)
    with pytest.raises(ValueError):
        preview_result_record(prepared, response)


@pytest.mark.parametrize("defect", ["budget", "seconds", "seed", "mode", "support"])
def test_route_search_keeps_bounded_nonexecuting_motion_contract(defect):
    prepared, _, _ = route_calculation()
    config = json.loads(prepared["route_search_json"])
    if defect == "budget":
        config["maximum_edges"] = 129
    elif defect == "seconds":
        config["maximum_seconds"] = 121
    elif defect == "seed":
        config["random_seed"] = -1
    elif defect == "mode":
        prepared.update(mode="joint_target", waypoints=[], joint_target={"name": ["joint"], "position": [0.]})
    else:
        prepared.pop("route_search_json")
        prepared = bind_support_departure(prepared, support_departure_record())
    prepared["route_search_json"] = json.dumps(config)
    with pytest.raises(ValueError):
        validate_preview_request(prepared)


def test_calculated_route_binds_recorded_root_model_before_strict_replay():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import recorded_root_model_fingerprint

    prepared, response, _ = route_calculation()
    root = {"header": {"frame_id": "world"}, "child_frame_id": "owner_base", "transform": {
        "translation": {"x": 0., "y": 0., "z": 0.}, "rotation": {"x": 0., "y": 0., "z": 0., "w": 1.}}}
    prepared["recorded_root_transforms"] = [root]
    with pytest.raises(ValueError, match="derived_model_fingerprint"):
        preview_result_record(prepared, response)
    response["derived_model_fingerprint"] = recorded_root_model_fingerprint("model", [root])
    assert preview_result_record(prepared, response)["route_candidate"]


@pytest.mark.parametrize("raw", ["missing", "unresolved"])
def test_route_candidate_is_reconstructed_only_from_solved_raw_capture(raw):
    prepared, response, _ = route_calculation()
    response["route_candidate"] = json.loads(response["route_candidate_json"])
    response["status"] = "NEEDS_CONTEXT"
    if raw == "missing":
        response.pop("route_candidate_json")
    else:
        candidate = json.loads(response["route_candidate_json"])
        candidate["solved"] = False
        response["route_candidate_json"] = json.dumps(candidate)
    assert "route_candidate" not in preview_result_record(prepared, response)


def test_route_calculation_cannot_insert_a_detour_between_mandatory_endpoints():
    prepared, response, _ = route_calculation()
    candidate = json.loads(response["route_candidate_json"])
    second = copy.deepcopy(prepared["waypoints"][0])
    second["position"]["z"] = -.1
    prepared["waypoints"].append(second)
    candidate["original_waypoints"] = copy.deepcopy(prepared["waypoints"])
    candidate["derived_waypoints"].extend([copy.deepcopy(candidate["derived_waypoints"][0]), second])
    candidate["mandatory_waypoint_indices"] = [1, 3]
    response["route_candidate_json"] = json.dumps(candidate)
    with pytest.raises(ValueError, match="mandatory Cartesian endpoints"):
        preview_result_record(prepared, response)


def cartesian_failure():
    """Return bound diagnostics without upgrading a partial motion result."""
    prepared, response = request(), result()
    prepared.update(mode="cartesian", joint_target={"name": [], "position": []}, waypoints=[{
        "position": {"x": .1, "y": 0., "z": 0.}, "orientation": {"x": 0., "y": 0., "z": 0., "w": 1.}}])
    response.update(status="NEEDS_CONTEXT", path_complete=False, fraction=.4, trajectory={})
    fields = ("resource_id", "capture_fingerprint", "proposal_fingerprint", "scene_fingerprint", "model_fingerprint")
    diagnostic = {"version": 1, "source": "PreviewRecoveryMotion", "diagnostic_only": True,
        "complete_inputs": True, "failure_kind": "ik_search_failed", "fraction": .4,
        "request_binding": {key: prepared[key] for key in fields},
        **{key: prepared[key] for key in ("group_name", "link_name", "frame_id", "max_translation_step",
                                        "max_rotation_step", "max_joint_step")},
        "ik_queries": [{"index": 0, "success": False, "error_code": -6, "timeout_seconds": .05,
                        "seed": [0.], "consistency_limits": [.01]}],
        "waypoints": [{"index": 0, "query_start": 0, "query_end": 1, "fraction": .4,
                       "jump_check_truncated": False, "last_accepted_state": {"joint_state": prepared["start_state"]["joint_state"]}}]}
    response["cartesian_diagnostics_json"] = json.dumps(diagnostic)
    return prepared, response


def test_cartesian_failure_preserves_solver_diagnostics_without_motion_approval():
    prepared, response = cartesian_failure()
    record = preview_result_record(prepared, response)
    assert record["cartesian_failure_evidence"] == json.loads(response["cartesian_diagnostics_json"])
    assert record["status"] == "NEEDS_CONTEXT"
    assert not record["coverage"]["path_complete"]
    assert not record["execution_authorized"]
    assert not record["command_sent"]


def test_cartesian_derived_fields_require_authoritative_native_capture():
    prepared, response = cartesian_failure()
    response["cartesian_failure_evidence"] = {"complete_inputs": True, "failure_kind": "joint_jump"}
    response["cartesian_diagnostics"] = copy.deepcopy(response["cartesian_failure_evidence"])
    record = preview_result_record(prepared, response)
    assert record["cartesian_failure_evidence"] == json.loads(response["cartesian_diagnostics_json"])
    response.pop("cartesian_diagnostics_json")
    record = preview_result_record(prepared, response)
    assert "cartesian_failure_evidence" not in record
    assert "cartesian_diagnostics" not in record


@pytest.mark.parametrize("defect", ["binding", "limits", "complete", "query", "fraction", "index", "jump", "mode"])
def test_cartesian_failure_rejects_inconsistent_diagnostics(defect):
    prepared, response = cartesian_failure()
    diagnostic = json.loads(response["cartesian_diagnostics_json"])
    if defect == "binding":
        diagnostic["request_binding"]["scene_fingerprint"] = "changed"
    elif defect == "limits":
        diagnostic["max_joint_step"] = 1.
    elif defect == "complete":
        response.update(result())
    elif defect == "query":
        diagnostic["ik_queries"][-1]["success"] = True
    elif defect == "fraction":
        diagnostic["fraction"] = 1.
    elif defect == "index":
        diagnostic["waypoints"][-1]["query_end"] = 8
    elif defect == "jump":
        diagnostic["failure_kind"] = "joint_jump"
    else:
        prepared["mode"] = "joint_target"
    response["cartesian_diagnostics_json"] = json.dumps(diagnostic)
    with pytest.raises(ValueError, match="Cartesian"):
        preview_result_record(prepared, response)


def collision_result(prepared=None, kind="unrelated_collision"):
    """Return a complete failed native sample with exact request provenance."""
    prepared = prepared or request()
    response = result()
    response.update(status="NEEDS_CONTEXT", path_complete=False, fraction=0., sampled_collision_rejected=True)
    pair = {"body_1": "tool", "body_2": "obstacle", "contacts": [
        {"depth": .01, "position": [0., 0., 0.], "normal": [1., 0., 0.]}]}
    support_pairs = []
    if kind != "unrelated_collision":
        pair.update(body_1="part/link/collision", body_2="table/link/collision")
        support_pairs = [{"part_object_id": pair["body_1"], "support_object_id": pair["body_2"],
            "initial_penetration_bound": .001, "minimum_signed_distance": -.01 if kind == "support_distance" else 0.,
            "previously_separated": kind == "support_reentry"}]
        response["support_departure_fingerprint"] = prepared["support_departure_fingerprint"]
    evidence = {"version": 1, "source": "PreviewRecoveryMotion", "phase": "joint_target", "state_index": 3,
        "kind": kind, "reason": "Recorded collision", "contacts_complete": True, "checked_contacts": 1,
        "contacts": [pair], "support_pairs": support_pairs, "continuous_collision_verified": False,
        "joint_state": {"name": ["joint"], "position": [.03]}, "request_binding": {
            **{field: prepared[field] for field in ("resource_id", "capture_fingerprint", "proposal_fingerprint",
                "scene_fingerprint", "model_fingerprint")},
            "support_departure_fingerprint": prepared.get("support_departure_fingerprint", "")}}
    response["collision_rejection_evidence_json"] = json.dumps(evidence)
    return response


@pytest.mark.parametrize("kind", ["unrelated_collision", "support_depth", "support_distance", "support_reentry"])
def test_collision_rejection_preserves_failed_configuration_and_complete_contacts(kind):
    prepared = request() if kind == "unrelated_collision" else bind_support_departure(request(), support_departure_record())
    response = collision_result(prepared, kind)
    original = copy.deepcopy(response)
    record = preview_result_record(prepared, response)
    assert record["collision_rejection_evidence"] == json.loads(response["collision_rejection_evidence_json"])
    assert response == original
    assert record["coverage"]["path_complete"] is False
    assert record["execution_authorized"] is False


@pytest.mark.parametrize("defect", ["missing", "binding", "incomplete", "count", "joint", "depth", "geometry",
    "phase", "unsupported_kind", "not_proven", "complete", "unexpected"])
def test_collision_rejection_requires_exact_failed_sample(defect):
    prepared, response = request(), collision_result()
    evidence = json.loads(response["collision_rejection_evidence_json"])
    if defect == "binding":
        evidence["request_binding"]["scene_fingerprint"] = "another scene"
    elif defect == "incomplete":
        evidence["contacts_complete"] = False
    elif defect == "count":
        evidence["checked_contacts"] = 4096
    elif defect == "joint":
        evidence["joint_state"]["name"] = ["another_joint"]
    elif defect == "depth":
        evidence["contacts"][0]["contacts"][0]["depth"] = -1.
    elif defect == "geometry":
        evidence["contacts"][0]["contacts"][0]["position"] = [0.]
    elif defect == "phase":
        evidence["phase"] = "unknown"
    elif defect == "unsupported_kind":
        evidence["kind"] = "unknown"
    elif defect == "not_proven":
        evidence.update(contacts=[], checked_contacts=0)
    elif defect == "complete":
        response.update(status="PREVIEW_COMPLETE", path_complete=True, fraction=1.)
    elif defect == "unexpected":
        response["sampled_collision_rejected"] = False
    response["collision_rejection_evidence_json"] = "" if defect == "missing" else json.dumps(evidence)
    with pytest.raises(ValueError):
        preview_result_record(prepared, response)


def test_collision_rejection_does_not_swap_part_and_support_identifiers():
    prepared = bind_support_departure(request(), support_departure_record())
    response = collision_result(prepared, "support_distance")
    evidence = json.loads(response["collision_rejection_evidence_json"])
    pair = evidence["support_pairs"][0]
    pair["part_object_id"], pair["support_object_id"] = pair["support_object_id"], pair["part_object_id"]
    response["collision_rejection_evidence_json"] = json.dumps(evidence)
    with pytest.raises(ValueError, match="conflicting support"):
        preview_result_record(prepared, response)


def test_validation_and_record_preserve_inputs_and_do_not_claim_safety():
    original = request()
    saved = copy.deepcopy(original)
    prepared = validate_preview_request(original)
    prepared["start_state"]["joint_state"]["position"][0] = 5.0
    assert original == saved
    record = preview_result_record(original, result())
    assert record["execution_authorized"] is False
    assert record["command_sent"] is False
    assert record["coverage"] == {
        "path_complete": True, "continuous_collision_verified": False, "cca_validated": False,
    }
    record["request"]["joint_target"]["position"][0] = 99.0
    assert original == saved


def test_preview_record_revalidation_preserves_provider_fields_and_detaches_callers():
    original = request()
    response = result()
    response["provider_metadata"] = {"notes": ["exact\nprovider text é"], "values": [None, False]}
    response["provider_text"] = '{"part_name":null,"text":"\\n"}'
    saved_request, saved_response = copy.deepcopy((original, response))

    first = preview_result_record(original, response)
    second = preview_result_record(original, first)

    assert second == first
    assert list(second) == list(first)
    assert second["provider_metadata"] == response["provider_metadata"]
    assert second["provider_text"] == response["provider_text"]
    first["provider_metadata"]["notes"].append("changed")
    first["trajectory"]["joint_trajectory"]["points"][0]["positions"][0] = 99.0
    first["joint_trajectory"]["points"][1]["positions"][0] = 99.0
    first["request"]["start_state"]["joint_state"]["position"][0] = 99.0
    assert (original, response) == (saved_request, saved_response)
    assert second == preview_result_record(saved_request, saved_response)
    assert second["trajectory"]["joint_trajectory"] is not second["joint_trajectory"]


@pytest.mark.parametrize("field", [
    "request", "joint_trajectory", "source", "coverage", "execution_authorized", "command_sent",
    "cartesian_diagnostics", "cartesian_failure_evidence", "route_candidate",
    "mating_contact_evidence", "scene_endpoint_evidence",
])
def test_preview_record_does_not_copy_discarded_enrichment(field):
    class DiscardedEnrichment:
        def __deepcopy__(self, memo):
            raise AssertionError("Discarded enrichment must not be copied during revalidation")

    original = request()
    expected = preview_result_record(original, result())
    enriched = dict(expected)
    enriched[field] = DiscardedEnrichment()

    checked = preview_result_record(original, enriched)

    assert checked == expected
    assert list(checked) == list(expected)


def test_preview_record_revalidation_checks_raw_evidence_despite_existing_enrichment():
    original = request()
    enriched = preview_result_record(original, result())
    enriched["request"]["proposal_fingerprint"] = "untrusted replacement"
    checked = preview_result_record(original, enriched)
    assert checked["request"] == original
    enriched["proposal_fingerprint"] = "stale raw result"
    with pytest.raises(ValueError, match="proposal_fingerprint differs"):
        preview_result_record(original, enriched)
    enriched["proposal_fingerprint"] = original["proposal_fingerprint"]
    enriched["trajectory"]["joint_trajectory"]["points"] = []
    with pytest.raises(ValueError, match="omitted its trajectory"):
        preview_result_record(original, enriched)


def test_preview_record_preserves_numerical_precision_separately_from_safety():
    response = result()
    fields = {"cartesian_position_precision": 1e-13, "cartesian_orientation_precision": 1e-13,
              "cartesian_position_residual": 1e-14, "cartesian_orientation_residual": 2e-15}
    response.update(fields)
    record = preview_result_record(request(), response)
    assert {field: record[field] for field in fields} == fields
    assert record["coverage"]["continuous_collision_verified"] is False
    assert record["coverage"]["cca_validated"] is False
    assert record["execution_authorized"] is False


def _recorded_root_case():
    configuration = {"root": "owner_base", "root_xyz": [.000002, -.000003, .500001],
                     "root_quaternion": [0., 0., 0., 1.], "description_sha256": "recorded-description"}
    capture = {"model_fingerprint": "model", "robot_description": '<robot name="r"><link name="world"/><link name="owner_base"/></robot>'}
    evidence = {"snapshot": {"frame": "world"},
        "resource_models": {"owner": {"configuration": copy.deepcopy(configuration)}},
        "resources": {"owner": {"continuous_motion_configuration": copy.deepcopy(configuration),
            "native_physics_source": {"instance_id": "recorded", "simulation_time": 4.}}}}
    return capture, evidence


def test_recorded_roots_preserve_measured_mount_and_separate_source_model():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import prepare_recorded_root_transforms, recorded_root_model_fingerprint

    capture, evidence = _recorded_root_case()
    original = copy.deepcopy((capture, evidence))
    prepared = prepare_recorded_root_transforms(capture, evidence)
    roots = prepared["recorded_root_transforms"]
    assert roots[0]["child_frame_id"] == "owner_base"
    assert list(roots[0]["transform"]["translation"].values()) == [.000002, -.000003, .500001]
    assert prepared["evidence"]["source_model_fingerprint"] == "model"
    assert prepared["evidence"]["resources"]["owner"]["native_physics_source"]["instance_id"] == "recorded"
    proposal = {**request(), "recorded_root_transforms": roots}
    assert validate_preview_request(proposal) == proposal
    response = {**result(), "derived_model_fingerprint": recorded_root_model_fingerprint("model", roots)}
    assert preview_result_record(proposal, response)["model_fingerprint"] == "model"
    response["derived_model_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="derived_model_fingerprint"):
        preview_result_record(proposal, response)
    assert (capture, evidence) == original
    assert json.loads(json.dumps(prepared)) == prepared


@pytest.mark.parametrize("defect", ["root", "pose", "duplicate", "missing_source"])
def test_recorded_roots_reject_conflicting_or_incomplete_owner_models(defect):
    from cais_spade_llm.recovery_framework.recovery_motion_preview import prepare_recorded_root_transforms

    capture, evidence = _recorded_root_case()
    if defect == "root":
        evidence["resource_models"]["owner"]["configuration"]["root"] = "absent"
    elif defect == "pose":
        evidence["resource_models"]["owner"]["configuration"]["root_xyz"][0] += .01
    elif defect == "duplicate":
        evidence["resource_models"]["other"] = copy.deepcopy(evidence["resource_models"]["owner"])
    else:
        evidence["resources"]["owner"]["continuous_motion_configuration"] = None
    with pytest.raises(ValueError):
        prepare_recorded_root_transforms(capture, evidence)


@pytest.mark.parametrize("field", [
    "resource_id", "capture_fingerprint", "proposal_fingerprint", "scene_fingerprint", "model_fingerprint",
])
def test_stale_result_rejected(field):
    response = result()
    response[field] = "changed"
    with pytest.raises(ValueError, match="Stale preview"):
        preview_result_record(request(), response)


@pytest.mark.parametrize("change", [
    {"path_complete": False}, {"fraction": 0.9}, {"trajectory": {}},
    {"continuous_collision_verified": True},
])
def test_partial_missing_or_overclaimed_preview_rejected(change):
    response = result()
    response.update(change)
    with pytest.raises(ValueError):
        preview_result_record(request(), response)


def test_complete_scene_and_exact_start_required():
    value = request()
    value["planning_scene"]["is_diff"] = True
    with pytest.raises(ValueError, match="is_diff"):
        validate_preview_request(value)
    value = request()
    value["start_state"]["joint_state"]["position"][0] = 0.5
    with pytest.raises(ValueError, match="differs"):
        validate_preview_request(value)


@pytest.mark.parametrize("field", [
    "max_translation_step", "max_rotation_step", "max_joint_step",
    "velocity_scaling", "acceleration_scaling", "timeout_seconds",
])
def test_missing_motion_settings_never_get_defaults(field):
    value = request()
    del value[field]
    with pytest.raises(ValueError, match=field):
        validate_preview_request(value)


def test_execution_fields_and_mixed_targets_rejected():
    value = request()
    value["execute"] = True
    with pytest.raises(ValueError, match="Unsupported preview fields"):
        validate_preview_request(value)
    value = request()
    value["waypoints"] = [{"position": {"x": 0.0}}]
    with pytest.raises(ValueError, match="Joint preview"):
        validate_preview_request(value)


def test_hardware_requests_rejected_before_any_ros_import():
    with patch.dict(sys.modules, {"rclpy": None, "cais_lab_robotics.action": None}):
        with pytest.raises(ValueError, match="simulation"):
            asyncio.run(preview_recovery_motion(request(), execution_mode="hardware"))
        with pytest.raises(ValueError, match="simulation"):
            asyncio.run(capture_preview_scene(execution_mode="hardware"))
        assert validate_preview_request(request())["resource_id"] == "ur5e-3"


def test_failed_preview_remains_unresolved_and_retains_partial_evidence():
    response = result()
    response.update(status="NEEDS_CONTEXT", path_complete=False, fraction=0.2, reason="partial path")
    record = preview_result_record(request(), response)
    assert record["status"] == "NEEDS_CONTEXT"
    assert record["coverage"]["path_complete"] is False
    assert record["reason"] == "partial path"
    assert record["joint_trajectory"] == response["trajectory"]["joint_trajectory"]


def test_named_pose_uses_exact_group_and_declared_values():
    srdf = '<robot><group_state name="home" group="ur5e_3_ur_manipulator"><joint name="ur5e_3_joint" value="0.31"/></group_state></robot>'
    assert named_pose_joint_target(srdf, group_name="ur5e_3_ur_manipulator", named_pose="home") == {
        "name": ["ur5e_3_joint"], "position": [0.31],
    }
    with pytest.raises(ValueError, match="exact SRDF"):
        named_pose_joint_target(srdf, group_name="ur5e_4_ur_manipulator", named_pose="home")
    with pytest.raises(ValueError, match="unsupported joint"):
        named_pose_joint_target(srdf.replace('value="0.31"', 'value=""'), group_name="ur5e_3_ur_manipulator", named_pose="home")


def test_collision_identities_use_recorded_model_declarations_and_preserve_sources(tmp_path):
    from cais_spade_llm.recovery_framework.geometry import collision_boxes
    from cais_spade_llm.recovery_framework.part_collision import observed_collision_identities

    world = tmp_path / 'world.sdf'
    world.write_text('<sdf><world><include><name>gear_small</name><uri>model://gear_asset</uri></include></world></sdf>')
    models = tmp_path / 'models'
    model = models / 'gear_asset' / 'model.sdf'
    model.parent.mkdir(parents=True)
    model.write_text('<sdf><model name="asset"><link name="link"><collision name="collision">'
                     '<geometry><box><size>0.1 0.2 0.3</size></box></geometry>'
                     '</collision></link></model></sdf>')
    observed = {'name': 'gear_small', 'frame': 'world', 'pose': [1., 2., 3., 0., 0., 0., 1.],
                'source': 'GETRECOVERYSTATE', 'instance_id': 'physics', 'simulation_time': 12.}
    original = copy.deepcopy(observed)
    identities = observed_collision_identities('gear_small', observed, world_path=world, models_path=models)
    actual = collision_boxes(world, models, {'gear_small': observed['pose']}, exact_models={'gear_small'})
    assert [row['id'] for row in identities['collision_objects']] == [row['id'] for row in actual if row['id'] != 'ground_plane']
    assert identities['observation'] == original
    assert identities['collision_scene_acknowledged'] is False
    assert identities['source_files'] == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (world, model)}
    identities['observation']['pose'][0] = 5.
    assert observed == original
    with pytest.raises(ValueError, match='measured model identity'):
        observed_collision_identities('unknown', observed, world_path=world, models_path=models)
    with pytest.raises(ValueError, match='exact collision producer'):
        observed_collision_identities('unknown', dict(observed, name='unknown'), world_path=world, models_path=models)


def projection_fixture():
    pose = lambda x: {"position": {"x": x, "y": 0., "z": 0.}, "orientation": {"x": 0., "y": 0., "z": 0., "w": 1.}}
    obj = {"id": "captured-42", "header": {"frame_id": "world"}, "operation": 0,
           "primitives": [{"type": 1, "dimensions": [0.1, 0.1, 0.1]}], "primitive_poses": [pose(1.2)]}
    scene = {"planning_scene": {"is_diff": False, "robot_state": {
        "is_diff": False, "joint_state": {"name": ["slider", "peer"], "position": [0., 1.5]},
        "attached_collision_objects": [],
    }, "world": {"collision_objects": [obj, dict(obj, id="unrelated-object")]}}, "scene_fingerprint": "captured"}
    physical = {"snapshot": {"frame": "world", "resources": {
        "ur5e-3": {"joint_names": ["slider"], "joint_positions": [0.], "current_pose": [0., 0., 0., 0., 0., 0., 1.], "held_part": None},
        "ur5e-4": {"joint_names": ["peer"], "joint_positions": [1.5], "current_pose": [1.5, 0., 0., 0., 0., 0., 1.], "held_part": None},
    }, "parts": {"gear_small": {"model_name": "gear_small", "current_pose": [1., 0., 0., 0., 0., 0., 1.], "contained_by": None}}},
        "resource_models": {
            "ur5e-3": {"configuration": {"joint_names": ["slider"], "reference_link": "tool", "joints": []}},
            "ur5e-4": {"configuration": {"joint_names": ["peer"], "reference_link": "peer_tool", "joints": []}},
        }, "provenance": {"source": "observed", "revision": 1},
    }
    payload = {"ur5e-3": {"model_name": "gear_small", "collision_objects": [{"id": "captured-42"}]}}
    targets = {"ur5e-3": {"link_name": "tool", "touch_links": ["tool"]}}
    return scene, physical, payload, targets


def test_scene_object_binding_owner_records_survive_exact_json_roundtrip():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import build_scene_object_bindings

    scene, physical, payload, targets = projection_fixture()
    payload["parts/gear_small"] = copy.deepcopy(payload["ur5e-3"])
    before = copy.deepcopy((scene, physical, payload, targets))
    bindings = build_scene_object_bindings(scene, physical, payload_evidence=payload, attachment_targets=targets)
    saved = json.loads(json.dumps(bindings))
    rebuilt = build_scene_object_bindings(scene, physical, payload_evidence=payload, attachment_targets=targets)

    assert bindings == saved == rebuilt
    assert bindings["gear_small"]["source"]["owner_records"] == [
        ["ur5e-3", payload["ur5e-3"]], ["parts/gear_small", payload["parts/gear_small"]],
    ]
    assert (scene, physical, payload, targets) == before
    changed = copy.deepcopy(saved)
    changed["gear_small"]["source"]["owner_records"][0][1]["model_name"] = "another_part"
    assert changed != rebuilt


def test_lumped_attachment_link_uses_only_captured_fixed_chain():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import build_scene_object_bindings

    scene, physical, payload, targets = projection_fixture()
    targets['ur5e-3']['link_name'] = 'tcp'
    targets['ur5e-3']['touch_links'] = ['tcp']
    configuration = physical['resource_models']['ur5e-3']['configuration']
    configuration.update(description_sha256='declared-robot', joints=[{
        'name': 'tcp_joint', 'parent': 'tool', 'child': 'tcp', 'type': 'fixed',
        'xyz': [0., 0., .218], 'rpy': [0., 0., 0.],
    }])
    saved = copy.deepcopy(physical)
    bindings = build_scene_object_bindings(scene, physical, payload_evidence=payload, attachment_targets=targets)
    attachment = bindings['gear_small']['attachments']['ur5e-3']
    assert attachment['link_to_reference'] == [0., 0., -.218, 0., 0., 0., 1.]
    assert attachment['source']['fixed_link_calculation']['source']['description_sha256'] == 'declared-robot'
    assert physical == saved
    configuration['joints'][0]['type'] = 'revolute'
    with pytest.raises(ValueError, match='fully declared fixed'):
        build_scene_object_bindings(scene, physical, payload_evidence=payload, attachment_targets=targets)


def test_projected_scene_retains_peers_and_changes_custody_only_in_copy():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import build_scene_object_bindings, project_preview_scene

    scene, physical, payload, targets = projection_fixture()
    original = copy.deepcopy(scene)
    physical["scene_object_bindings"] = build_scene_object_bindings(scene, physical, payload_evidence=payload, attachment_targets=targets)
    physical["snapshot"]["resources"]["ur5e-3"].update(
        joint_positions=[2.], current_pose=[2., 0., 0., 0., 0., 0., 1.],
        held_part="gear_small", grasp_transform=[0.1, 0., 0., 0., 0., 0., 1.],
    )
    physical["snapshot"]["parts"]["gear_small"]["current_pose"][0] = 2.1
    projected = project_preview_scene(scene, physical)
    assert projected["start_state"]["joint_state"]["position"] == [2., 1.5]
    attached = projected["start_state"]["attached_collision_objects"][0]
    assert attached["object"]["id"] == "captured-42"
    assert attached["object"]["primitive_poses"][0]["position"]["x"] == pytest.approx(0.3)
    assert [item["id"] for item in projected["planning_scene"]["world"]["collision_objects"]] == ["unrelated-object"]
    assert scene == original
    physical["snapshot"]["resources"]["ur5e-3"]["held_part"] = None
    physical["snapshot"]["parts"]["gear_small"]["current_pose"][0] = 3.
    deposited = project_preview_scene(projected, physical)
    assert deposited["start_state"]["attached_collision_objects"] == []
    obj = deposited["planning_scene"]["world"]["collision_objects"][-1]
    assert obj["primitive_poses"][0]["position"]["x"] == pytest.approx(3.2)
    assert scene == original


def test_scene_binding_includes_collision_object_pose_without_applying_it_twice():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import build_scene_object_bindings, project_preview_scene

    scene, physical, payload, targets = projection_fixture()
    obj = scene["planning_scene"]["world"]["collision_objects"][0]
    obj["pose"] = copy.deepcopy(obj["primitive_poses"][0])
    obj["primitive_poses"][0]["position"]["x"] = 0.0
    original = copy.deepcopy(scene)
    physical["scene_object_bindings"] = build_scene_object_bindings(
        scene, physical, payload_evidence=payload, attachment_targets=targets)
    local = physical["scene_object_bindings"]["gear_small"]["objects"][0]["part_local_object"]
    assert local["pose"]["position"]["x"] == 0.0
    assert local["primitive_poses"][0]["position"]["x"] == pytest.approx(0.2)
    physical["snapshot"]["parts"]["gear_small"]["current_pose"][0] = 3.0
    projected = project_preview_scene(scene, physical)
    moved = projected["planning_scene"]["world"]["collision_objects"][-1]
    assert moved["pose"]["position"]["x"] == 0.0
    assert moved["primitive_poses"][0]["position"]["x"] == pytest.approx(3.2)
    assert scene == original


def test_saved_native_collision_replaces_stale_scene_pose_and_contact_allowance():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import build_scene_object_bindings, project_preview_scene

    scene, physical, payload, targets = projection_fixture()
    original = copy.deepcopy(scene)
    payload["ur5e-3"]["collision_objects"][0].update(model="gear_small", link="link", collision="collision")
    physical["physics_evidence"] = {"instance_id": "captured", "simulation_time": 3., "models": {
        "gear_small": {"pose": [1., 0., 0., 0., 0., 0., 1.], "links": {"link": {"collisions": [{
            "id": "gear_small::link::collision", "pose": [1., 0., .025, 0., 0., 0., 1.],
            "geometry": {"type": "box", "size": [.004, .004, .05]},
            "bounds": [[.998, 1.002], [-.002, .002], [0., .05]],
        }]}}}}}
    physical["scene_object_bindings"] = build_scene_object_bindings(
        scene, physical, payload_evidence=payload, attachment_targets=targets)
    binding = physical["scene_object_bindings"]["gear_small"]["objects"][0]
    assert binding["part_local_object"]["primitive_poses"][0]["position"]["x"] == 0.
    assert binding["part_local_object"]["primitives"][0]["dimensions"] == [.004, .004, .05]
    physical["snapshot"]["parts"]["gear_small"]["current_pose"][0] = 3.
    projected = project_preview_scene(scene, physical)
    moved = projected["planning_scene"]["world"]["collision_objects"][-1]
    assert moved["primitive_poses"][0]["position"]["x"] == 3.
    assert moved["primitive_poses"][0]["position"]["z"] == .025
    physical["snapshot"]["parts"]["gear_small"]["current_pose"][0] = 4.
    repeated = project_preview_scene(projected, physical)
    assert repeated["planning_scene"]["world"]["collision_objects"][-1]["primitive_poses"][0]["position"]["x"] == 4.
    assert scene == original
    binding["part_local_object"]["primitives"][0]["dimensions"][2] = .049
    with pytest.raises(ValueError, match="geometry changed without evidence"):
        project_preview_scene(scene, physical)


def test_saved_native_mesh_rejects_scale_inconsistent_with_observed_bounds():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import _recorded_part_collision_object

    original = {"id": "gear_small/link/collision", "meshes": []}
    evidence = {"physics_evidence": {"models": {"gear_small": {
        "pose": [0., 0., 0., 0., 0., 0., 1.], "links": {"link": {"collisions": [{
            "id": "gear_small::link::collision", "pose": [0., 0., 0., 0., 0., 0., 1.],
            "geometry": {"type": "mesh", "uri": "saved.stl", "scale": [1., 1., 1.]},
            "bounds": [[0., .01], [0., .01], [0., 0.]],
        }]}}}}}, "physics_meshes": {"saved.stl": {"uri": "saved.stl", "source_sha256": "hash",
            "source_kind": "recorded_collision_mesh", "vertex_count": 3,
            "vertices": [[0., 0., 0.], [10., 0., 0.], [0., 10., 0.]],
            "triangles": [[2, 0, 1]], "triangle_count": 1}}}
    with pytest.raises(ValueError, match="shape/scale conflicts"):
        _recorded_part_collision_object(original, evidence, part_model="gear_small",
                                        identity=("gear_small", "link", "collision"))
    evidence["physics_evidence"]["models"]["gear_small"]["links"]["link"]["collisions"][0]["geometry"]["scale"] = [.001] * 3
    rebuilt, source = _recorded_part_collision_object(original, evidence, part_model="gear_small",
                                                      identity=("gear_small", "link", "collision"))
    assert rebuilt["meshes"][0]["vertices"][1]["x"] == .01
    assert rebuilt["meshes"][0]["triangles"] == [{"vertex_indices": [2, 0, 1]}]
    assert source["mesh_source"]["source_sha256"] == "hash"
    del evidence["physics_meshes"]["saved.stl"]["triangles"]
    with pytest.raises(ValueError, match="Complete saved collision mesh"):
        _recorded_part_collision_object(original, evidence, part_model="gear_small",
                                        identity=("gear_small", "link", "collision"))


def test_ros_saved_scene_roundtrip_restores_exact_octets_without_live_interfaces():
    pytest.importorskip("moveit_msgs.msg")
    from moveit_msgs.msg import CollisionObject, PlanningScene
    from rosidl_runtime_py.convert import message_to_ordereddict
    from cais_spade_llm.recovery_framework.recovery_motion_preview import (
        _scene_message_fingerprint, _set_recorded_message_fields, preview_scene_fingerprint,
    )

    original = PlanningScene()
    original.world.collision_objects = [CollisionObject(id="gear_small/link/collision", operation=bytes([0]))]
    saved = json.loads(json.dumps(message_to_ordereddict(original)))
    assert saved["world"]["collision_objects"][0]["operation"] == "\x00"
    assert preview_scene_fingerprint(saved) == _scene_message_fingerprint(original)
    restored = PlanningScene()
    _set_recorded_message_fields(restored, saved)
    assert restored == original
    assert message_to_ordereddict(restored) == saved
    saved["world"]["collision_objects"][0]["operation"] = 0
    _set_recorded_message_fields(restored, saved)
    assert restored == original


def test_ros_saved_scene_restores_nested_fields_without_copying_whole_subtrees():
    pytest.importorskip("moveit_msgs.msg")
    from geometry_msgs.msg import Point
    from moveit_msgs.msg import CollisionObject, PlanningScene
    from shape_msgs.msg import Mesh, MeshTriangle
    from rosidl_runtime_py.convert import message_to_ordereddict
    from cais_spade_llm.recovery_framework.recovery_motion_preview import (
        _scene_message_fingerprint, _set_recorded_message_fields,
    )

    class RecordedFields(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("Restore nested message fields without copying their whole subtree")

    original = PlanningScene()
    original.world.collision_objects = [CollisionObject(
        id="gear_small/link/collision", operation=bytes([0]), meshes=[Mesh(
            vertices=[Point(x=0., y=0., z=0.), Point(x=1., y=0., z=0.), Point(x=0., y=1., z=0.)],
            triangles=[MeshTriangle(vertex_indices=[0, 1, 2])])])]
    saved = json.loads(json.dumps(message_to_ordereddict(original)))
    expected = copy.deepcopy(saved)
    saved["world"] = RecordedFields(saved["world"])
    saved["world"]["collision_objects"][0] = RecordedFields(saved["world"]["collision_objects"][0])
    mesh = saved["world"]["collision_objects"][0]["meshes"][0]
    mesh["triangles"][0] = RecordedFields(mesh["triangles"][0])
    restored = PlanningScene()

    _set_recorded_message_fields(restored, saved)

    assert message_to_ordereddict(restored) == expected
    assert _scene_message_fingerprint(restored) == _scene_message_fingerprint(original)
    assert saved == expected
    mesh["vertices"][0]["x"] = 42.
    mesh["triangles"][0]["vertex_indices"][0] = 2
    assert restored == original
    restored.world.collision_objects[0].meshes[0].vertices[1].y = 5.
    assert mesh["vertices"][1]["y"] == 0.


@pytest.mark.parametrize("operation", [None, True, -1, 256, "", "ab", "\u0100", b"", b"ab"])
def test_ros_saved_scene_rejects_invalid_octets_without_changing_the_target(operation):
    pytest.importorskip("moveit_msgs.msg")
    from moveit_msgs.msg import CollisionObject, PlanningScene
    from rosidl_runtime_py.convert import message_to_ordereddict
    from cais_spade_llm.recovery_framework.recovery_motion_preview import _set_recorded_message_fields

    target = PlanningScene()
    target.world.collision_objects = [CollisionObject(id="gear_small/link/collision", operation=bytes([0]))]
    saved = json.loads(json.dumps(message_to_ordereddict(target)))
    saved["world"]["collision_objects"][0]["operation"] = operation
    before = copy.deepcopy(target)

    with pytest.raises(ValueError, match="Recorded octet is invalid"):
        _set_recorded_message_fields(target, saved)

    assert target == before


def test_scene_projection_requires_exact_identity_mapping_and_geometry():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import build_scene_object_bindings, project_preview_scene

    scene, physical, payload, targets = projection_fixture()
    with pytest.raises(ValueError, match="scene_object_bindings"):
        project_preview_scene(scene, physical)
    with pytest.raises(ValueError, match="identities are missing"):
        build_scene_object_bindings(scene, physical, payload_evidence={}, attachment_targets=targets)
    physical["scene_object_bindings"] = build_scene_object_bindings(scene, physical, payload_evidence=payload, attachment_targets=targets)
    physical["scene_object_bindings"]["gear_small"]["objects"][0]["part_local_object"]["primitives"][0]["dimensions"][0] = 0.5
    with pytest.raises(ValueError, match="geometry changed"):
        project_preview_scene(scene, physical)


def mounting_fixture():
    collision = '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>'
    description = f'''<robot name="mounting_test">
      <link name="world"/><link name="ur5e_1_base_link"/>
      <link name="ur5e_1_base_link_inertia">{collision}</link>
      <link name="ur5e_1_shoulder_link">{collision}</link>
      <link name="KMR_base_link">{collision}</link><link name="iiwa_link_0">{collision}</link>
      <joint name="ur5e_1_world_joint" type="fixed"><parent link="world"/><child link="ur5e_1_base_link"/>
        <origin xyz="1 2 0.8" rpy="0 0 0"/></joint>
      <joint name="ur5e_1_base_link-base_link_inertia" type="fixed">
        <parent link="ur5e_1_base_link"/><child link="ur5e_1_base_link_inertia"/></joint>
      <joint name="ur5e_1_shoulder_pan_joint" type="revolute">
        <parent link="ur5e_1_base_link_inertia"/><child link="ur5e_1_shoulder_link"/></joint>
      <joint name="KMR_base_yaw_joint" type="continuous"><parent link="world"/><child link="KMR_base_link"/></joint>
      <joint name="iiwa_root_joint" type="fixed"><parent link="KMR_base_link"/><child link="iiwa_link_0"/>
        <origin xyz="-0.25 0 0.70" rpy="0 0 1.57079632679"/></joint>
    </robot>'''
    pose = [1., 2., .78, 0., 0., 0., 1.]
    geometry = {"type": "box", "size": [.4, .4, .04]}
    object_id = "ur5e-1 pedestal/link/mounting_plate"
    obj = {"id": object_id, "operation": 0, "header": {"frame_id": "world"},
        "pose": {"position": dict(zip(("x", "y", "z"), pose[:3])),
                 "orientation": dict(zip(("x", "y", "z", "w"), pose[3:]))},
        "primitives": [{"type": 1, "dimensions": [.4, .4, .04]}],
        "primitive_poses": [{"position": {"x": 0., "y": 0., "z": 0.},
                             "orientation": {"x": 0., "y": 0., "z": 0., "w": 1.}}],
        "meshes": [], "planes": [], "subframe_names": [], "subframe_poses": []}
    names = ["ur5e_1_base_link_inertia", "ur5e_1_shoulder_link", "KMR_base_link", "iiwa_link_0"]
    scene = {"world": {"collision_objects": [obj, {**copy.deepcopy(obj), "id": "unrelated"}]},
        "allowed_collision_matrix": {"entry_names": names,
            "entry_values": [{"enabled": [False] * len(names)} for _ in names],
            "default_entry_names": [], "default_entry_values": []}}
    physical = {"physics_evidence": {"instance_id": "saved", "simulation_time": 4.0,
        "models": {"ur5e-1 pedestal": {"static": True, "links": {"link": {"collisions": [{
            "id": "ur5e-1 pedestal::link::mounting_plate", "geometry": copy.deepcopy(geometry),
            "pose": pose[:],
        }]}}}}}}
    declarations = {"fixed_joints": ["iiwa_root_joint"], "world_contacts": [{
        "resource_id": "ur5e-1", "robot_collision_link": "ur5e_1_base_link_inertia",
        "mounting_frame": "ur5e_1_base_link", "world_collision_object_id": object_id,
        "physics_collision_id": "ur5e-1 pedestal::link::mounting_plate", "geometry": geometry, "pose": pose,
    }]}
    return {"planning_scene": scene, "robot_description": description, "robots": [{
        "resource_id": "ur5e-1", "prefix": "ur5e_1_", "base_xyz": [1., 2., .8], "base_rpy": [0., 0., 0.],
    }], "physical_evidence": physical, "declarations": declarations}


def test_declared_mounting_contacts_preserve_source_and_unrelated_pairs(monkeypatch):
    from cais_spade_llm.recovery_framework import recovery_motion_preview as preview
    from cais_spade_llm.recovery_framework.recovery_observations import fingerprint

    data = mounting_fixture()
    before = copy.deepcopy(data)
    monkeypatch.setattr(preview, "_new_node", lambda: pytest.fail("Mounting proof contacted ROS"))
    result = preview.apply_declared_mounting_contacts(**data)
    assert data == before
    assert result["planning_scene"]["world"] == before["planning_scene"]["world"]
    matrix = result["planning_scene"]["allowed_collision_matrix"]
    allowed = {frozenset((a, b)) for i, a in enumerate(matrix["entry_names"])
               for j, b in enumerate(matrix["entry_names"]) if matrix["entry_values"][i]["enabled"][j]}
    assert allowed == {frozenset(("KMR_base_link", "iiwa_link_0")),
        frozenset(("ur5e_1_base_link_inertia", "ur5e-1 pedestal/link/mounting_plate"))}
    proof = result["evidence"]
    assert proof["input_scene_fingerprint"] == fingerprint(before["planning_scene"])
    assert proof["output_scene_fingerprint"] == fingerprint(result["planning_scene"])
    assert proof["declarations_fingerprint"] == fingerprint(data["declarations"])
    assert proof["contacts"][1]["physics_instance_id"] == "saved"
    assert proof["contacts"][1]["physics_collision_id"] == "ur5e-1 pedestal::link::mounting_plate"
    assert proof["contacts"][1]["fixed_joint_chain"] == ["ur5e_1_base_link-base_link_inertia", "ur5e_1_world_joint"]
    assert proof["execution_authorized"] is proof["command_sent"] is False


@pytest.mark.parametrize("change", [
    "missing_declaration", "moving_joint", "moving_collision_link", "moving_model", "wrong_pose",
    "wrong_geometry", "wrong_scene_pose", "wrong_object", "wrong_physics_id", "wrong_mount",
    "collision_defaults", "asymmetric_matrix",
])
def test_declared_mounting_contacts_reject_unproved_relationships(change):
    from cais_spade_llm.recovery_framework.recovery_motion_preview import apply_declared_mounting_contacts

    data = mounting_fixture()
    declaration = data["declarations"]["world_contacts"][0]
    collision = data["physical_evidence"]["physics_evidence"]["models"]["ur5e-1 pedestal"]["links"]["link"]["collisions"][0]
    if change == "missing_declaration":
        del data["declarations"]["world_contacts"]
    elif change == "moving_joint":
        data["robot_description"] = data["robot_description"].replace('name="iiwa_root_joint" type="fixed"',
            'name="iiwa_root_joint" type="revolute"')
    elif change == "moving_collision_link":
        declaration["robot_collision_link"] = "ur5e_1_shoulder_link"
    elif change == "moving_model":
        data["physical_evidence"]["physics_evidence"]["models"]["ur5e-1 pedestal"]["static"] = False
    elif change == "wrong_pose":
        collision["pose"][2] = .77
    elif change == "wrong_geometry":
        collision["geometry"]["size"][0] = .5
    elif change == "wrong_scene_pose":
        data["planning_scene"]["world"]["collision_objects"][0]["pose"]["position"]["x"] = 1.1
    elif change == "wrong_object":
        declaration["world_collision_object_id"] = "unrelated"
    elif change == "wrong_physics_id":
        declaration["physics_collision_id"] = "missing::link::mounting_plate"
    elif change == "wrong_mount":
        data["robots"][0]["base_xyz"][0] = 2.
    elif change == "collision_defaults":
        data["planning_scene"]["allowed_collision_matrix"]["default_entry_names"] = ["world"]
        data["planning_scene"]["allowed_collision_matrix"]["default_entry_values"] = [True]
    elif change == "asymmetric_matrix":
        data["planning_scene"]["allowed_collision_matrix"]["entry_values"][0]["enabled"][1] = True
    before = copy.deepcopy(data)
    with pytest.raises(ValueError):
        apply_declared_mounting_contacts(**data)
    assert data == before


def mating_request():
    from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_mating_contact

    original = request()
    original.update(mode="cartesian", joint_target={}, waypoints=[{
        "position": {"x": 0., "y": 0., "z": -.1}, "orientation": {"x": 0., "y": 0., "z": 0., "w": 1.}}])
    evidence = {"version": 1, "status": "prepared", "resource_id": original["resource_id"],
        "part_name": "part", "model_name": "part", "part_object_id": "part/link/collision",
        "reference_link": "tool", "frame_id": "world", "command_sent": False, "execution_authorized": False,
        "start_tool_pose": [0., 0., 0., 0., 0., 0., 1.], "target_tool_pose": [0., 0., -.1, 0., 0., 0., 1.],
        "start_part_pose": [0., 0., 0., 0., 0., 0., 1.], "tool_to_part_pose": [0., 0., 0., 0., 0., 0., 1.],
        "start_part_z": 0., "target_origin_pose": {"x": 0., "y": 0., "z": -.1},
        "binding": {key: key for key in ("state_fingerprint", "model_fingerprint", "owner_evidence_fingerprint",
            "physics_fingerprint", "scene_object_bindings_fingerprint", "peer_fingerprint")},
        "simulation_mating_contact": {"target_collision_object": "shaft/link/collision", "axis_tolerance_m": .0005,
            "max_contact_depth_m": .0005, "target_yaw_rad": 0., "yaw_tolerance_rad": .005,
            "ignore_tooth_contact_with": ["gear/link/collision"], "retain_fixture_attachment": True}}
    return bind_mating_contact(original, evidence)


def mating_result(prepared):
    response = result()
    response["mating_contact_fingerprint"] = prepared["mating_contact_fingerprint"]
    binding = {field: prepared[field] for field in ("resource_id", "capture_fingerprint", "proposal_fingerprint",
        "scene_fingerprint", "model_fingerprint")}
    binding.update(support_departure_fingerprint="", mating_contact_fingerprint=prepared["mating_contact_fingerprint"])
    audit = {"version": 1, "source": "PreviewRecoveryMotion", "request_binding": binding,
        "pairs_complete": True, "continuous_collision_verified": False, "samples": [{
            "state_index": index, "phase": "trajectory", "joint_state": {"name": ["joint"], "position": [value]},
            "part_pose": [0., 0., -value, 0., 0., 0., 1.], "checked_contacts": 1, "contacts": [{
                "body_1": "part/link/collision", "body_2": "shaft/link/collision", "depth": .0001,
                "position": [0., 0., -value], "normal": [0., 0., 1.]}]}
            for index, value in enumerate((0., .1))]}
    response["mating_contact_evidence_json"] = json.dumps(audit)
    return response


def test_mating_wire_preserves_exact_policy_and_requires_separate_continuous_validation():
    prepared = mating_request()
    original = copy.deepcopy(prepared)
    record = preview_result_record(prepared, mating_result(prepared))
    assert prepared == original
    assert record["mating_contact_evidence"]["pairs_complete"] is True
    assert record["coverage"]["continuous_collision_verified"] is False
    assert record["execution_authorized"] is False
    assert json.loads(prepared["mating_contact_json"])["simulation_mating_contact"]["max_contact_depth_m"] == .0005


@pytest.mark.parametrize("defect", ["stale", "fingerprint", "joint", "route", "support", "depth", "missing_source", "part", "execution"])
def test_mating_wire_rejects_changed_or_missing_owner_evidence(defect):
    from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_mating_contact

    prepared = mating_request()
    evidence = json.loads(prepared["mating_contact_json"])
    if defect == "stale":
        prepared["scene_fingerprint"] = "changed"
    elif defect == "fingerprint":
        prepared["mating_contact_json"] += " "
    elif defect == "joint":
        prepared.update(mode="joint_target", waypoints=[], joint_target={"name": ["joint"], "position": [0.]})
    elif defect == "route":
        prepared["route_search_json"] = "{}"
    elif defect == "support":
        prepared["support_departure_json"] = "{}"
    else:
        if defect == "depth":
            evidence["simulation_mating_contact"]["max_contact_depth_m"] = -.0005
        elif defect == "missing_source":
            evidence["binding"].pop("peer_fingerprint")
        elif defect == "part":
            evidence["part_object_id"] = "unrelated"
        else:
            evidence["execution_authorized"] = True
        with pytest.raises(ValueError):
            bind_mating_contact(prepared, evidence)
        return
    with pytest.raises(ValueError):
        validate_preview_request(prepared)


@pytest.mark.parametrize("defect", ["missing_raw", "stale", "missing_sample", "joint", "corridor", "depth", "pair", "count", "continuous"])
def test_mating_wire_rejects_incomplete_or_forged_native_samples(defect):
    prepared = mating_request()
    response = mating_result(prepared)
    audit = json.loads(response["mating_contact_evidence_json"])
    if defect == "missing_raw":
        response["mating_contact_evidence"] = audit
        response["mating_contact_evidence_json"] = ""
    else:
        if defect == "stale":
            audit["request_binding"]["scene_fingerprint"] = "changed"
        elif defect == "missing_sample":
            audit["samples"].pop()
        elif defect == "joint":
            audit["samples"][1]["joint_state"]["position"] = [.2]
        elif defect == "corridor":
            audit["samples"][1]["part_pose"][0] = .001
        elif defect == "depth":
            audit["samples"][1]["contacts"][0]["depth"] = .0006
        elif defect == "pair":
            audit["samples"][1]["contacts"][0]["body_2"] = "unrelated"
        elif defect == "count":
            audit["samples"][1]["checked_contacts"] = 2
        else:
            audit["continuous_collision_verified"] = True
        response["mating_contact_evidence_json"] = json.dumps(audit)
    with pytest.raises(ValueError):
        preview_result_record(prepared, response)


def test_mating_wire_discards_parsed_injection_from_an_unresolved_response():
    prepared = mating_request()
    response = mating_result(prepared)
    response.update(status="NEEDS_CONTEXT", path_complete=False, fraction=0., trajectory={},
        mating_contact_evidence_json="", mating_contact_evidence={"pairs_complete": True})
    assert "mating_contact_evidence" not in preview_result_record(prepared, response)


@pytest.fixture
def file_preview_environment(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    from pathlib import Path
    from cais_spade_llm.recovery_framework import recovery_motion_preview as preview, recovery_observations

    source = tmp_path / 'observations.json'
    source.write_text('{}')
    parameters = tmp_path / 'parameters.yaml'
    parameters.write_text('exact recorded parameters')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    archive = {'source_path': str(source), 'parameters_path': str(parameters), 'manifest_path': str(parameters),
        'manifest_sha256': digest(parameters), 'files': {str(parameters): digest(parameters)},
        'binding': {'model_fingerprint': 'model', 'source_sha256': digest(source)}, 'capture_verification': {'original': True}}
    monkeypatch.setattr(recovery_observations, 'read_native_preview_model', lambda *a: archive)
    action = ModuleType('cais_lab_robotics.action')
    action.PreviewRecoveryMotion = SimpleNamespace(Goal=SimpleNamespace, Result=SimpleNamespace)
    serialization = ModuleType('rclpy.serialization')
    serialization.serialize_message = lambda value: json.dumps(vars(value)).encode()
    serialization.deserialize_message = lambda value, cls: json.loads(value)
    convert = ModuleType('rosidl_runtime_py.convert')
    convert.message_to_ordereddict = lambda value: value
    for name, module in [('cais_lab_robotics.action', action), ('rclpy.serialization', serialization),
                         ('rosidl_runtime_py.convert', convert)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(preview, '_set_recorded_message_fields', lambda obj, data: vars(obj).update(data))
    monkeypatch.setattr(preview, '_scene_message_fingerprint', lambda _: 'scene')
    monkeypatch.setattr(preview, '_native_preview_executable', lambda: Path(sys.executable))
    monkeypatch.setattr(preview, '_new_node', lambda: pytest.fail('File preview called ROS transport'))
    original_spawn = subprocess.Popen
    children = []
    settings = {'delay': 0., 'exit_code': 0, 'change_parameters': False, 'spawn_delay': 0.}

    def spawn(args, **kwargs):
        assert args[1] == '--input-goal' and args[3] == '--output-result'
        assert args[-1] == str(parameters)
        code = ('import sys,time,json,pathlib; time.sleep(' + str(settings['delay']) + '); '
                + ('pathlib.Path(' + repr(str(parameters)) + ').write_text("changed"); ' if settings['change_parameters'] else '')
                + 'pathlib.Path(sys.argv[1]).write_text(' + repr(json.dumps(settings.get('result', result()))) + '); '
                + 'sys.exit(' + str(settings['exit_code']) + ')')
        child = original_spawn([sys.executable, '-c', code, args[4]], **kwargs)
        children.append(child)
        time.sleep(settings['spawn_delay'])
        return child
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    def no_child_watcher():
        raise NotImplementedError('The agent event-loop policy has no child watcher')
    monkeypatch.setattr(asyncio.get_event_loop_policy(), 'get_child_watcher', no_child_watcher)
    return preview, archive, tmp_path / 'calls', children, settings


def test_file_native_preview_captures_exact_messages_and_local_provenance(file_preview_environment):
    from pathlib import Path
    preview, archive, directory, children, _ = file_preview_environment
    output = asyncio.run(preview.preview_recovery_motion_from_files(request(), execution_mode='simulation',
        model_archive=archive, artifact_directory=directory))
    evidence = output['recorded_file_evidence']
    path = Path(evidence['directory'])
    assert json.loads((path / 'goal.cdr').read_bytes()) == request()
    assert json.loads((path / 'result.cdr').read_bytes()) == result()
    provenance = json.loads((path / 'provenance.json').read_text())
    assert provenance['status'] == 'complete' and provenance['cancellation_complete'] is True
    assert provenance['command_sent'] is False and provenance['execution_authorized'] is False
    assert children[0].returncode == 0


@pytest.mark.parametrize('cancel', [False, True])
def test_native_fingerprint_without_child_watcher_reaps_owned_process(file_preview_environment, monkeypatch, cancel):
    from types import SimpleNamespace

    preview, archive, directory, children, settings = file_preview_environment
    directory.mkdir()
    settings['result'] = {'model_fingerprint': 'model', 'path_complete': False,
        'trajectory': {'joint_trajectory': {'points': []}},
        'reason': 'Resource, capture, and proposal fingerprints are required'}
    settings['delay'] = 10. if cancel else 0.
    serialization = sys.modules['rclpy.serialization']
    monkeypatch.setattr(serialization, 'deserialize_message',
        lambda value, cls: json.loads(value, object_hook=lambda row: SimpleNamespace(**row)))

    async def run():
        event = asyncio.Event()
        task = asyncio.create_task(preview.verify_native_model_parameters(archive['parameters_path'],
            expected_model_fingerprint='model', artifact_directory=directory, cancel_event=event))
        if cancel:
            while not children:
                await asyncio.sleep(.01)
            event.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            record = await task
            assert record['fingerprint_verified'] is True
            assert record['command_sent'] is False and record['execution_authorized'] is False
            assert json.loads((directory / 'fingerprint-goal.cdr').read_bytes()) == {'timeout_seconds': 1.}
    asyncio.run(run())
    assert len(children) == 1 and children[0].returncode is not None


@pytest.mark.parametrize('failure', ['cancel', 'timeout', 'exit', 'changed'])
def test_file_native_preview_reaps_child_and_retains_failure(file_preview_environment, failure):
    preview, archive, directory, children, settings = file_preview_environment
    if failure in {'cancel', 'timeout'}:
        settings['delay'] = 10
    settings['exit_code'] = 2 if failure == 'exit' else 0
    settings['change_parameters'] = failure == 'changed'

    async def run():
        cancel = asyncio.Event()
        task = asyncio.create_task(preview.preview_recovery_motion_from_files(request(), execution_mode='simulation',
            model_archive=archive, artifact_directory=directory, cancel_event=cancel,
            timeout_seconds=.1 if failure == 'timeout' else 5.))
        if failure == 'cancel':
            while not children:
                await asyncio.sleep(.01)
            cancel.set()
        with pytest.raises((asyncio.CancelledError, TimeoutError, RuntimeError, ValueError)):
            await task
    asyncio.run(run())
    assert len(children) == 1 and children[0].returncode is not None
    provenance = json.loads(next(directory.glob('*/provenance.json')).read_text())
    assert provenance['status'] in {'failed', 'cancelled'}
    assert provenance['cancellation_complete'] is True and provenance['error']


def test_file_native_preview_rejects_hardware_without_importing_ros(tmp_path):
    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_recovery_motion_from_files
    with pytest.raises(ValueError, match='simulation'):
        asyncio.run(preview_recovery_motion_from_files(request(), execution_mode='hardware',
            model_archive={}, artifact_directory=tmp_path))


def test_file_native_preview_drains_cancellation_during_spawn(file_preview_environment):
    preview, archive, directory, children, settings = file_preview_environment
    settings.update(delay=10., spawn_delay=.2)

    async def run():
        task = asyncio.create_task(preview.preview_recovery_motion_from_files(request(), execution_mode='simulation',
            model_archive=archive, artifact_directory=directory))
        while not children:
            await asyncio.sleep(.01)
        task.cancel()
        await asyncio.sleep(.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert children[0].returncode is not None
    provenance = json.loads(next(directory.glob('*/provenance.json')).read_text())
    assert provenance['status'] == 'cancelled' and provenance['cancellation_complete'] is True
