from __future__ import annotations

"""Recorded mating policy and private continuous insertion evidence."""

from copy import deepcopy
import hashlib
import json
import math

import pytest

from cais_spade_llm.resources.recovery_mating_contact import (
    prepare_mating_contact, projected_fixture_handoff, validate_mating_contact_samples, verify_mating_contact,
)


@pytest.fixture(autouse=True)
def _clear_private_corridor_cache(monkeypatch):
    from cais_spade_llm.resources import recovery_mating_contact as module

    monkeypatch.setattr(module, "_mating_corridor_cache", None)


def _case():
    configuration = {"joint_names": ["slide"], "interpolation": "splines", "root": "world",
        "root_xyz": [0., 0., 0.], "root_rpy": [0., 0., 0.], "reference_link": "body",
        "joints": [{"name": "slide", "parent": "world", "child": "body", "type": "prismatic",
                    "xyz": [0., 0., 0.], "rpy": [0., 0., 0.], "axis": [0., 0., 1.]}],
        "components": [{"id": "body", "link": "body", "bounds": [[-.01, .01]] * 3}]}
    part_pose = [0., 0., .01, 0., 0., 0., 1.]
    state = {"frame": "world", "resources": {"ur5e-3": {"held_part": "gear_small", "joint_names": ["slide"],
        "joint_positions": [.01], "current_pose": part_pose, "grasp_transform": [0., 0., 0., 0., 0., 0., 1.]},
        "ur5e-4": {"held_part": None, "current_pose": [1., 0., 0., 0., 0., 0., 1.]}},
        "parts": {"gear_small": {"model_name": "gear_small", "current_pose": part_pose}}}
    policy = {"target_collision_object": "Gear_Plate/Gear_Shaft_1/collision",
        "axis_tolerance_m": .0005, "max_contact_depth_m": .0005, "target_yaw_rad": 0.,
        "yaw_tolerance_rad": .005, "ignore_tooth_contact_with": [], "retain_fixture_attachment": True}
    owner = {"execution_mode": "simulation", "source": {"capture": "original"}, "controller_config": {
        "payload_collision": {"enabled": True}, "move_group": {"ee_link": "body", "frame_id": "world"},
        "services": {"attach": "/ATTACHLINK", "detach": "/DETACHLINK"}}, "inputs": {"geometry": {
        "assembly_board": {"model_name": "assembly_board_v1", "component_attachment_model": "assembly_board_v1",
            "mating_contact_by_part": {"gear_small": policy}, "center": {"x": 0., "y": 0.},
            "slots": {"gear_small": [0., 0.]}, "target_origin_z_m": {"gear_small": 0.}},
        "parts": {"model_map": {"gear_small": "gear_small"},
                  "assembly_target_map": {"gear_small": "Gear_Plate/Gear_Shaft_1"}}}}}
    origin = [0., 0., 0., 0., 0., 0., 1.]
    physics = {"instance_id": "capture", "simulation_time": 12., "models": {
        "gear_small": {"joint_count": 0, "links": {"link": {"collisions": [
            {"id": "gear_small::link::collision", "geometry": {"type": "mesh"}}]}}},
        "Gear_Plate": {"static": True, "links": {"Gear_Shaft_1": {"collisions": [
            {"id": "Gear_Plate::Gear_Shaft_1::collision", "geometry": {"type": "cylinder"}}]}}},
        "assembly_board_v1": {"static": True, "links": {"link": {"pose": origin, "collisions": []}}}}}
    evidence = {"snapshot": deepcopy(state), "resource_models": {"ur5e-3": {"configuration": configuration}},
        "physics_evidence": physics, "scene_object_bindings": {"gear_small": {"objects": [{
            "object_id": "gear_small/link/collision", "recorded_geometry": {"collision_id": "gear_small::link::collision"}}]}},
        "support_collision_object_bindings": {"Gear_Plate::Gear_Shaft_1::collision": {
            "object_id": "Gear_Plate/Gear_Shaft_1/collision", "source": "captured SDF"}}}
    trajectory = {"joint_names": ["slide"], "duration_ns": 1_000_000_000, "points": [
        {"positions": [value], "velocities": [0.], "time_from_start": {"sec": index, "nanosec": 0}}
        for index, value in enumerate((.01, 0.))]}
    kwargs = {"resource_id": "ur5e-3", "part_name": "gear_small", "state": state, "evidence": evidence,
              "owner_evidence": owner, "target_origin_pose": {"x": 0., "y": 0., "z": 0.}, "target_tool_pose": origin}
    return kwargs, trajectory


def _proof(kwargs, trajectory, prepared):
    return verify_mating_contact(**{key: kwargs[key] for key in (
        "resource_id", "state", "evidence", "owner_evidence")}, joint_trajectory=trajectory, mating_contact=prepared)


def _native(prepared, trajectory):
    text = json.dumps(prepared, sort_keys=True)
    return {"mating_contact_json": text, "mating_contact_fingerprint": hashlib.sha256(text.encode()).hexdigest(),
        "mating_contact": deepcopy(prepared), "joint_trajectory": deepcopy(trajectory),
        "native_mating_contact_evidence": {"version": 1, "source": "PreviewRecoveryMotion", "pairs_complete": True,
            "continuous_collision_verified": False, "samples": [
                {"state_index": index, "phase": "trajectory", "joint_state": {
                    "name": ["slide"], "position": deepcopy(point["positions"])},
                    "part_pose": [0., 0., point["positions"][0], 0., 0., 0., 1.],
                    "checked_contacts": int(index > 0), "contacts": ([] if index == 0 else [{
                        "body_1": "gear_small/link/collision", "body_2": "Gear_Plate/Gear_Shaft_1/collision",
                        "depth": .00001, "position": [0., 0., 0.], "normal": [1., 0., 0.]}])}
                for index, point in enumerate(trajectory["points"])]}}


def test_mating_captured_policy_and_continuous_corridor_are_separate_from_native_contacts():
    kwargs, trajectory = _case()
    before = deepcopy(kwargs)
    prepared = prepare_mating_contact(**kwargs)
    assert prepared["status"] == "prepared", prepared
    assert prepared["simulation_mating_contact"] == kwargs["owner_evidence"]["inputs"]["geometry"]["assembly_board"]["mating_contact_by_part"]["gear_small"]
    proof = _proof(kwargs, trajectory, prepared)
    assert proof["status"] == "supported", proof
    assert proof["continuous_mating_corridor_verified"]
    assert proof["continuous_collision_verified"] is False
    assert proof["sampled_contact_depth_requires_native_evidence"]
    validate_mating_contact_samples(_native(prepared, trajectory))
    assert kwargs == before


@pytest.mark.parametrize("defect", ["hardware", "payload_disabled", "wrong_part", "second_holder", "missing_transform",
    "changed_target", "changed_collision", "missing_geometry", "depth_limit", "null_policy", "off_axis", "yaw", "frame"])
def test_mating_preparation_keeps_missing_and_conflicting_declarations_unresolved(defect):
    kwargs, _ = _case()
    policy = kwargs["owner_evidence"]["inputs"]["geometry"]["assembly_board"]["mating_contact_by_part"]["gear_small"]
    if defect == "hardware": kwargs["owner_evidence"]["execution_mode"] = "hardware"
    elif defect == "payload_disabled": kwargs["owner_evidence"]["controller_config"]["payload_collision"]["enabled"] = False
    elif defect == "wrong_part": kwargs["state"]["parts"]["gear_small"]["model_name"] = "gear_medium"
    elif defect == "second_holder": kwargs["state"]["resources"]["ur5e-4"]["held_part"] = "gear_small"
    elif defect == "missing_transform": kwargs["state"]["resources"]["ur5e-3"].pop("grasp_transform")
    elif defect == "changed_target": kwargs["target_origin_pose"]["x"] = .01
    elif defect == "changed_collision": policy["target_collision_object"] = "Gear_Plate/Gear_Shaft_2/collision"
    elif defect == "missing_geometry": kwargs["evidence"]["support_collision_object_bindings"].clear()
    elif defect == "depth_limit": policy["max_contact_depth_m"] = .001
    elif defect == "null_policy": policy["axis_tolerance_m"] = None
    elif defect == "off_axis": kwargs["target_tool_pose"][0] = .002
    elif defect == "yaw": kwargs["target_tool_pose"][3:] = [0., 0., math.sin(.1), math.cos(.1)]
    elif defect == "frame": kwargs["state"]["frame"] = "another_frame"
    assert prepare_mating_contact(**kwargs)["status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("defect", ["stale_owner", "stale_peer", "changed_start", "endpoint", "interior", "cell_limit", "tracking_error"])
def test_continuous_mating_never_substitutes_endpoint_or_stale_evidence(defect, monkeypatch):
    from cais_spade_llm.resources import recovery_mating_contact as module

    kwargs, trajectory = _case()
    prepared = prepare_mating_contact(**kwargs)
    if defect == "stale_owner": kwargs["owner_evidence"]["source"]["capture"] = "new"
    elif defect == "stale_peer": kwargs["state"]["resources"]["ur5e-4"]["current_pose"][0] = 2.
    elif defect == "changed_start": trajectory["points"][0]["positions"] = [.009]
    elif defect == "endpoint": trajectory["points"][-1]["positions"] = [.001]
    elif defect == "interior":
        trajectory["points"][0]["velocities"] = [-.2]
        trajectory["points"][-1]["velocities"] = [.2]
    elif defect == "cell_limit": monkeypatch.setattr(module, "_LIMIT", 0)
    elif defect == "tracking_error":
        kwargs["evidence"]["resource_models"]["ur5e-3"]["configuration"]["joint_position_error"] = {"slide": .001}
        prepared = prepare_mating_contact(**kwargs)
    assert _proof(kwargs, trajectory, prepared)["status"] == ("rejected" if defect == "interior" else "NEEDS_CONTEXT")


@pytest.mark.parametrize("defect", ["missing_first", "missing_last", "sample_order", "wrong_joints", "truncated", "depth", "other_pair",
                                    "bad_pose", "false_continuous", "json_hash", "null_depth", "boolean_depth", "nan_depth"])
def test_native_mating_audit_preserves_exact_pairs_depth_and_full_waypoints(defect):
    kwargs, trajectory = _case()
    native = _native(prepare_mating_contact(**kwargs), trajectory)
    audit = native["native_mating_contact_evidence"]
    if defect == "missing_first": audit["samples"].pop(0)
    elif defect == "missing_last": audit["samples"].pop()
    elif defect == "sample_order": audit["samples"][-1]["state_index"] = 0
    elif defect == "wrong_joints": audit["samples"][-1]["joint_state"]["position"] = [.1]
    elif defect == "truncated": audit["samples"][-1]["checked_contacts"] = 2
    elif defect == "depth": audit["samples"][-1]["contacts"][0]["depth"] = .00050001
    elif defect == "other_pair": audit["samples"][-1]["contacts"][0]["body_2"] = "other/link/collision"
    elif defect == "bad_pose": audit["samples"][-1]["part_pose"][0] = .001
    elif defect == "false_continuous": audit["continuous_collision_verified"] = True
    elif defect == "json_hash": native["mating_contact_json"] += " "
    elif defect == "null_depth": audit["samples"][-1]["contacts"][0]["depth"] = None
    elif defect == "boolean_depth": audit["samples"][-1]["contacts"][0]["depth"] = False
    elif defect == "nan_depth": audit["samples"][-1]["contacts"][0]["depth"] = math.nan
    with pytest.raises(ValueError): validate_mating_contact_samples(native)


@pytest.mark.parametrize("defect", [None, "missing_native", "wrong_proof", "moving", "wrong_endpoint", "off_seat", "unretained", "moving_fixture", "missing_service"])
def test_projected_fixture_handoff_requires_stopped_supported_declared_seat(defect):
    kwargs, trajectory = _case()
    prepared = prepare_mating_contact(**kwargs)
    proof = _proof(kwargs, trajectory, prepared)
    native = _native(prepared, trajectory)
    state = deepcopy(kwargs["state"])
    state["resources"]["ur5e-3"].update(joint_positions=[0.], current_pose=[0., 0., 0., 0., 0., 0., 1.])
    state["parts"]["gear_small"]["current_pose"] = [0., 0., 0., 0., 0., 0., 1.]
    endpoint = {"mating_start_state": kwargs["state"], "mating_native_record": native, "mating_corridor_evidence": proof}
    if defect == "missing_native": native.pop("native_mating_contact_evidence")
    elif defect == "wrong_proof": proof["refinement_cells"] += 1
    elif defect == "moving": native["joint_trajectory"]["points"][-1]["velocities"] = [.01]
    elif defect == "wrong_endpoint": state["parts"]["gear_small"]["current_pose"][0] = .001
    elif defect == "off_seat": state["parts"]["gear_small"]["current_pose"][2] = .001
    elif defect == "unretained": prepared["simulation_mating_contact"]["retain_fixture_attachment"] = False
    elif defect == "moving_fixture": kwargs["evidence"]["physics_evidence"]["models"]["assembly_board_v1"]["static"] = False
    elif defect == "missing_service": kwargs["owner_evidence"]["controller_config"]["services"].pop("attach")
    before = deepcopy((kwargs, state))
    result = projected_fixture_handoff(resource_id=kwargs["resource_id"], part_name=kwargs["part_name"], state=state,
        evidence=kwargs["evidence"], owner_evidence=kwargs["owner_evidence"], mating_contact=prepared, endpoint_evidence=endpoint)
    assert result["status"] == ("supported" if defect is None else "NEEDS_CONTEXT"), result
    assert (kwargs, state) == before
    assert result["physical_execution_verified"] is False and result["settling_modeled"] is False
    if defect is None:
        assert result["current_pose"] == state["parts"]["gear_small"]["current_pose"]
        assert result["fixture_model"] == "assembly_board_v1" and result["fixture_link"] == "link"
        assert result["fixture_to_part_pose"] == [0., 0., 0., 0., 0., 0., 1.]
        assert result["kind"] == "projected_fixture_handoff"
        assert result["seated"] is True and result["retained_fixture"] is True
        assert result["target_origin_pose"] == kwargs["target_origin_pose"]


def test_native_mating_audit_retains_the_exact_issued_request_binding():
    kwargs, trajectory = _case()
    prepared = prepare_mating_contact(**kwargs)
    prepared["preview_binding"] = {"capture_fingerprint": "saved", "proposal_fingerprint": "current"}
    native = _native(prepared, trajectory)
    audit = native["native_mating_contact_evidence"]
    audit["request_binding"] = {**prepared["preview_binding"],
                                "mating_contact_fingerprint": native["mating_contact_fingerprint"]}
    validate_mating_contact_samples(native)
    audit["request_binding"]["proposal_fingerprint"] = "another"
    with pytest.raises(ValueError, match="request binding"):
        validate_mating_contact_samples(native)


@pytest.mark.parametrize("kind", ["off_axis", "yaw"])
def test_continuous_mating_rejects_interior_departure_despite_valid_endpoints(kind):
    kwargs, trajectory = _case()
    configuration = kwargs["evidence"]["resource_models"]["ur5e-3"]["configuration"]
    configuration["joint_names"].append("excursion")
    configuration["joints"].append({"name": "excursion", "parent": "body", "child": "tip",
        "type": "prismatic" if kind == "off_axis" else "revolute", "xyz": [0., 0., 0.],
        "rpy": [0., 0., 0.], "axis": [1., 0., 0.] if kind == "off_axis" else [0., 0., 1.]})
    configuration["reference_link"] = "tip"
    kwargs["owner_evidence"]["controller_config"]["move_group"]["ee_link"] = "tip"
    owner = kwargs["state"]["resources"]["ur5e-3"]
    owner["joint_names"].append("excursion")
    owner["joint_positions"].append(0.)
    trajectory["joint_names"].append("excursion")
    velocity = .01 if kind == "off_axis" else .1
    for index, point in enumerate(trajectory["points"]):
        point["positions"].append(0.)
        point["velocities"].append(velocity if index == 0 else -velocity)
    prepared = prepare_mating_contact(**kwargs)
    assert prepared["status"] == "prepared", prepared
    proof = _proof(kwargs, trajectory, prepared)
    assert proof["status"] == "rejected"
    assert "interpolated part leaves" in proof["reason"]
    assert proof["counterexample"]["violated_conditions"]
    assert json.loads(json.dumps(proof)) == proof


@pytest.mark.parametrize("defect", [None, "legacy", "wrong_target", "wrong_policy", "wrong_pose", "unretained", "unseated"])
def test_owner_completion_requires_matching_fixture_proof_for_a_nonexact_seated_pose(defect):
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
    from cais_spade_llm.resources.recovery_geometry import _owner_product_completion

    kwargs, trajectory = _case()
    # A supported owner endpoint lies inside the existing seating corridor;
    # it is retained rather than snapped to the part's nominal coordinates.
    trajectory["points"][-1]["positions"] = [1e-7]
    kwargs["target_tool_pose"][2] = 1e-7
    prepared = prepare_mating_contact(**kwargs)
    proof = _proof(kwargs, trajectory, prepared)
    assert proof["status"] == "supported", proof
    native = _native(prepared, trajectory)
    state = deepcopy(kwargs["state"])
    rid, part = kwargs["resource_id"], kwargs["part_name"]
    pose = [0., 0., 1e-7, 0., 0., 0., 1.]
    state["resources"][rid].update(joint_positions=[1e-7], current_pose=pose)
    state["parts"][part]["current_pose"] = pose
    support = projected_fixture_handoff(resource_id=rid, part_name=part, state=state,
        evidence=kwargs["evidence"], owner_evidence=kwargs["owner_evidence"], mating_contact=prepared,
        endpoint_evidence={"mating_start_state": kwargs["state"], "mating_native_record": native,
                           "mating_corridor_evidence": proof})
    assert support["status"] == "supported", support
    state["resources"][rid]["held_part"] = None
    state["parts"][part].update(processCompleted=[], processCompleted_complete=True,
        processCompleted_evidence={"complete": True, "source_kind": "recorded", "checkpoint": "capture"})
    target = "Gear_Plate/Gear_Shaft_1"
    event = {"event_name": "place_insert", "function_name": "place_insert", "event_id": 3,
        "updates": {"part_location": {"set": "assembly_board-v1"}},
        "parameter_bindings": {"resource_id": {"equals": rid}, "destination_location": {"equals": "assembly_board-v1"},
            "part_name": {"reference": "part_name"}, "target": {"type": "string"}},
        "product_effects": {"state": "assembled", "target": {"set_from_param": "target"},
            "processCompleted": [{"process": "assembly", "target": {"set_from_param": "target"}}]}}
    geometry = {"parts": {part: {"target": target}}}
    evidence = {"environment_models": {rid: {"events": [event]}}, "product_name": "assembly_board-v1",
                "requirements": {part: [{"state": "assembled", "target": target}]}}
    policy = deepcopy(prepared["simulation_mating_contact"])
    contract = {"version": 6, "owner_declarations": [{"event_name": "place_insert"}], "placement_target": {
        "status": "calculated", "destination_location": "assembly_board-v1", "target": target,
        "target_reference": {"surface_role": "assembly_slot", "target_point": "inserted_part_origin"},
        "target_origin_pose": kwargs["target_origin_pose"], "simulation_mating_contact": policy}}
    if defect == "legacy": contract["version"] = 5
    elif defect == "wrong_target": support["target_origin_pose"] = {"x": 0., "y": 0., "z": .1}
    elif defect == "wrong_policy": support["policy_fingerprint"] = recovery_validation_fingerprint({})
    elif defect == "wrong_pose": support["current_pose"] = [0., 0., 0., 0., 0., 0., 1.]
    elif defect == "unretained": support["retained_fixture"] = False
    elif defect == "unseated": support["seated"] = False
    before = deepcopy(state)
    result = _owner_product_completion(task={"part_name": part, "resource_jid": rid, "outline_id": "recorded_place"},
        state=state, geometry=geometry, evidence=evidence, contract=contract, rid=rid, support={part: support})
    assert (result is not None) == (defect is None)
    assert state == before
    if result:
        assert result["kind"] == "predicted"
        assert result["product_effects"][part]["processCompleted"] == [{"process": "assembly", "target": target}]


def test_mating_reuses_fixed_origins_with_exact_fk_outputs(monkeypatch):
    from cais_spade_llm.resources import recovery_mating_contact as module

    kwargs, trajectory = _case()
    original = module.kinematic_transforms
    reused = []

    def checked(configuration, positions, origins=None):
        actual = original(configuration, positions, origins)
        expected = original(configuration, positions)
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
        if origins is not None:
            reused.append(origins)
        return actual

    monkeypatch.setattr(module, "kinematic_transforms", checked)
    proof = _proof(kwargs, trajectory, prepare_mating_contact(**kwargs))
    assert proof["status"] == "supported", proof
    assert len(reused) >= 2
    assert all(row is reused[0] for row in reused)


def test_private_mating_cache_reuses_only_local_proofs_and_returns_detached_records(monkeypatch):
    from cais_spade_llm.resources import recovery_mating_contact as module

    kwargs, trajectory = _case()
    calls = []
    original = module._corridor_bounds

    def checked(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, "_corridor_bounds", checked)
    prepared = prepare_mating_contact(**kwargs)
    first = _proof(kwargs, trajectory, prepared)
    expected, count = deepcopy(first), len(calls)
    assert first["status"] == "supported" and count
    first["cells"].clear()
    first["refinement_cells"] = -1
    assert _proof(kwargs, trajectory, prepared) == expected
    assert len(calls) == count
    second = _proof(kwargs, trajectory, prepared)
    second["binding"].clear()
    assert _proof(kwargs, trajectory, prepared) == expected
    assert module._mating_corridor_cache["result"] == expected


@pytest.mark.parametrize("change", ["state", "peer", "owner", "model", "physics", "scene", "trajectory", "limit", "calculation"])
def test_private_mating_cache_requires_fresh_dependencies(change, monkeypatch):
    from cais_spade_llm.resources import recovery_mating_contact as module

    kwargs, trajectory = _case()
    original, calls = module._corridor_bounds, []

    def checked(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, "_corridor_bounds", checked)
    assert _proof(kwargs, trajectory, prepare_mating_contact(**kwargs))["status"] == "supported"
    count = len(calls)
    if change == "state": kwargs["state"]["revision"] = 2
    elif change == "peer": kwargs["state"]["resources"]["ur5e-4"]["current_pose"][0] = 2.
    elif change == "owner": kwargs["owner_evidence"]["source"]["capture"] = "new"
    elif change == "model": kwargs["evidence"]["resource_models"]["ur5e-3"]["configuration"]["revision"] = 2
    elif change == "physics": kwargs["evidence"]["physics_evidence"]["instance_id"] = "new"
    elif change == "scene": kwargs["evidence"]["scene_object_bindings"]["gear_small"]["revision"] = 2
    elif change == "trajectory":
        trajectory["points"][-1]["time_from_start"]["sec"] = 2
        trajectory["duration_ns"] = 2_000_000_000
    elif change == "limit": monkeypatch.setattr(module, "_LIMIT", 4097)
    elif change == "calculation": monkeypatch.setattr(module, "_corridor_bounds", lambda *args: checked(*args))
    result = _proof(kwargs, trajectory, prepare_mating_contact(**kwargs))
    assert result["status"] == "supported", result
    assert len(calls) > count


def test_private_mating_cache_cannot_be_seeded_by_supplied_proof(monkeypatch):
    from cais_spade_llm.resources import recovery_mating_contact as module

    kwargs, trajectory = _case()
    prepared = prepare_mating_contact(**kwargs)
    prepared["mating_corridor_evidence"] = {"status": "supported", "continuous_mating_corridor_verified": True}
    trajectory["points"][0]["velocities"] = [-.2]
    trajectory["points"][-1]["velocities"] = [.2]
    proof = _proof(kwargs, trajectory, prepared)
    assert proof["status"] == "rejected", proof
    assert module._mating_corridor_cache is None
    assert _proof(kwargs, trajectory, prepared) == proof
    assert module._mating_corridor_cache is None


@pytest.mark.parametrize("condition", ["axis_tolerance_m", "target_origin_pose.z", "start_part_z", "upright", "yaw_tolerance_rad"])
def test_mating_counterexample_requires_separated_outward_bounds(condition):
    from cais_spade_llm.resources.continuous_motion import quaternion_transform
    from cais_spade_llm.resources.recovery_mating_contact import _corridor_violations

    kwargs, _ = _case()
    prepared = prepare_mating_contact(**kwargs)
    matrix = quaternion_transform([0., 0., .005], [0., 0., 0., 1.])
    assert _corridor_violations(matrix, prepared) == []
    if condition == "axis_tolerance_m": matrix[0][3] = (.0006, .0007)
    elif condition == "target_origin_pose.z": matrix[2][3] = (-.001, -.0006)
    elif condition == "start_part_z": matrix[2][3] = (.0106, .011)
    elif condition == "upright": matrix[2][2] = (.99, .991)
    elif condition == "yaw_tolerance_rad":
        matrix = quaternion_transform([0., 0., .005], [0., 0., math.sin(.01), math.cos(.01)])
    violations = _corridor_violations(matrix, prepared)
    assert condition in {row["condition"] for row in violations}
    for row in violations:
        if row["relation"] == ">": assert row["value_bounds"][0] > row["limit_bounds"][1]
        else: assert row["value_bounds"][1] < row["limit_bounds"][0]


def test_mating_uncertain_bounds_are_not_a_counterexample():
    from cais_spade_llm.resources.continuous_motion import quaternion_transform
    from cais_spade_llm.resources.recovery_mating_contact import _corridor_violations

    kwargs, _ = _case()
    matrix = quaternion_transform([0., 0., .005], [0., 0., 0., 1.])
    matrix[0][3] = (0., .001)
    matrix[2][3] = (-.001, .011)
    matrix[2][2] = (.99, 1.)
    matrix[1][0] = (-.1, .1)
    assert _corridor_violations(matrix, prepare_mating_contact(**kwargs)) == []
