from __future__ import annotations

"""Exact recorded nominal-program coverage without controllers or execution."""

from copy import deepcopy
import json

import pytest

from cais_spade_llm.resources.robot.recovery_task_projection import (
    apply_nominal_task_effects,
    check_nominal_task_effects,
    nominal_step_condition,
    nominal_task_bindings,
    nominal_task_projection,
    project_nominal_snap,
    resolve_nominal_task_step,
)


def _pose(x=0., z=0.):
    return {"x": x, "y": 0., "z": z, "qx": 0., "qy": 0., "qz": 0., "qw": 1.}


def _case(steps, intervals):
    model = {"events": [{"event_name": "recorded_task", "composition_contract": {
        "params": {}, "steps": steps, "effects": []}}]}
    rows = [{"step_id": step["id"], "state_before": {"_position": _pose(),
             "_held_part": "KET4_Square_4mm", "_task_ctx": {}},
             "interval_indices": [index] if index < len(intervals) else []}
            for index, step in enumerate(steps)]
    return {"task": {"required_task_contract": "recorded_task", "params": {}},
            "recovery_des_model": model, "resource_evidence": {"execution_mode": "simulation"},
            "step_evidence": rows, "intervals": intervals}


def _motion(target):
    return {"motion_request": {"mode": "cartesian", "waypoints": [{
        "position": {key: target[key] for key in ("x", "y", "z")},
        "orientation": {key: target["q" + key] for key in ("x", "y", "z", "w")}}]},
        "joint_trajectory": {"joint_names": ["joint"], "points": [{"positions": [0.]}]},
        "native_preview_binding": {"scene_fingerprint": "verified-scene"}}


def test_exact_nominal_program_covers_each_declared_step_and_preserves_source():
    steps = [{"id": "release", "op": "release_part", "params": {"part_name": "KET4_Square_4mm"}},
             {"id": "lift", "op": "move_relative", "params": {"dx": 0., "dy": 0., "dz": .2}}]
    intervals = [{"custody_effects": [{"kind": "release", "part": "KET4_Square_4mm"}],
                  "interval_complete": True}, _motion(_pose(z=.2))]
    kwargs = _case(steps, intervals)
    before = deepcopy(kwargs)
    result = nominal_task_projection(**kwargs)
    assert result["status"] == "supported"
    assert [step["step_id"] for step in result["steps"]] == ["release", "lift"]
    assert all(len(step["effect_evidence_fingerprint"]) == 64 for step in result["steps"])
    assert kwargs == before


@pytest.mark.parametrize("name,expected", [("square", "supported"), ("other", "NEEDS_CONTEXT"), (None, "NEEDS_CONTEXT")])
def test_nominal_custody_keeps_explicit_part_and_model_binding(name, expected):
    step = {"id": "release", "op": "release_part", "params": {"part_name": "KET4_Square_4mm", "model_name": name}}
    kwargs = _case([step], [{"custody_effects": [{"kind": "release", "part": "KET4_Square_4mm"}], "interval_complete": True}])
    kwargs["resource_evidence"]["inputs"] = {"geometry": {"parts": {"model_map": {"KET4_Square_4mm": "square"}}}}
    assert nominal_task_projection(**kwargs)["status"] == expected


@pytest.mark.parametrize("defect", ["omitted_snap", "unknown_active", "null_part", "wrong_target", "unclaimed_interval"])
def test_nominal_program_cannot_hide_effectful_steps_or_substitute_values(defect):
    step = {"id": "release", "op": "release_part", "params": {"part_name": "KET4_Square_4mm"}}
    interval = {"custody_effects": [{"kind": "release", "part": "KET4_Square_4mm"}], "interval_complete": True}
    kwargs = _case([step], [interval])
    if defect == "omitted_snap":
        kwargs["recovery_des_model"]["events"][0]["composition_contract"]["steps"].append(
            {"id": "hidden_snap", "op": "snap_part_to_slot", "params": {"slot_x": .3}})
    elif defect == "unknown_active":
        step["op"] = "unmodelled_owner_operation"
    elif defect == "null_part":
        step["params"]["part_name"] = None
    elif defect == "wrong_target":
        step.update(op="move_cartesian", params=_pose(x=.3))
        kwargs["intervals"] = [_motion(_pose())]
    else:
        kwargs["intervals"].append(_motion(_pose(z=.2)))
    assert nominal_task_projection(**kwargs)["status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("value,status", [(None, "not_applicable"), ("square", "applicable")])
def test_nominal_guard_distinguishes_recorded_null_from_missing(value, status):
    step = {"when": [{"predicate": "task_ctx_key_truthy", "args": {"key": "model_name"}}]}
    assert nominal_step_condition(step=step, bindings={}, state={"_task_ctx": {"model_name": value}},
                                  resource_evidence={})["status"] == status
    missing = nominal_step_condition(step=step, bindings={}, state={"_task_ctx": {}}, resource_evidence={})
    assert missing["status"] == "NEEDS_CONTEXT"
    assert missing["facts"][0]["source"]["available"] is False


def test_skipped_nominal_step_cannot_have_an_interval_or_effect():
    step = {"id": "localize", "op": "detect_parts", "params": {},
            "when": [{"predicate": "execution_mode", "args": {"mode": "physical"}}]}
    kwargs = _case([step], [])
    assert nominal_task_projection(**kwargs)["steps"][0]["status"] == "skipped"
    kwargs["step_evidence"][0]["effect_evidence"] = {"invented": True}
    assert nominal_task_projection(**kwargs)["status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("field,private,value", [("held_part", "_held_part", None),
    ("resource_state", "_current_state", "placed"), ("resource_location", "_recovery_pose_ref", "assembly_board-v1"),
    ("gripper_state", "_gripper_state", "open")])
def test_nominal_condition_reads_exact_owner_public_field_declarations(field, private, value):
    step = {"when": [{"condition": {"field": field, "operator": "equals", "value": value}}]}
    assert nominal_step_condition(step=step, bindings={}, state={private: value}, resource_evidence={})["status"] == "applicable"
    assert nominal_step_condition(step=step, bindings={}, state={}, resource_evidence={})["status"] == "NEEDS_CONTEXT"


def test_nominal_simulation_insert_resolves_only_arguments_used_by_its_owner_branch():
    step = {"id": "insert", "op": "move_insert", "params": {
        "part_name": "KET4_Square_4mm", "expected_start_pose": _pose(z=.1),
        "target_pose": _pose(), "trial_id": "", "profile_sha256": {"$state": "missing_profile"}}}
    result = resolve_nominal_task_step(step=step, bindings={}, state={}, resource_evidence={"execution_mode": "simulation"})
    assert result["status"] == "resolved" and "profile_sha256" not in result["params"]
    assert result["unused_parameters"]["profile_sha256"]["expression"] == {"$state": "missing_profile"}
    step["params"]["unknown_new_argument"] = None
    assert resolve_nominal_task_step(step=step, bindings={}, state={}, resource_evidence={"execution_mode": "simulation"})["status"] == "NEEDS_CONTEXT"


def test_nominal_trace_identifier_records_owner_formatting_without_filling_motion_facts():
    step = {"id": "insert", "op": "move_insert", "params": {
        "part_name": "KET4_Square_4mm", "expected_start_pose": _pose(z=.1), "target_pose": _pose(),
        "trial_id": {"$format": "{trial_id}", "values": {
            "trial_id": {"$state": "_task_ctx", "path": ["move_insert_trial_id"]}}}}}
    args = {"step": step, "bindings": {}, "state": {"_task_ctx": {}}, "resource_evidence": {"execution_mode": "simulation"}}
    result = resolve_nominal_task_step(**args)
    assert result["status"] == "resolved" and result["params"]["trial_id"] == ""
    assert result["parameter_evidence"]["trial_id"]["source_field_present"] is False
    step["params"]["target_pose"] = {"$state": "_task_ctx", "path": ["insert_pose"]}
    assert resolve_nominal_task_step(**args)["status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("context,status", [({}, "NEEDS_CONTEXT"), ({"surface_role": "buffer"}, "not_applicable"),
    ({"surface_role": "assembly_slot", "move_insert_mode": "simulation_direct"}, "applicable"),
    ({"surface_role": "assembly_slot", "move_insert_mode": None}, "not_applicable")])
def test_nominal_insert_required_guard_retains_absence_and_explicit_null(context, status):
    result = nominal_step_condition(step={"when": [{"predicate": "move_insert_required"}]},
        bindings={}, state={"_task_ctx": context}, resource_evidence={"execution_mode": "simulation"})
    assert result["status"] == status


@pytest.mark.parametrize("start_z,expected", [(0., "supported"), (.02, "NEEDS_CONTEXT")])
def test_nominal_insert_checks_start_against_existing_owner_tolerance(start_z, expected):
    step = {"id": "insert", "op": "move_insert", "params": {"part_name": "KET4_Square_4mm",
        "expected_start_pose": _pose(z=start_z), "target_pose": _pose(z=-.01), "trial_id": ""}}
    kwargs = _case([step], [_motion(_pose(z=-.01))])
    kwargs["resource_evidence"]["controller_config"] = {"motion": {"cartesian_position_tolerance_m": .005}}
    assert nominal_task_projection(**kwargs)["status"] == expected


def test_nominal_named_motion_needs_exact_declared_joint_target():
    step = {"id": "home", "op": "move_to_named_pose", "params": {"pose_name": "home"}}
    interval = {"motion_request": {"mode": "joint_target", "joint_target": {"name": ["joint"], "position": [.2]}},
                "joint_trajectory": {"joint_names": ["joint"], "points": [{"positions": [.2]}]},
                "native_preview_binding": {"model_fingerprint": "recorded-model"}}
    kwargs = _case([step], [interval])
    kwargs["resource_model"] = {"named_joint_poses": {"home": {"joint_names": ["joint"], "positions": [.2]}}}
    assert nominal_task_projection(**kwargs)["status"] == "supported"
    interval["joint_trajectory"]["points"][-1]["positions"] = [.3]
    assert nominal_task_projection(**kwargs)["status"] == "NEEDS_CONTEXT"


def test_nominal_defaults_require_matching_registered_argument_declaration():
    from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry

    registered = robot_task_registry()["place_approach"]
    program = {"params": {row.name: row.param_schema() for row in registered.arguments},
               "required_params": [row.name for row in registered.arguments if row.required]}
    model = {"events": [{"event_name": "place_approach", "composition_contract": program}]}
    task = {"required_task_contract": "place_approach", "params": {}}
    result = nominal_task_bindings(task=task, recovery_des_model=model)
    assert result["bindings"]["speed"] is None
    assert result["default_evidence"]["speed"]["recorded_argument_was_present"] is False
    program["params"]["new_required_fact"] = {"type": "number"}
    assert "speed" not in nominal_task_bindings(task=task, recovery_des_model=model)["bindings"]


def test_nominal_effects_keep_physical_pose_and_skip_only_proven_absent_outputs():
    kwargs = _case([], [])
    kwargs["recovery_des_model"]["events"][0]["composition_contract"]["effects"] = [
        {"target": "position", "action": "set", "value": {"x": 0., "y": 0., "z": 445.}},
        {"target": "task_ctx", "action": "merge", "skip_empty_values": True,
         "value": {"insert_pose": {"$step": "placement", "path": ["insert_pose"]},
                   "profile": {"$step": "placement", "path": ["move_insert_profile"]}}}]
    args = {key: kwargs[key] for key in ("task", "recovery_des_model", "resource_evidence")}
    args.update(state={"_position": _pose(z=1.2), "_task_ctx": {}}, step_outputs={"placement": {"insert_pose": _pose()}})
    assert apply_nominal_task_effects(**args)["status"] == "NEEDS_CONTEXT"
    result = apply_nominal_task_effects(**args, absent_output_fields={"placement": ["move_insert_profile"]})
    assert result["status"] == "supported"
    assert result["state"]["_position"] == _pose(z=1.2)
    assert result["state"]["_task_ctx"] == {"insert_pose": _pose()}
    assert apply_nominal_task_effects(**args, absent_output_fields={"placement": ["another_field"]})["status"] == "NEEDS_CONTEXT"


def _snap_case():
    part, rid, pose = "KET4_Square_4mm", "ur5e-3", [0., 0., 0., 0., 0., 0., 1.]
    state = {"frame": "world", "resources": {rid: {"held_part": None}, "ur5e-4": {"held_part": None}},
             "parts": {part: {"model_name": "square", "current_pose": pose[:]}}}
    physics = {"instance_id": "physics", "simulation_time": 2., "gravity": [0., 0., -9.81],
        "physics_engine": "ode", "contact_surface_layer": 0., "models": {
            "square": {"static": False, "pose": pose[:], "links": {"link": {
                "mass": .1, "pose": pose[:], "center_of_mass": [0., 0., 0.],
                "linear_velocity": [0., 0., 0.], "angular_velocity": [0., 0., 0.],
                "collisions": [{"id": "square::collision", "pose": pose[:], "contact_min_depth": 0.,
                                "geometry": {"type": "box", "size": [.1, .1, .1]}}]}}},
            "table": {"static": True, "links": {"link": {"collisions": [{"id": "table::collision",
                "pose": [0., 0., -.1, 0., 0., 0., 1.], "contact_min_depth": 0.,
                "geometry": {"type": "box", "size": [1., 1., .1]}}]}}},
            "assembly_board-v1": {"static": True, "links": {"link": {"collisions": []}}}}, "contacts": []}
    ros_pose = {"position": {"x": 0., "y": 0., "z": 0.}, "orientation": {"x": 0., "y": 0., "z": 0., "w": 1.}}
    part_object = {"id": "part/link/collision", "header": {"frame_id": "world"},
                   "primitives": [{"type": 1, "dimensions": [.1, .1, .1]}], "primitive_poses": [deepcopy(ros_pose)]}
    table_object = {"id": "support/link/collision", "header": {"frame_id": "world"},
                   "primitives": [{"type": 1, "dimensions": [1., 1., .1]}],
                   "primitive_poses": [{**deepcopy(ros_pose), "position": {"x": 0., "y": 0., "z": -.1}}]}
    unrelated = deepcopy(table_object)
    unrelated.update(id="unrelated/link/collision")
    unrelated["primitive_poses"][0]["position"]["x"] = 3.
    scene = {"is_diff": False, "robot_state": {"is_diff": False,
        "joint_state": {"name": ["peer_joint"], "position": [.2]}, "attached_collision_objects": []},
        "world": {"collision_objects": [part_object, table_object, unrelated]}}
    evidence = {"physics_evidence": physics, "resource_models": {}, "resources": {},
        "part_support_policy": {"kind": "gravity_supported_rest", "requires_no_active_goals": True,
            "future_execution_tracking": "not_established", "stationary_linear_speed_max": .001,
            "stationary_angular_speed_max": .001},
        "scene_object_bindings": {part: {"source": "recorded geometry", "objects": [{
            "object_id": part_object["id"], "part_local_object": deepcopy(part_object)}]}},
        "support_collision_object_bindings": {
            "square::collision": {"object_id": part_object["id"], "source": "recorded geometry"},
            "table::collision": {"object_id": table_object["id"], "source": "recorded geometry"}},
        "planning_scene_capture": {"planning_scene": scene, "model_fingerprint": "model"},
        "planning_scene_configuration": {"robots": []}}
    owner = {"source": "recorded owner", "execution_mode": "simulation", "controller_config": {
        "services": {"attach": "/ATTACHLINK", "detach": "/DETACHLINK"},
        "move_group": {"group_name": "arm", "ee_link": "tool", "frame_id": "world"}},
        "inputs": {"geometry": {"assembly_board": {
            "component_attachment_model": "assembly_board-v1", "model_name": "assembly_board-v1",
            "center": {"x": 0., "y": 0., "z": 0.}, "slots": {part: [0., 0.]},
            "slot_floor_z_m": -.05, "target_origin_z_m": {part: 0.}},
            "parts": {"model_map": {part: "square"}, "heights_m": {part: .1}}}}}
    params = {"model_name": "square", "slot_x": 0., "slot_y": 0., "part_height": .1,
              "board_top_z": -.05, "part_origin_z": 0., "destination_location": "assembly_board-v1"}
    return {"resource_id": rid, "params": params, "state": state, "evidence": evidence, "owner_evidence": owner}


def _snap_native_result(kwargs, *, mutate_scene=None, rejected=False):
    from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_scene_endpoint, project_preview_scene

    candidate = project_nominal_snap(**kwargs)
    assert "candidate_state" in candidate, candidate
    scene = project_preview_scene(kwargs["evidence"]["planning_scene_capture"],
        {**kwargs["evidence"], "snapshot": candidate["candidate_state"]})
    if mutate_scene is not None:
        mutate_scene(scene)
    request = {**scene, "group_name": "arm", "link_name": "tool", "frame_id": "world",
        "resource_id": "ur5e-3", "capture_fingerprint": "capture", "proposal_fingerprint": "proposal",
        "scene_fingerprint": "scene", "model_fingerprint": "model", "max_translation_step": .01,
        "max_rotation_step": .01, "max_joint_step": .01, "velocity_scaling": .1,
        "acceleration_scaling": .1, "timeout_seconds": 10.}
    request = bind_scene_endpoint(request, candidate["scene_endpoint_record"])
    contact = {"shape_1": 0, "shape_2": 0, "depth": .01 if rejected else 0.,
               "position": [0., 0., -.05], "normal": [0., 0., 1.]}
    audit = {"version": 1, "source": "PreviewRecoveryMotion", "endpoint_only": True,
        "pairs_complete": True, "continuous_collision_verified": False,
        "request_binding": {key: request[key] for key in (
            "resource_id", "capture_fingerprint", "proposal_fingerprint", "scene_fingerprint", "model_fingerprint")},
        "scene_endpoint_fingerprint": request["scene_endpoint_fingerprint"],
        "changed_object_ids": ["part/link/collision"],
        "world_object_ids": sorted(row["id"] for row in request["planning_scene"]["world"]["collision_objects"]),
        "robot_collision_checked": True, "robot_collision_free": True, "robot_contacts": [], "checked_contacts": 1,
        "support_pairs": [{**pair, "minimum_signed_distance": 0.} for pair in candidate["scene_endpoint_record"]["support_pairs"]],
        "checked_pairs": [{"body_1": "part/link/collision", "body_2": "support/link/collision", "shape_pairs": 1,
                           "allowed_support_tangency": True, "contacts": [contact]}],
        "collision_free": not rejected}
    if "unrelated/link/collision" in audit["world_object_ids"]:
        audit["checked_pairs"].append({"body_1": "part/link/collision", "body_2": "unrelated/link/collision",
                                      "shape_pairs": 1, "allowed_support_tangency": False, "contacts": []})
    return {"request": request, **audit["request_binding"], "status": "SCENE_ENDPOINT_REJECTED" if rejected else "SCENE_ENDPOINT_COMPLETE",
        "reason": "Recorded endpoint collision" if rejected else "", "path_complete": False,
        "fraction": 0., "sampled_collision_rejected": False, "continuous_collision_verified": False,
        "trajectory": {}, "scene_endpoint_fingerprint": request["scene_endpoint_fingerprint"],
        "scene_endpoint_evidence_json": json.dumps(audit)}


def test_nominal_snap_is_an_uncommitted_prediction_until_complete_endpoint_check():
    kwargs = _snap_case()
    before = deepcopy(kwargs)
    candidate = project_nominal_snap(**kwargs)
    assert candidate["status"] == "NEEDS_CONTEXT" and "candidate_state" in candidate, candidate
    assert "projected_snapshot" not in candidate
    supported = project_nominal_snap(**kwargs, endpoint_result=_snap_native_result(kwargs))
    assert supported["status"] == "supported", supported
    assert supported["projected_snapshot"]["resources"] == kwargs["state"]["resources"]
    assert supported["physical_execution_verified"] is False
    assert kwargs == before


def test_nominal_snap_accepts_serialized_empty_ros_trajectory_but_no_motion_claim():
    kwargs = _snap_case()
    response = _snap_native_result(kwargs)
    response["trajectory"] = {"joint_trajectory": {"header": {"frame_id": ""}, "joint_names": [], "points": []}}
    assert project_nominal_snap(**kwargs, endpoint_result=response)["status"] == "supported"
    response["trajectory"]["joint_trajectory"]["points"] = [{"positions": [0.]}]
    assert project_nominal_snap(**kwargs, endpoint_result=response)["status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("defect", ["unsupported_before", "held", "geometry", "missing_frame"])
def test_nominal_snap_does_not_retroactively_supply_support_or_invent_geometry(defect):
    kwargs = _snap_case()
    if defect == "unsupported_before":
        kwargs["state"]["parts"]["KET4_Square_4mm"]["current_pose"][2] = .02
    elif defect == "held":
        kwargs["state"]["resources"]["ur5e-4"]["held_part"] = "KET4_Square_4mm"
    elif defect == "geometry":
        kwargs["params"]["slot_x"] = .3
    else:
        kwargs["state"].pop("frame")
    result = project_nominal_snap(**kwargs)
    assert result["status"] == "NEEDS_CONTEXT" and "candidate_state" not in result


@pytest.mark.parametrize("defect", ["peer_joint", "dropped_object", "changed_shape", "frame"])
def test_nominal_snap_reconstructs_complete_scene_instead_of_trusting_rebound_metadata(defect):
    kwargs = _snap_case()

    def mutate(scene):
        if defect == "peer_joint":
            scene["planning_scene"]["robot_state"]["joint_state"]["position"][0] = .8
            scene["start_state"]["joint_state"]["position"][0] = .8
        elif defect == "dropped_object":
            scene["planning_scene"]["world"]["collision_objects"] = [row for row in scene["planning_scene"]["world"]["collision_objects"]
                                                                      if row["id"] != "unrelated/link/collision"]
        elif defect == "changed_shape":
            scene["planning_scene"]["world"]["collision_objects"][0]["primitives"][0]["dimensions"][0] = .01
        else:
            scene["planning_scene"]["world"]["collision_objects"][0]["header"]["frame_id"] = "other"

    record = _snap_native_result(kwargs, mutate_scene=mutate)
    result = project_nominal_snap(**kwargs, endpoint_result=record)
    assert result["status"] == "NEEDS_CONTEXT" and "complete independently projected candidate" in result["reason"]
    assert "projected_snapshot" not in result


def test_nominal_snap_retains_complete_native_rejection_without_committing_candidate():
    kwargs = _snap_case()
    result = project_nominal_snap(**kwargs, endpoint_result=_snap_native_result(kwargs, rejected=True))
    assert result["status"] == "rejected", result
    assert result["scene_endpoint_rejected"] is True
    assert result["endpoint_evidence"]["collision_free"] is False
    assert "projected_snapshot" not in result


@pytest.mark.parametrize("target,conflict", [("current_state", "failed"),
    ("held_part", "gear_small"), ("gripper_state", "closed")])
def test_nominal_state_effect_conflict_cannot_be_discarded_after_program_validation(target, conflict):
    expected = {"current_state": "placed", "held_part": None, "gripper_state": "open"}
    effects = {"status": "supported", "effects": [{"target": target, "status": "supported"}],
               "state": {"_" + target: conflict}}
    with pytest.raises(ValueError, match="conflicts with supported"):
        check_nominal_task_effects(nominal_effects=effects,
            resource_updates={"resource_state": "placed", "held_part": None, "gripper_state": "open"},
            physical_resource={"held_part": None, "gripper_state": "open"})
    effects["state"]["_" + target] = expected[target]
    assert check_nominal_task_effects(nominal_effects=effects,
        resource_updates={"resource_state": "placed", "held_part": None, "gripper_state": "open"},
        physical_resource={"held_part": None, "gripper_state": "open"})["status"] == "supported"


@pytest.mark.parametrize("missing", ["owner_value", "physical_value", "public_state"])
def test_nominal_state_effect_check_distinguishes_missing_from_explicit_null(missing):
    effects = {"status": "supported", "effects": [{"target": "held_part", "status": "supported"}],
               "state": {"_held_part": None}}
    public, physical = {"resource_state": "placed", "held_part": None}, {"held_part": None}
    if missing == "owner_value":
        effects["state"] = {}
    elif missing == "physical_value":
        physical = {}
    else:
        effects = {"status": "supported", "effects": [{"target": "current_state", "status": "supported"}],
                   "state": {"_current_state": None}}
        public = {}
    with pytest.raises(ValueError, match="no resolved value|lacks supported"):
        check_nominal_task_effects(nominal_effects=effects, resource_updates=public, physical_resource=physical)


@pytest.mark.parametrize("name,public,physical", [
    ("move_home", {"resource_state": "idle"}, {"held_part": None, "gripper_state": "open"}),
    ("place_insert", {"resource_state": "placed", "held_part": None, "gripper_state": "open"},
     {"held_part": None, "gripper_state": "open"})])
def test_registered_nominal_state_effects_agree_with_supported_results(name, public, physical):
    from dataclasses import asdict
    from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry

    registered = robot_task_registry()[name]
    program = json.loads(json.dumps(asdict(registered.program)))
    program["params"] = registered.argument_properties()
    program["required_params"] = registered.required_argument_names()
    effects = apply_nominal_task_effects(task={"required_task_contract": name, "params": {}},
        recovery_des_model={"events": [{"event_name": name, "composition_contract": program}]},
        resource_evidence={"execution_mode": "simulation"},
        state={"_task_ctx": {}, "_position": _pose(z=1.2)}, step_outputs={})
    assert effects["status"] == "supported"
    before = deepcopy((effects, public, physical))
    checked = check_nominal_task_effects(nominal_effects=effects, resource_updates=public, physical_resource=physical)
    assert checked["status"] == "supported"
    assert (effects, public, physical) == before
