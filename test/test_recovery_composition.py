from __future__ import annotations

"""Contract-level regression coverage for generalized ordered composition."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cais_spade_llm.resources.recovery_composition import (
    composition_program_error, composition_start_views, resolve_composition_value,
    validate_recovery_primitive_composition,
)


def feasible(**kwargs):
    return {"allowed": True, "feasibility_status": "FEASIBLE", "params": kwargs["params"]}


@pytest.fixture
def contract():
    return {"task": {"outline_id": "restore", "event_name": "restore", "resource_jid": "owner",
                "part_name": "P", "expected_start_state": {"mode": "fault", "quality": "pending"},
                "expected_end_state": {"mode": "ready", "quality": "checked"}},
        "recovery_des_model": {"state_variables": {"mode": {"scope": "resource"},
                                                    "quality": {"scope": "part"}}},
        "recovery_snapshot": {"mode": "fault"}, "part_context": {"quality": "pending"},
        "grounding_context": {"limit": 3}, "validate_primitive": feasible,
        "primitive_catalog": [
            {"name": "reset", "params": {}, "preconditions": {"mode": {"equals": "fault"}},
             "effects": {"mode": {"set": "idle"}}},
            {"name": "qualify", "params": {"count": {"type": "integer", "minimum": 1, "maximum": 3}},
             "required_params": ["count"], "preconditions": {"mode": {"equals": "idle"}},
             "effects": {"mode": {"set": "ready"}, "quality": {"set": "checked"}}}],
        "primitive_steps": [{"primitive": "reset", "params": {}},
                            {"primitive": "qualify", "params": {"count": {"context_ref": "/limit"}}}]}


@pytest.mark.parametrize("resource_type", ["machine", "Conveyor", "kmr", "unfamiliar_resource"])
def test_owner_fields_and_product_effects_without_robot_state(contract, resource_type):
    contract["recovery_snapshot"]["resource_type"] = resource_type
    before = deepcopy(contract)
    result = validate_recovery_primitive_composition(**contract)
    assert result["feasibility_status"] == "FEASIBLE"
    assert result["derived_end_state"] == {"mode": "ready", "quality": "checked"}
    assert result["trace"][1]["start_snapshot"]["mode"] == "idle"
    assert result["trace"][1]["resolved_params"] == {"count": 3}
    assert result["execution_authorized"] is False
    assert contract == before


@pytest.mark.parametrize("change, status, step", [
    ("order", "INFEASIBLE", 0), ("reference", "NEEDS_CONTEXT", 1),
    ("params", "INFEASIBLE", 1), ("extra_param", "INFEASIBLE", 1),
    ("identity", "INFEASIBLE", 0), ("effects", "INFEASIBLE", None),
    ("missing_effects", "NEEDS_CONTEXT", 1), ("missing_contract", "NEEDS_CONTEXT", None),
])
def test_rejections_do_not_substitute_requested_postconditions(contract, change, status, step):
    if change == "order":
        contract["primitive_steps"].reverse()
    elif change == "reference":
        contract["grounding_context"] = {}
    elif change == "params":
        contract["grounding_context"]["limit"] = 20
    elif change == "extra_param":
        contract["primitive_steps"][1]["params"]["other"] = 1
    elif change == "identity":
        contract["primitive_steps"][0]["primitive"] = "RESET"
    elif change == "effects":
        contract["primitive_catalog"][1]["effects"].pop("quality")
    elif change == "missing_effects":
        contract["primitive_catalog"][1].pop("effects")
    else:
        contract["recovery_des_model"] = {}
    result = validate_recovery_primitive_composition(**contract)
    assert result["feasibility_status"] == status
    assert result["findings"][0]["step_index"] == step
    assert result["allowed"] is False
    assert result["derived_end_state"] != contract["task"]["expected_end_state"]


def test_a_base_move_keeps_the_shape_of_the_recorded_pose(contract):
    # The KMR's move_base declares [x, y, yaw]; its recorded base_pose is position and quaternion.
    contract["primitive_catalog"].append({
        "name": "travel", "params": {"target_pose": {"type": "array", "items": {"type": "number"}}},
        "required_params": ["target_pose"], "preconditions": {},
        "effects": {"base_pose": {"pose_absolute_from_target_pose": ["x", "y", "yaw"]}}})
    contract["primitive_steps"].append({"primitive": "travel", "params": {"target_pose": [1.0, 2.0, 0.0]}})
    contract["recovery_snapshot"]["base_pose"] = [-5.54, 2.13, 0.1, 0.0, 0.0, 0.7071, 0.7071]
    result = validate_recovery_primitive_composition(**contract)
    assert result["feasibility_status"] == "FEASIBLE"
    assert result["trace"][2]["projected_snapshot"]["base_pose"] == [1.0, 2.0, 0.1, 0.0, 0.0, 0.0, 1.0]
    # Without a recorded pose there is no shape to keep, and a pose never lands on a non-pose field.
    contract["recovery_snapshot"].pop("base_pose")
    assert validate_recovery_primitive_composition(**contract)["feasibility_status"] == "NEEDS_CONTEXT"
    contract["primitive_catalog"][2]["effects"] = {"mode": {"pose_absolute_from_target_pose": ["x", "y", "yaw"]}}
    assert validate_recovery_primitive_composition(**contract)["feasibility_status"] == "NEEDS_CONTEXT"


def test_an_owner_composition_starts_a_missing_declared_fact_from_recorded_evidence(contract):
    from cais_spade_llm.resources.recovery_composition_owner import validate_owner_composition

    # A capture records the KMR's base_pose in its evidence, not in the composition prefix.
    travel = {"name": "travel", "params": {"target_pose": {"type": "array", "items": {"type": "number"}}},
              "required_params": ["target_pose"], "preconditions": {},
              "effects": {"base_pose": {"pose_absolute_from_target_pose": ["x", "y", "yaw"]}}}
    entry = {"recovery_des_model": contract["recovery_des_model"],
             "execution_primitive_catalog": [*contract["primitive_catalog"], travel],
             "resource_evidence": {"snapshot": {"base_pose": [-5.54, 2.13, 0.1, 0.0, 0.0, 0.7071, 0.7071],
                                                "mode": "recorded"}}}
    owner = SimpleNamespace(jid="owner", check_recovery_primitive_feasibility=lambda **_: {
        "allowed": True, "feasibility_status": "FEASIBLE"})
    steps = [*contract["primitive_steps"], {"primitive": "travel", "params": {"target_pose": [1.0, 2.0, 0.0]}}]
    result = validate_owner_composition(owner, task=contract["task"], primitive_steps=steps,
                                        prepared={"recovery_resources": {"owner": entry}},
                                        recovery_snapshot=contract["recovery_snapshot"],
                                        part_context=contract["part_context"],
                                        grounding_context=contract["grounding_context"])
    assert result["feasibility_status"] == "FEASIBLE"
    assert result["trace"][2]["projected_snapshot"]["base_pose"] == [1.0, 2.0, 0.1, 0.0, 0.0, 0.0, 1.0]
    # The prefix's own fact wins over the recorded one.
    assert result["trace"][0]["start_snapshot"]["mode"] == "fault"


def test_null_is_a_fact_and_missing_is_unknown(contract):
    contract["task"]["expected_start_state"]["mode"] = None
    contract["primitive_catalog"][0]["preconditions"]["mode"] = {"equals": None}
    contract["recovery_snapshot"]["mode"] = None
    assert validate_recovery_primitive_composition(**contract)["allowed"]
    contract["recovery_snapshot"].pop("mode")
    assert validate_recovery_primitive_composition(**contract)["feasibility_status"] == "NEEDS_CONTEXT"


def test_parameter_references_use_intermediate_part_effects(contract):
    contract["grounding_context"]["part"] = {"quality": "pending"}
    contract["primitive_catalog"][0]["effects"]["quality"] = {"set": "measured"}
    contract["primitive_catalog"][1]["params"]["quality"] = {"const": "measured"}
    contract["primitive_steps"][1]["params"]["quality"] = {"context_ref": "/part/quality"}
    assert validate_recovery_primitive_composition(**contract)["feasibility_status"] == "FEASIBLE"


@pytest.mark.parametrize("reference", ["/resource/measurement", "/resources/owner/measurement"])
def test_unknown_effect_clears_previously_available_reference(contract, reference):
    contract["recovery_des_model"]["state_variables"]["measurement"] = {"scope": "resource"}
    contract["recovery_snapshot"]["measurement"] = 3
    contract["grounding_context"]["resources"] = {"owner": {"measurement": 3}}
    contract["primitive_catalog"][0]["effects"]["measurement"] = {"set_unknown": True}
    contract["primitive_steps"][1]["params"]["count"] = {"context_ref": reference}
    result = validate_recovery_primitive_composition(**contract)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["findings"][0]["step_index"] == 1


def test_literal_identifiers_are_never_interpreted_as_references():
    assert resolve_composition_value("/owner/null", {}) == "/owner/null"
    assert resolve_composition_value({"context_ref": "/A B/value"}, {"A B": {"value": None}}) is None
    with pytest.raises(KeyError):
        resolve_composition_value({"context_ref": "/a b/value"}, {"A B": {"value": None}})


def test_declared_macro_effects_are_not_applied_to_incomplete_program(contract):
    contract["recovery_des_model"]["events"] = [{"event_name": "complete",
        "composition_contract": {"steps": [{"op": "extra"}]},
        "updates": {"quality": {"set": "checked"}}}]
    contract["primitive_catalog"][1]["effects"].pop("quality")
    result = validate_recovery_primitive_composition(**contract)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["projected_part_context"]["quality"] == "pending"


def _prepared(contract):
    return {"composition_validation_version": 1,
        "grounding_context": {"parts": {"P": deepcopy(contract["part_context"])}},
        "recovery_resources": {"owner": {"recovery_snapshot": deepcopy(contract["recovery_snapshot"]),
            "recovery_des_model": deepcopy(contract["recovery_des_model"]),
            "execution_primitive_catalog": deepcopy(contract["primitive_catalog"])}}}


def _row(contract):
    return {**deepcopy(contract["task"]), "primitive_steps": deepcopy(contract["primitive_steps"]),
            "composition_validation": validate_recovery_primitive_composition(**contract)}


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "wrong_owner", "wrong_program", "wrong_contract"])
def test_exact_coverage_and_current_evidence_bindings(contract, mutation):
    prepared, program = _prepared(contract), [_row(contract)]
    assert composition_program_error([contract["task"]], program, prepared=prepared, session={}) is None
    if mutation == "duplicate":
        program.append(deepcopy(program[0]))
    elif mutation == "missing":
        program = []
    elif mutation == "wrong_owner":
        program[0]["resource_jid"] = "another"
    elif mutation == "wrong_program":
        program[0]["primitive_steps"].reverse()
    else:
        prepared["recovery_resources"]["owner"]["recovery_des_model"]["revision"] = 2
    assert composition_program_error([contract["task"]], program, prepared=prepared, session={})


def test_an_owner_validated_program_binds_the_catalog_the_owner_checked(contract):
    from cais_spade_llm.resources.recovery_composition_owner import validate_owner_composition
    from cais_spade_llm.resources.resource_primitives import INTERNAL_GRIPPER_PRIMITIVES

    # A recorded robot or KMR catalog keeps its internal gripper controls; recovery hides them from the owner check.
    prepared = _prepared(contract)
    entry = prepared["recovery_resources"]["owner"]
    entry.update(resource_type="kmr", execution_primitive_catalog=[
        *contract["primitive_catalog"],
        {"name": sorted(INTERNAL_GRIPPER_PRIMITIVES)[0], "params": {}, "preconditions": {}, "effects": {}}])
    owner = SimpleNamespace(jid="owner", check_recovery_primitive_feasibility=lambda **_: {
        "allowed": True, "feasibility_status": "FEASIBLE"})
    evidence = validate_owner_composition(owner, task=contract["task"], primitive_steps=contract["primitive_steps"],
                                          prepared=prepared, recovery_snapshot=contract["recovery_snapshot"],
                                          part_context=contract["part_context"],
                                          grounding_context=contract["grounding_context"])
    assert evidence["feasibility_status"] == "FEASIBLE"
    row = {**deepcopy(contract["task"]), "primitive_steps": deepcopy(contract["primitive_steps"]),
           "composition_validation": evidence}
    assert composition_program_error([contract["task"]], [row], prepared=prepared, session={}) is None


def test_robot_pick_completion_reads_the_aligned_part_model(monkeypatch):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    # Since the aligned part model (October 23) pick_grasp updates part_location, never part_state.
    pick_grasp = {"event_name": "pick_grasp", "updates": {
        "resource_state": {"set": "picked"}, "held_part": {"set_from_param": "part_name"},
        "resource_location": {"set_from_param": "origin_resource_location"}, "part_location": {"set": "robot"}},
        "composition_contract": {"steps": [
            {"op": "grasp_part", "params": {"part_name": {"$arg": "part_name"}}},
            {"op": "move_relative", "params": {"dx": 0., "dy": 0., "dz": {"$sub": [
                {"$state": "_task_ctx", "path": ["travel_z"]}, {"$state": "_position", "path": ["z"]}]}}}]}}
    model = {"events": [pick_grasp], "state_variables": {
        "held_part": {"scope": "resource", "domain": [None, "P"]},
        "part_location": {"scope": "part", "domain": ["Exit", "robot"]},
        "resource_location": {"scope": "resource", "domain": ["Exit", "home"]},
        "resource_state": {"scope": "resource", "domain": ["at_pick", "picked"]}}}
    grasp = {"x": 0.1, "y": 0.2, "z": 0.3, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    retreat = {**grasp, "z": 0.4}
    target = {"target_pose": grasp, "retreat_pose": retreat, "model_name": "P_model",
              "observed_pose": {**grasp, "z": 0.25}}
    trace = [{"step_index": 0, "primitive": "compute_pick_targets", "params": {}, "resolved_params": {"part_name": "P"},
              "preview_output": target, "start_snapshot": {}},
             {"step_index": 1, "primitive": "grasp_part", "params": {}, "resolved_params": {"part_name": "P"},
              "start_snapshot": {"current_pose": grasp}},
             {"step_index": 2, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
              "start_snapshot": {"current_pose": grasp}, "projected_snapshot": {"current_pose": retreat}}]
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "pick", "event_name": "pick", "resource_jid": "robot", "part_name": "P"}, trace=trace,
        recovery_des_model=model, recovery_snapshot={"resource_state": "at_pick"},
        projected_snapshot={"resource_state": "at_pick", "current_pose": retreat}, part_context={"part_location": "Exit"},
        grounding_context={}, primitive_catalog=[], resource_evidence={}, physical_evidence={})
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    # The completion adds the part's new location and the pickup location; held_part is grasp_part's own effect.
    assert result["projected_part_context"]["part_location"] == "robot"
    assert result["projected_snapshot"]["resource_location"] == "Exit"




def test_a_robot_may_move_on_after_the_retreat_that_follows_its_transfer(monkeypatch):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    pick_grasp = {"event_name": "pick_grasp", "updates": {"part_location": {"set": "robot"}},
        "composition_contract": {"steps": [{"op": "grasp_part", "params": {}}, {"op": "move_relative", "params": {
            "dx": 0., "dy": 0., "dz": {"$sub": [{"$state": "_task_ctx", "path": ["travel_z"]},
                                                {"$state": "_position", "path": ["z"]}]}}}]}}
    model = {"events": [pick_grasp], "state_variables": {"part_location": {"scope": "part", "domain": ["robot"]}}}
    grasp = {"x": 0.1, "y": 0.2, "z": 0.3, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    retreat, elsewhere = {**grasp, "z": 0.4}, {**grasp, "x": 0.5, "z": 0.4}
    target = {"target_pose": grasp, "retreat_pose": retreat, "model_name": "P_model", "observed_pose": {**grasp, "z": 0.25}}
    trace = [{"step_index": 0, "primitive": "compute_pick_targets", "params": {}, "resolved_params": {"part_name": "P"},
              "preview_output": target, "start_snapshot": {}},
             {"step_index": 1, "primitive": "grasp_part", "params": {}, "resolved_params": {"part_name": "P"},
              "start_snapshot": {"current_pose": grasp}},
             {"step_index": 2, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
              "start_snapshot": {"current_pose": grasp}, "projected_snapshot": {"current_pose": retreat}},
             {"step_index": 3, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
              "start_snapshot": {"current_pose": retreat}, "projected_snapshot": {"current_pose": elsewhere}}]
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "pick", "event_name": "pick", "resource_jid": "robot", "part_name": "P"}, trace=trace,
        recovery_des_model=model, recovery_snapshot={}, projected_snapshot={"current_pose": elsewhere},
        part_context={"part_location": None}, grounding_context={}, primitive_catalog=[], resource_evidence={},
        physical_evidence={})
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")


@pytest.mark.parametrize("last", ["move_cartesian", "move_to_named_pose"])
def test_a_move_without_a_transfer_leaves_the_robot_where_its_pose_is(monkeypatch, last):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    above = {"x": 0.0, "y": 0.1, "z": 1.1, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    trace = [{"step_index": 0, "primitive": last, "params": {}, "resolved_params": {"pose_name": "home"},
              "start_snapshot": {}, "projected_snapshot": {"current_pose": above}}]
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "move", "event_name": "move", "resource_jid": "robot", "part_name": None,
              "expected_end_state": {"resource_location": "Board"}}, trace=trace,
        recovery_des_model={"events": [], "state_variables": {}}, recovery_snapshot={},
        projected_snapshot={"resource_location": None, "current_location": None, "current_pose": above},
        part_context={}, grounding_context={}, primitive_catalog=[], resource_evidence={}, physical_evidence={},
        locate_pose=lambda pose, stated=None: "Board" if pose["z"] > 1.0 else None)
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    # A named posture is the declared move_home outcome, never a place read from the pose.
    expected = "Board" if last == "move_cartesian" else None
    assert result["projected_snapshot"]["resource_location"] == expected
    assert result["projected_snapshot"]["current_location"] == expected

@pytest.mark.parametrize("lifted", [True, False])
def test_one_event_takes_a_part_and_puts_it_down_with_each_transfer_checked(monkeypatch, lifted):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    retreat_step = {"op": "move_relative", "params": {"dx": 0., "dy": 0., "dz": {"$sub": [
        {"$state": "_task_ctx", "path": ["travel_z"]}, {"$state": "_position", "path": ["z"]}]}}}
    pick_grasp = {"event_name": "pick_grasp", "updates": {
        "resource_location": {"set_from_param": "origin_resource_location"}, "part_location": {"set": "robot"}},
        "composition_contract": {"steps": [{"op": "grasp_part", "params": {}}, retreat_step]}}
    place_insert = {"event_name": "place_insert", "updates": {
        "resource_state": {"set": "placed"}, "resource_location": {"set_from_param": "destination_location"}},
        "composition_contract": {"steps": [{"op": "release_part", "params": {}}, retreat_step]}}
    model = {"events": [pick_grasp, place_insert], "state_variables": {
        "part_location": {"scope": "part", "domain": ["Rack", "robot"]},
        "resource_location": {"scope": "resource", "domain": ["Rack", "home"]},
        "resource_state": {"scope": "resource", "domain": ["idle", "placed"]}}}
    down = {"qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    grasp, lift = {"x": 0.0, "y": 0.5, "z": 1.1, **down}, {"x": 0.0, "y": 0.5, "z": 1.3, **down}
    place, away = {"x": 0.2, "y": 0.5, "z": 1.1, **down}, {"x": 0.2, "y": 0.5, "z": 1.3, **down}
    pick = {"target_pose": grasp, "retreat_pose": lift, "model_name": "P_model", "observed_pose": {**grasp, "z": 1.0}}
    put = {"target_pose": place, "retreat_pose": away, "destination_location": "Rack",
           "target_reference": {"surface_role": "support"}, "target_origin_pose": {"x": 0.2, "y": 0.5, "z": 1.0},
           "support_placement": {"part_pose": [0.2, 0.5, 1.0, 0., 0., 0., 1.]}}
    # Each row: the step, the pose it starts from, the pose it ends at, and its calculated output.
    rows = [("compute_pick_targets", {}, {}, pick), ("grasp_part", grasp, grasp, None),
            ("move_cartesian", grasp, lift if lifted else {**grasp, "z": 1.2}, None),
            ("compute_place_targets", lift, lift, put), ("move_cartesian", lift, place, None),
            ("release_part", place, place, None), ("move_cartesian", place, away, None)]
    trace = [{"step_index": index, "primitive": name, "params": {}, "resolved_params": {"part_name": "P"},
              "start_snapshot": {"current_pose": start}, "projected_snapshot": {"current_pose": end},
              **({"preview_output": output} if output else {})}
             for index, (name, start, end, output) in enumerate(rows)]
    slot = {"reference": "part_name", "scope": "resource"}
    physical = {"product_name": "board", "jids": {"Rack": "rack"},
                "environment_models": {"Rack": {"state_variables": {"zone_1_part": slot, "zone_2_part": slot},
                    "assignments": {"zones": [{"zone": 1, "pose": [0.0, 0.5, 1.0]}, {"zone": 2, "pose": [0.2, 0.5, 1.0]}]}}},
                "environment_valuation": {"Rack": {"zone_1_part": "P", "zone_2_part": None}}}
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "move", "event_name": "move", "resource_jid": "robot", "part_name": "P"}, trace=trace,
        recovery_des_model=model, recovery_snapshot={"resource_state": "idle"},
        projected_snapshot={"resource_state": "idle", "current_pose": away}, part_context={"part_location": "Rack"},
        grounding_context={"resources": {"rack": {"zone_1_part": "P", "zone_2_part": None}}}, primitive_catalog=[],
        resource_evidence={}, physical_evidence=physical)
    if not lifted:
        # The motion right after the grasp must reach the grasp's calculated retreat pose.
        assert result["feasibility_status"] == "INFEASIBLE" and "retreat_pose" in result["reason"]
        return
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    # The part leaves its source slot and is received by the slot under the place target, in one record.
    assert result["projected_resources"]["rack"] == {"zone_1_part": None, "zone_2_part": "P"}
    assert result["projected_part_context"]["part_location"] == "Rack"
    assert result["projected_snapshot"]["resource_location"] == "Rack"
    assert result["resource_evidence"]["attachment"] == {}

def test_a_supported_placement_is_received_by_the_slot_whose_custody_locates_the_part():
    from cais_spade_llm.resources.robot.recovery_composition import _receiving_slot

    part = {"reference": "part_name", "scope": "resource"}
    physical = {"product_name": "board", "environment_models": {
        "Rack": {"state_variables": {"zone_1_part": part, "zone_2_part": part},
                 "assignments": {"zones": [{"zone": 1, "pose": [0.0, 0.5, 1.0]}, {"zone": 2, "pose": [0.2, 0.5, 1.0]}]}},
        "M9": {"state_variables": {"part_name": part, "staging_part": part}},
        "arm": {"state_variables": {"held_part": part}}}}
    # A zone of the destination under the target, or another resource's slot for a place it owns.
    assert _receiving_slot("P", "Rack", {"x": 0.2, "y": 0.5}, physical) == ("Rack", "zone_2_part")
    assert _receiving_slot("P", "M9 staging tray", {"x": 3.0, "y": 1.0}, physical) == ("M9", "staging_part")
    assert _receiving_slot("P", "Rack", {"x": 9.0, "y": 9.0}, physical) is None
    assert _receiving_slot("P", "Shelf", {"x": 0.0, "y": 0.0}, physical) is None




def test_a_pick_empties_every_value_that_records_the_part_as_the_ra_finds_them():
    from cais_spade_llm.resources.robot.recovery_composition import _source_records

    slot = {"reference": "part_name", "scope": "resource"}
    physical = {"jids": {"M9": "m9", "Belt": "belt", "Store": "store", "arm": "arm"}, "environment_models": {
        "M9": {"state_variables": {"part_name": slot, "staging_part": slot}},
        "Belt": {"state_variables": {"part_location.{part_name}": {"scope": "resource"},
                                     "part_order.{part_name}": {"scope": "resource"}}},
        "Store": {"state_variables": {"inventory.{part_name}": {"scope": "resource", "domain": [True, False]}}},
        "arm": {"state_variables": {"held_part": slot}}},
        "environment_valuation": {"M9": {"part_name": None, "staging_part": "P"},
                                  "Belt": {"part_location.P": "loading_position_2", "part_order.P": 0},
                                  "Store": {"inventory.P": True}, "arm": {"held_part": None}}}
    records = _source_records("P", "arm", {"resources": {}}, physical)
    # A staging slot, a belt slot with its order (0 still records the part), and a storage flag (false: no null).
    assert sorted(records) == [("belt", "part_location.P", None), ("belt", "part_order.P", None),
                               ("m9", "staging_part", None), ("store", "inventory.P", False)]


def test_a_destination_receives_into_its_own_slot():
    from cais_spade_llm.resources.robot.recovery_composition import _receiving_slot

    part = {"reference": "part_name", "scope": "resource"}
    physical = {"product_name": "board", "environment_models": {"M9": {"state_variables": {
        "part_name": part, "staging_part": part}}}}
    # A machine's workholding slot locates the part at the machine; its staging slot at its staging tray.
    assert _receiving_slot("P", "M9", {"x": 0.0, "y": 0.0}, physical) == ("M9", "part_name")
    assert _receiving_slot("P", "M9 staging tray", {"x": 0.0, "y": 0.0}, physical) == ("M9", "staging_part")


@pytest.mark.parametrize("stated, declared, status", [
    ("loaded", False, "FEASIBLE"), ("paused", True, "INFEASIBLE")])
def test_a_neighbour_state_no_primitive_of_it_declares_is_named_by_the_event(stated, declared, status):
    catalog = [{"name": "pause_job", "effects": {"resource_state": {"set": "paused"}}}] if declared else []
    task = {"outline_id": "load", "event_name": "load", "resource_jid": "owner", "part_name": None,
            "expected_start_state": {}, "expected_end_state": {"resource_state": "idle"},
            "neighbor_end_state": {"m9": {"resource_state": stated}}}
    result = validate_recovery_primitive_composition(
        task=task, primitive_steps=[{"primitive": "move_cartesian", "params": {}}],
        recovery_des_model={"state_variables": {"resource_state": {"scope": "resource"}}},
        recovery_snapshot={"resource_state": "idle"}, part_context={},
        grounding_context={"resources": {"m9": {"resource_state": "idle", "execution_primitive_catalog": catalog}}},
        primitive_catalog=[{"name": "move_cartesian", "params": {}, "required_params": [], "effects": {}}],
        validate_primitive=feasible)
    assert result["feasibility_status"] == status, result["reason"]
    if status == "FEASIBLE":
        assert result["projected_resources"]["m9"]["resource_state"] == "loaded"


def test_a_named_state_never_replaces_the_state_a_step_declares():
    task = {"outline_id": "start", "event_name": "start", "resource_jid": "owner", "part_name": None,
            "expected_start_state": {}, "expected_end_state": {"resource_state": "loaded"}}
    result = validate_recovery_primitive_composition(
        task=task, primitive_steps=[{"primitive": "start_job", "params": {}}],
        recovery_des_model={"state_variables": {"resource_state": {"scope": "resource"}}},
        recovery_snapshot={"resource_state": "idle"}, part_context={}, grounding_context={},
        primitive_catalog=[{"name": "start_job", "params": {}, "required_params": [],
                            "effects": {"resource_state": {"set": "running"}}}],
        validate_primitive=feasible)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert "achieved 'running', required 'loaded'" in result["reason"]


@pytest.mark.parametrize("after, stated, expected", [
    ("move_cartesian", "Rack", None), ("move_cartesian", "M1", "M1"), ("move_to_named_pose", "home", "home")])
def test_where_a_robot_ends_after_its_transfer_and_retreat_decides_its_location(monkeypatch, after, stated, expected):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    retreat_step = {"op": "move_relative", "params": {"dx": 0., "dy": 0., "dz": {"$sub": [
        {"$state": "_task_ctx", "path": ["travel_z"]}, {"$state": "_position", "path": ["z"]}]}}}
    pick_grasp = {"event_name": "pick_grasp", "updates": {
        "resource_location": {"set_from_param": "origin_resource_location"}, "part_location": {"set": "robot"}},
        "composition_contract": {"steps": [{"op": "grasp_part", "params": {}}, retreat_step]}}
    home = {"event_name": "move_home", "updates": {"resource_location": {"set": "home"}, "resource_state": {"set": "idle"}},
            "composition_contract": {"steps": [{"op": "move_to_named_pose", "params": {"pose_name": "home"}}]}}
    model = {"events": [pick_grasp, home], "state_variables": {
        "part_location": {"scope": "part", "domain": ["Rack", "robot"]},
        "resource_location": {"scope": "resource", "domain": ["Rack", "M1", "home"]}}}
    down = {"qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    grasp, lift, away = ({"x": 0.0, "y": 0.5, "z": 1.1, **down}, {"x": 0.0, "y": 0.5, "z": 1.3, **down},
                         {"x": 2.0, "y": -1.0, "z": 1.3, **down})
    pick = {"target_pose": grasp, "retreat_pose": lift, "model_name": "P_model", "observed_pose": {**grasp, "z": 1.0}}
    trace = [{"step_index": 0, "primitive": "compute_pick_targets", "params": {}, "resolved_params": {"part_name": "P"},
              "preview_output": pick, "start_snapshot": {}, "projected_snapshot": {}},
             {"step_index": 1, "primitive": "grasp_part", "params": {}, "resolved_params": {"part_name": "P"},
              "start_snapshot": {"current_pose": grasp}, "projected_snapshot": {"current_pose": grasp}},
             {"step_index": 2, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
              "start_snapshot": {"current_pose": grasp}, "projected_snapshot": {"current_pose": lift}},
             {"step_index": 3, "primitive": after, "params": {}, "resolved_params": {"pose_name": "home"},
              "start_snapshot": {"current_pose": lift}, "projected_snapshot": {"current_pose": away}}]
    seen = []

    def locate(pose, stated=None):
        seen.append(stated)
        # The final pose lies in M1's recorded area only.
        return "M1" if pose["x"] > 1.0 else "Rack"

    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "take", "event_name": "take", "resource_jid": "robot", "part_name": "P",
              "expected_end_state": {"resource_location": stated}}, trace=trace,
        recovery_des_model=model, recovery_snapshot={"resource_location": "Rack"},
        projected_snapshot={"resource_location": "Rack", "current_pose": away}, part_context={"part_location": "Rack"},
        grounding_context={}, primitive_catalog=[], resource_evidence={}, physical_evidence={}, locate_pose=locate)
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    # After its retreat the robot moved on, so its pickup place no longer holds: its final pose decides.
    located = result["projected_snapshot"]["resource_location"]
    assert located == ("M1" if after == "move_cartesian" else expected)
    assert seen == ([stated] if after == "move_cartesian" else [])


def test_a_restated_location_the_robot_keeps_holds_without_locating_its_pose(monkeypatch):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    above = {"x": 0.0, "y": 0.5, "z": 1.4, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    trace = [{"step_index": 0, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
              "start_snapshot": {}, "projected_snapshot": {"current_pose": above}}]
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "hover", "event_name": "hover", "resource_jid": "robot", "part_name": None,
              "expected_end_state": {"resource_location": "Rack"}}, trace=trace,
        recovery_des_model={"events": [], "state_variables": {}}, recovery_snapshot={"resource_location": "Rack"},
        projected_snapshot={"resource_location": "Rack", "current_pose": above}, part_context={},
        grounding_context={}, primitive_catalog=[], resource_evidence={}, physical_evidence={},
        locate_pose=lambda pose, stated=None: None)
    # The RA lets a location the robot already has hold; only a changed one is read from the pose.
    assert result["projected_snapshot"]["resource_location"] == "Rack"

def test_a_place_target_starts_from_the_zone_the_event_puts_its_part_in():
    from cais_spade_llm.resources.robot.recovery_capability import _stated_zone_origin

    scene = {"Rack": {"zones": [{"zone": 1, "pose": [0.0, 0.5, 1.0]}, {"zone": 2, "pose": [0.2, 0.5, 1.0]}]}}
    evidence, physical = {"inputs": {"scene": scene}}, {"jids": {"Rack": "rack"}}
    task = {"part_name": "P", "neighbor_end_state": {"rack": {"zone_2_part": "P"}}}
    assert _stated_zone_origin(task, "Rack", evidence, physical) == {"x": 0.2, "y": 0.5, "z": 1.0}
    # No zone named, or two, leaves the destination's own recorded target.
    assert _stated_zone_origin({**task, "neighbor_end_state": {}}, "Rack", evidence, physical) is None
    both = {"rack": {"zone_1_part": "P", "zone_2_part": "P"}}
    assert _stated_zone_origin({**task, "neighbor_end_state": both}, "Rack", evidence, physical) is None


@pytest.mark.parametrize("received", ["zone_1_part", "zone_4_part"])
def test_the_values_an_event_states_for_other_resources_must_be_what_its_program_leaves(received):
    def complete(*, projected_snapshot, part_context, **_):
        values = {"zone_1_part": None, "zone_4_part": None, received: "P"}
        return {"feasibility_status": "FEASIBLE", "projected_snapshot": projected_snapshot,
                "projected_part_context": part_context, "projected_resources": {"rack": values}}

    task = {"outline_id": "put", "event_name": "put", "resource_jid": "owner", "part_name": "P",
            "expected_start_state": {}, "expected_end_state": {"resource_state": "idle"},
            "neighbor_end_state": {"rack": {"zone_1_part": "P"}}}
    result = validate_recovery_primitive_composition(
        task=task, primitive_steps=[{"primitive": "move_cartesian", "params": {}}],
        recovery_des_model={"state_variables": {"resource_state": {"scope": "resource"}}},
        recovery_snapshot={"resource_state": "idle"}, part_context={},
        grounding_context={"resources": {"rack": {"zone_1_part": None, "zone_4_part": None}}},
        primitive_catalog=[{"name": "move_cartesian", "params": {}, "required_params": [], "effects": {}}],
        validate_primitive=feasible, project_completion=complete)
    if received == "zone_1_part":
        assert result["feasibility_status"] == "FEASIBLE", result["reason"]
    else:
        assert result["feasibility_status"] == "INFEASIBLE" and "rack's zone_1_part" in result["reason"]


@pytest.mark.parametrize("stated, status", [(1.5, "FEASIBLE"), (1.4, "INFEASIBLE")])
def test_a_captured_start_pose_the_outline_still_carries_yields_to_where_an_accepted_program_left_it(stated, status):
    declarations = {"pose": {"scope": "resource", "type": "object", "fields": ["x", "y", "z"], "effect": "current_pose"}}
    task = {"outline_id": "next", "event_name": "next", "resource_jid": "owner", "part_name": None,
            "expected_start_state": {"pose": {"x": 0.0, "y": 0.0, "z": stated}},
            "expected_end_state": {"pose": {"x": 0.0, "y": 0.0, "z": 1.2}}}
    catalog = [{"name": "move_cartesian", "params": {name: {"type": "number"} for name in ("x", "y", "z")},
                "required_params": ["x", "y", "z"], "effects": {"current_pose": {"pose_absolute_from_params": ["x", "y", "z"]}}}]
    result = validate_recovery_primitive_composition(
        task=task, primitive_steps=[{"primitive": "move_cartesian", "params": {"x": 0.0, "y": 0.0, "z": 1.2}}],
        recovery_des_model={"state_variables": declarations},
        recovery_snapshot={"current_pose": {"x": 0.3, "y": 0.0, "z": 1.3}}, part_context={}, grounding_context={},
        primitive_catalog=catalog, validate_primitive=feasible,
        captured_snapshot={"current_pose": {"x": 0.0, "y": 0.0, "z": 1.5}})
    # The captured pose (z 1.5) is what the outline carried; a pose the outline states otherwise must hold.
    assert result["feasibility_status"] == status, result["reason"]

def test_a_place_that_is_not_a_resource_receives_a_part_on_its_own_recorded_model():
    from cais_spade_llm.resources.robot.recovery_capability import _receiving_geometry_source

    physical = {"resources": {"M9": {"geometry_source": {"model": "M9", "links": ["enclosure"]}}},
                "physics_evidence": {"models": {"M9": {}, "M9 staging tray": {}}}}
    assert _receiving_geometry_source("M9", physical) == {"model": "M9", "links": ["enclosure"]}
    assert _receiving_geometry_source("M9 staging tray", physical) == {"model": "M9 staging tray"}
    with pytest.raises(KeyError):
        _receiving_geometry_source("Shelf", physical)


def test_a_declared_support_surface_is_the_only_face_a_placed_part_rests_on():
    from cais_spade_llm.resources.robot.recovery_capability import _declared_support_collisions

    regions = {"M9.staging": {"source": {"surface_collision": "M9 staging tray::link::surface"}},
               "M9.door": {"source": {"kind": "declared safety region"}},
               "Rack.zone_1": {"source": {"surface_collision": "Rack::link::belt"}}}
    physical = {"geometry": {"regions": regions}}
    assert _declared_support_collisions("M9 staging tray", physical) == {"M9 staging tray::link::surface"}
    # A body that declares no support surface keeps every recorded face as a candidate.
    assert _declared_support_collisions("Shelf", physical) == set()


def _resume_contract(program):
    return {"name": "resume_job", "params": {"job_id": {"type": "string"}}, "required_params": ["job_id"],
            "preconditions": {"resource_state": {"equals": "paused"}, "part_name": {"equals_from_param": "part_name"},
                              "selected_program": {"equals": program}},
            "product_guards": {"current_process": {"matches": {"result": program, "machine": "M9"}}},
            "effects": {"resource_state": {"set": "running"}},
            "product_effects": {"current_process": {"set_status": "running"}}}


@pytest.mark.parametrize("selected", ["square", "circle"])
def test_a_step_uses_the_first_declared_alternative_that_applies_with_its_part_effects(selected):
    declarations = {"resource_state": {"scope": "resource"}, "selected_program": {"scope": "resource"},
                    "part_name": {"scope": "resource"},
                    "current_process": {"scope": "part", "domain": [], "domain_complete": False}}
    job = {"process": "trim", "result": "square", "status": "paused", "machine": "M9"}
    task = {"outline_id": "resume", "event_name": "resume", "resource_jid": "owner", "part_name": "P",
            "expected_start_state": {"resource_state": "paused"},
            "expected_end_state": {"resource_state": "running", "current_process": {**job, "status": "running"}}}
    result = validate_recovery_primitive_composition(
        task=task, primitive_steps=[{"primitive": "resume_job", "params": {"job_id": "job-1"}}],
        recovery_des_model={"state_variables": declarations},
        recovery_snapshot={"resource_state": "paused", "part_name": "P", "selected_program": selected},
        part_context={"current_process": job}, grounding_context={},
        primitive_catalog=[_resume_contract("circle"), _resume_contract("square")], validate_primitive=feasible)
    # The contract's part_name is the event's part; the square alternative's part guard holds for the part's job.
    if selected == "square":
        assert result["feasibility_status"] == "FEASIBLE", result["reason"]
        assert result["projected_part_context"]["current_process"]["status"] == "running"
    else:
        assert result["feasibility_status"] == "INFEASIBLE"
        assert "no declared alternative applies" in result["reason"]


_JOB_DECLARATIONS = {"resource_state": {"scope": "resource"}, "part_name": {"scope": "resource"},
                     "current_process": {"scope": "part", "domain": [], "domain_complete": False},
                     "processCompleted": {"scope": "part"}}


def _never_checked(**_):
    raise AssertionError("An awaited outcome sends no command, so no command capability is checked")


@pytest.mark.parametrize("state", ["running", "paused"])
def test_an_uncontrollable_outcome_is_awaited_with_its_declared_effects(state):
    complete = {"name": "complete_job", "params": {"result": {}}, "required_params": ["result"], "controllable": False,
                "after_sec": {"$evidence": "job.processing_time_sec"},
                "preconditions": {"resource_state": {"equals": "running"}, "part_name": {"equals_from_param": "part_name"}},
                "product_guards": {"current_process": {"matches": {"machine": "M9"}}},
                "effects": {"resource_state": {"set": "completed"}},
                "product_effects": {"current_process": {"set": None},
                                    "processCompleted": [{"process": "trim", "result": {"set_from_param": "result"}}]}}
    job = {"process": "trim", "result": "square", "status": "running", "machine": "M9"}
    task = {"outline_id": "complete", "event_name": "complete", "resource_jid": "owner", "part_name": "P",
            "expected_start_state": {"resource_state": state},
            "expected_end_state": {"resource_state": "completed", "current_process": None,
                                   "processCompleted": [{"process": "trim", "result": "square"}]}}
    result = validate_recovery_primitive_composition(
        task=task, primitive_steps=[{"primitive": "complete_job",
                                     "params": {"result": {"context_ref": "/parts/P/current_process/result"}}}],
        recovery_des_model={"state_variables": _JOB_DECLARATIONS},
        recovery_snapshot={"resource_state": state, "part_name": "P"},
        part_context={"current_process": job, "processCompleted": []}, grounding_context={},
        primitive_catalog=[complete], validate_primitive=_never_checked)
    if state == "paused":
        # It happens only while the job runs, as its declared precondition states.
        assert result["feasibility_status"] == "INFEASIBLE" and "preconditions are not satisfied" in result["reason"]
        return
    assert result["feasibility_status"] == "FEASIBLE", result["reason"]
    assert result["trace"][0]["controllable"] is False and "capability_evidence" not in result["trace"][0]
    assert result["trace"][0]["after_sec"] == {"$evidence": "job.processing_time_sec"}
    assert result["derived_end_state"] == task["expected_end_state"]


def test_a_command_starts_the_job_record_its_declared_template_names():
    start = {"name": "start_job", "params": {name: {"type": "string"} for name in ("job_id", "process", "result")},
             "required_params": ["job_id", "process", "result"], "preconditions": {"resource_state": {"equals": "loaded"}},
             "product_guards": {"current_process": {"not_matches": {"machine": "M9"}}},
             "effects": {"resource_state": {"set": "running"}},
             "product_effects": {"current_process": {"set": {"process": {"set_from_param": "process"},
                                                             "result": {"set_from_param": "result"},
                                                             "status": "running", "machine": "M9"}}}}
    job = {"process": "trim", "result": "square", "status": "running", "machine": "M9"}
    task = {"outline_id": "start", "event_name": "start", "resource_jid": "owner", "part_name": "P",
            "expected_start_state": {"resource_state": "loaded"},
            "expected_end_state": {"resource_state": "running", "current_process": job}}
    result = validate_recovery_primitive_composition(
        task=task, primitive_steps=[{"primitive": "start_job",
                                     "params": {"job_id": "job-2", "process": "trim", "result": "square"}}],
        recovery_des_model={"state_variables": _JOB_DECLARATIONS},
        recovery_snapshot={"resource_state": "loaded", "part_name": "P"},
        part_context={"current_process": None, "processCompleted": []}, grounding_context={},
        primitive_catalog=[start], validate_primitive=feasible)
    assert result["feasibility_status"] == "FEASIBLE", result["reason"]
    assert result["projected_part_context"]["current_process"] == job



def _machine_job_case(steps, snapshot, part_context, end_state):
    import json

    from cais_spade_llm.recovery_framework import SCENE_PATH
    from cais_spade_llm.resources.machine.machine_jobs import MachineJobs, machine_primitive_contracts, project_job_command
    from cais_spade_llm.resources.recovery_composition_owner import validate_owner_composition

    machine = json.loads(SCENE_PATH.read_text())["machines"][1]
    part = "RGOCG4-50_Round_4mm"
    observed = {"part_name": part, "correct_part_present": True, "robot_access_clear": True}
    jobs = MachineJobs(machine, "run")
    jobs.command("start_job", {"job_id": "circle-job", "task_id": "circle-job", "part_name": part, "process": "trim",
                               "result": "circle"}, command_id="circle-job", now=100, observations=observed)
    contracts = machine_primitive_contracts(machine)
    schemas = {"pause_job": {"job_id": {"type": "string"}}, "select_program": {"program": {"type": "string"}},
               "complete_job": {"result": {}}}
    catalog = [{"name": name, "params": schema, "required_params": list(schema), **contracts[name][0]}
               for name, schema in schemas.items()]
    seen = []

    def check(*, primitive, params, resource_evidence, **_):
        record = resource_evidence["processing_observations"]
        seen.append(record["jobs"]["circle-job"]["job_state"])
        record, _ = project_job_command(machine, record, primitive, params, observed, command_id=primitive)
        return {"allowed": True, "feasibility_status": "FEASIBLE",
                "evidence": {"projected_processing_observations": record}}

    declarations = {**_JOB_DECLARATIONS, "selected_program": {"scope": "resource"}}
    evidence = {"resource_id": machine["resource_id"], "inputs": {"scene": {"machines": [machine]}},
                "processing_observations": jobs.snapshot()}
    prepared = {"recovery_resources": {"owner": {"resource_type": "resource", "execution_primitive_catalog": catalog,
                                                 "resource_evidence": evidence,
                                                 "recovery_des_model": {"state_variables": declarations}}}}
    task = {"outline_id": "job", "event_name": "job", "resource_jid": "owner", "part_name": part,
            "expected_start_state": {"resource_state": snapshot["resource_state"]}, "expected_end_state": end_state}
    result = validate_owner_composition(SimpleNamespace(jid="owner", check_recovery_primitive_feasibility=check),
        task=task, primitive_steps=steps, prepared=prepared, recovery_snapshot=snapshot,
        part_context=part_context, grounding_context={})
    return machine, result, seen


def test_machine_commands_leave_the_job_record_their_later_steps_and_events_read():
    machine, result, seen = _machine_job_case(
        [{"primitive": "pause_job", "params": {"job_id": "circle-job"}},
         {"primitive": "select_program", "params": {"program": "square"}}],
        {"resource_state": "running", "part_name": "RGOCG4-50_Round_4mm", "selected_program": "circle"},
        {"current_process": {"process": "trim", "result": "circle", "status": "running", "machine": "M2"},
         "processCompleted": []},
        {"resource_state": "paused", "selected_program": "square"})
    assert result["feasibility_status"] == "FEASIBLE", result["reason"]
    # The program selection is checked against the record the pause left, not the failure-time one.
    assert seen == ["running", "paused"]
    record = result["resource_evidence"]["processing_observations"]
    assert record["selected_program"] == "square" and record["jobs"]["circle-job"]["job_state"] == "paused"


def test_an_awaited_completion_finishes_the_active_job_in_the_record_later_events_read():
    machine, result, seen = _machine_job_case(
        [{"primitive": "complete_job", "params": {"result": "circle"}}],
        {"resource_state": "running", "part_name": "RGOCG4-50_Round_4mm", "selected_program": "circle"},
        {"current_process": {"process": "trim", "result": "circle", "status": "running", "machine": "M2"},
         "processCompleted": []},
        {"resource_state": "completed", "current_process": None,
         "processCompleted": [{"process": "trim", "result": "circle"}]})
    assert result["feasibility_status"] == "FEASIBLE", result["reason"]
    assert seen == []
    record = result["resource_evidence"]["processing_observations"]
    job = record["jobs"]["circle-job"]
    assert record["active_job"] is None and job["job_state"] == "completed"
    assert job["processing_active_sec"] == machine["simulation_process"]["processing_time_sec"]


@pytest.mark.parametrize("named, status", [("standby", "FEASIBLE"), ("idle", "INFEASIBLE")])
def test_a_resource_state_no_primitive_declares_is_named_by_the_event(named, status):
    catalog = [{"name": "move_cartesian", "params": {}, "required_params": [], "effects": {}},
               {"name": "move_to_named_pose", "params": {}, "required_params": [],
                "effects": {"current_state": {"set": "idle"}}}]
    task = {"outline_id": "label", "event_name": "label", "resource_jid": "owner", "part_name": None,
            "expected_start_state": {"resource_state": "failed"}, "expected_end_state": {"resource_state": named}}
    result = validate_recovery_primitive_composition(
        task=task, primitive_steps=[{"primitive": "move_cartesian", "params": {}}],
        recovery_des_model={"state_variables": {"resource_state": {"scope": "resource"}}},
        recovery_snapshot={"resource_state": "failed"}, part_context={}, grounding_context={},
        primitive_catalog=catalog, validate_primitive=feasible)
    assert result["feasibility_status"] == status, result["reason"]
    if status == "FEASIBLE":
        # Later events start from the named state, as the outline's projection does.
        assert result["projected_snapshot"]["resource_state"] == "standby"
    else:
        # move_to_named_pose declares idle, so idle comes only from that primitive in the program.
        assert "achieved 'failed', required 'idle'" in result["reason"]

def test_a_derived_pose_condition_is_read_through_the_effect_field_it_follows():
    declarations = {"pose": {"scope": "resource", "type": "object", "fields": ["x", "y", "z"], "effect": "current_pose"}}
    move = {"name": "move_cartesian", "params": {axis: {"type": "number"} for axis in ("x", "y", "z")},
            "required_params": ["x", "y", "z"], "preconditions": {},
            "effects": {"current_pose": {"pose_absolute_from_params": ["x", "y", "z"]}}}
    start = {"x": 0.0, "y": 0.0, "z": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

    def compose(target):
        task = {"outline_id": "move", "event_name": "move", "resource_jid": "owner", "part_name": None,
                "expected_start_state": {"pose": dict(start)}, "expected_end_state": {"pose": {"x": 0.5, "y": -0.5, "z": 1.04}}}
        return task, validate_recovery_primitive_composition(
            task=task, primitive_steps=[{"primitive": "move_cartesian", "params": target}],
            recovery_des_model={"state_variables": declarations}, recovery_snapshot={"current_pose": dict(start)},
            part_context={}, grounding_context={}, primitive_catalog=[move], validate_primitive=feasible)

    # The pose is the current_pose its primitives write, compared on the fields the condition declares.
    task, result = compose({"x": 0.5, "y": -0.5, "z": 1.04})
    assert result["feasibility_status"] == "FEASIBLE", result["reason"]
    assert result["derived_end_state"] == task["expected_end_state"]
    _, elsewhere = compose({"x": 0.4, "y": -0.5, "z": 1.04})
    assert elsewhere["feasibility_status"] == "INFEASIBLE" and "Postcondition 'pose'" in elsewhere["reason"]


def test_other_resources_consume_validated_product_effects(contract):
    prepared = _prepared(contract)
    prepared["recovery_resources"]["other"] = {"recovery_snapshot": {"phase": "waiting"}}
    session = {"accepted_primitive_program": [_row(contract)]}
    resource, part = composition_start_views(session=session, prepared=prepared,
        task={"resource_jid": "other", "part_name": "P"})
    assert resource == {"phase": "waiting"}
    assert part["quality"] == "checked"
    session["accepted_primitive_program"][0]["composition_validation"]["feasibility_status"] = "INFEASIBLE"
    _, part = composition_start_views(session=session, prepared=prepared, task={"resource_jid": "other", "part_name": "P"})
    assert part["quality"] == "pending"


def test_generation_routes_one_event_at_a_time_and_preserves_source(contract, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_composition import generate_next_composition_event

    prepared = _prepared(contract)
    second = deepcopy(contract)
    second["task"].update(outline_id="next", resource_jid="other",
        expected_start_state={"mode": "fault", "quality": "checked"})
    second["part_context"]["quality"] = "checked"
    prepared["recovery_resources"]["other"] = deepcopy(prepared["recovery_resources"]["owner"])
    events = [contract["task"], second["task"]]
    session = {"accepted_outline_prefix": deepcopy(events)}
    calls = []
    async def generate(**kwargs):
        calls.append(kwargs["resource_jid"])
        assert len(kwargs["assigned_outline_events"]) == 1
        if len(calls) == 2:
            assert kwargs["session_state"]["accepted_primitive_program"][0]["composition_validation"]["derived_end_state"]["quality"] == "checked"
        return {"decision": "draft_ready", "primitive_events": [_row(contract if len(calls) == 1 else second)], "turns": []}
    monkeypatch.setattr(multi_turn, "_run_resource_primitive_batch", generate)
    async def run():
        assert (await generate_next_composition_event(planner=SimpleNamespace(), session_state=session,
            prepared_recovery_request=prepared))[0] == "primitive_steps_ready"
        assert (await generate_next_composition_event(planner=SimpleNamespace(), session_state=session,
            prepared_recovery_request=prepared))[0] == "draft_ready"
    asyncio.run(run())
    assert calls == ["owner", "other"]
    assert session["accepted_outline_prefix"] == events


@pytest.mark.parametrize("scope", ["product", "unknown", None])
def test_unknown_scopes_require_context(contract, scope):
    contract["recovery_des_model"]["state_variables"]["quality"]["scope"] = scope
    assert validate_recovery_primitive_composition(**contract)["feasibility_status"] == "NEEDS_CONTEXT"


def test_evidence_content_tampering_is_rejected(contract):
    row = _row(contract)
    row["composition_validation"]["projected_snapshot"]["mode"] = "fault"
    assert "fingerprint" in composition_program_error([contract["task"]], [row], prepared=_prepared(contract), session={})


def test_peer_resource_and_multiple_part_effects_reach_next_owner(contract):
    from cais_spade_llm.resources.recovery_composition import composition_grounding_context

    prepared = _prepared(contract)
    prepared["recovery_resources"]["receiver"] = {"recovery_snapshot": {"zone_A_part": None}}
    row = _row(contract)
    row["composition_validation"].update(projected_resources={"receiver": {"zone_A_part": "P"}},
        projected_parts={"Q": {"location": "receiver", "quality": "checked"}})
    session = {"accepted_primitive_program": [row]}
    context = composition_grounding_context(session=session, prepared=prepared,
        task={"resource_jid": "receiver", "part_name": "Q"})
    assert context["resource"]["zone_A_part"] == "P"
    assert context["part"] == {"location": "receiver", "quality": "checked"}


def test_rejected_event_does_not_advance_accepted_program(contract, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_composition import generate_next_composition_event

    prepared = _prepared(contract)
    session = {"accepted_outline_prefix": [deepcopy(contract["task"])], "accepted_primitive_program": []}
    row = _row(contract)
    row["composition_validation"]["feasibility_status"] = "INFEASIBLE"
    async def reject(**kwargs):
        return {"primitive_events": [row], "turns": [], "feedback": [{"reason": "wrong order"}]}
    monkeypatch.setattr(multi_turn, "_run_resource_primitive_batch", reject)
    decision, _, _ = asyncio.run(generate_next_composition_event(planner=SimpleNamespace(), session_state=session,
        prepared_recovery_request=prepared))
    assert decision == "primitive_blocked"
    assert session["accepted_primitive_program"] == []
    assert session["primitive_rejection_feedback"][0]["reason"] == "wrong order"


def test_composition_carries_history_references_without_copying_full_transcripts(contract, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    class ArchivedTranscript:
        def __deepcopy__(self, memo):
            raise AssertionError("Full archived transcripts must not be copied into the next resource batch")

    async def generate(**kwargs):
        carried = kwargs["carried_session_state"]
        assert carried["turns"][0]["response_artifact_path"] == "recorded_response.txt"
        assert "prompt_input" not in carried["turns"][0]
        assert carried["accepted_primitive_program"][0]["outline_id"] == "restore"
        return {"decision": "draft_ready", "primitive_events": []}

    monkeypatch.setattr(multi_turn, "_resource_agent_map", lambda planner: {
        "owner": SimpleNamespace(generate_recovery_primitives_batch=generate)})
    session = {"accepted_primitive_program": [_row(contract)], "turns": [{
        "phase": "primitive_generation", "response_artifact_path": "recorded_response.txt",
        "prompt_input": ArchivedTranscript()}]}
    asyncio.run(multi_turn._run_resource_primitive_batch(planner=SimpleNamespace(),
        prepared_recovery_request=_prepared(contract), session_state=session,
        resource_jid="owner", assigned_outline_events=[contract["task"]]))


def test_composition_parent_retains_substream_artifacts_without_nesting_batch_session(contract, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_composition import generate_next_composition_event

    full_turn = {"phase": "primitive_generation", "outline_id": "restore",
        "prompt_artifact_path": "recorded_prompt.txt", "response_artifact_path": "recorded_response.txt",
        "prompt_input": {"earlier_session": {"turns": ["large retained transcript"]}}}
    result = {"decision": "draft_ready", "primitive_events": [_row(contract)],
              "turns": [full_turn], "session_state": {"turns": [full_turn]}}

    async def generate(**kwargs):
        return result

    monkeypatch.setattr(multi_turn, "_run_resource_primitive_batch", generate)
    session = {"accepted_outline_prefix": [contract["task"]]}
    decision, turn, response = asyncio.run(generate_next_composition_event(planner=SimpleNamespace(),
        session_state=session, prepared_recovery_request=_prepared(contract)))
    assert decision == "draft_ready"
    child = turn["primitive_substream_turns"][0]
    assert child["prompt_artifact_path"] == "recorded_prompt.txt"
    assert child["response_artifact_path"] == "recorded_response.txt"
    assert "prompt_input" not in child
    assert "session_state" not in response["resource_batches"][0]
    assert "turns" not in response["resource_batches"][0]
    assert result["session_state"]["turns"][0] == full_turn


def test_composition_stops_after_three_same_cause_rejections(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_primitive_generation as generation, multi_turn_prompts as prompts,
    )

    calls = []

    async def ask(prompt, **kwargs):
        calls.append(prompt)
        return {"decision": "primitive_steps_ready", "revision": len(calls)}

    async def recorded(fn, **kwargs):
        return await fn(kwargs["prompt"])

    async def reject(**kwargs):
        feedback = [{"outline_id": "restore", "resource_jid": "owner",
            "constraint_code": "composition_infeasible", "reason": "Retreat is missing",
            "evidence_fingerprint": str(len(calls))}]
        kwargs["session_state"]["primitive_rejection_feedback"] = feedback
        return "need_primitive_revision", {"primitive_rejection_feedback": feedback}

    monkeypatch.setattr(prompts, "build_multi_turn_phase_prompt_input", lambda **kwargs: {})
    monkeypatch.setattr(prompts, "render_multi_turn_phase_prompt", lambda value: "author")
    monkeypatch.setattr(generation, "build_primitive_generation_prompt_context", lambda **kwargs: {})
    monkeypatch.setattr(generation, "recorded_structured_call", recorded)
    monkeypatch.setattr(generation, "_handle_primitive_generation_phase", reject)
    monkeypatch.setattr(generation, "_write_primitive_subturn_artifact", lambda **kwargs: {})
    result = asyncio.run(generation.generate_primitive_batch_with_llm_agent(
        llm_agent=SimpleNamespace(ask_llm_structured=ask),
        prepared_recovery_request={"composition_validation_version": 1},
        assigned_outline_events=[{"outline_id": "restore", "resource_jid": "owner"}]))

    assert len(calls) == 3
    assert result["decision"] == "primitive_blocked"
    assert result["primitive_events"] == []
    assert result["feedback"][0]["reason"] == "Retreat is missing"
    assert result["turns"][-1]["composition_same_cause_failures"] == 3


def test_move_home_completion_cannot_override_a_later_motion():
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    model = {"events": [{"event_name": "move_home", "composition_contract": {
        "steps": [{"op": "move_to_named_pose", "params": {"pose_name": "home"}}]},
        "updates": {"resource_location": {"set": "home"}}}]}
    trace = [{"primitive": "move_to_named_pose", "params": {"pose_name": "home"},
              "resolved_params": {"pose_name": "home"}, "step_index": 0},
             {"primitive": "move_cartesian", "params": {"x": 2., "y": 0., "z": 1.},
              "resolved_params": {"x": 2., "y": 0., "z": 1.}, "step_index": 1}]
    result = project_composition_completion(task={"resource_jid": "robot"}, trace=trace,
        recovery_des_model=model, recovery_snapshot={}, projected_snapshot={"resource_location": "elsewhere"},
        part_context={}, grounding_context={}, primitive_catalog=[], resource_evidence={}, physical_evidence={})
    assert result["projected_snapshot"]["resource_location"] == "elsewhere"
    assert result["completed_contracts"] == []


@pytest.mark.parametrize("changed", [False, True])
def test_place_input_binding_uses_recorded_context_before_owner_output_enrichment(monkeypatch, changed):
    from cais_spade_llm.resources.robot import recovery_capability
    from cais_spade_llm.resources.robot.recovery_composition import preview_composition_output

    recorded = {"state": {"_task_ctx": {"part_name": "P", "pick_tcp_z": 1.2, "travel_z": 1.5}}}
    calculated = deepcopy(recorded)
    calculated["state"]["_task_ctx"].update(surface_role="support", insert_pose={"z": 1.0})
    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **kwargs: {
        "status": "calculated", "destination_location": "buffer", "target_pose": {"x": 0., "y": 0., "z": 1.0}})
    params = {"part_name": "P", "destination_location": "buffer",
              "pick_ctx": deepcopy(recorded["state"]["_task_ctx"])}
    if changed:
        params["pick_ctx"]["pick_tcp_z"] = 2.0
    result = preview_composition_output(primitive="compute_place_targets", params=params,
        recovery_snapshot={}, part_context={}, task={"part_name": "P", "resource_jid": "robot"},
        grounding_context={"resources": {"robot": {"resource_evidence": recorded}}},
        resource_evidence=calculated, physical_evidence={})
    assert result["status"] == ("NEEDS_CONTEXT" if changed else "calculated")
    assert "surface_role" not in recorded["state"]["_task_ctx"]


def test_pick_override_feedback_identifies_all_unsupported_inputs_without_accepting_them(monkeypatch):
    from cais_spade_llm.resources.robot import recovery_capability
    from cais_spade_llm.resources.robot.recovery_composition import preview_composition_output

    monkeypatch.setattr(recovery_capability, "calculate_recovery_pick_target", lambda **kwargs: {
        "status": "calculated", "target_pose": {"x": 0., "y": 0., "z": 1.}, "travel_z": 1.5})
    values = dict(primitive="compute_pick_targets", recovery_snapshot={}, part_context={},
        grounding_context={}, task={"part_name": "P", "resource_jid": "robot"},
        resource_evidence={}, physical_evidence={})
    result = preview_composition_output(params={"part_name": "P", "product_geometry": {},
        "target_pose": {"x": 0., "y": 0., "z": 1.}, "target_pose_source": "observed_pose"}, **values)
    assert result["status"] == "NEEDS_CONTEXT"
    assert "product_geometry, target_pose, target_pose_source" in result["reason"]
    assert "Omit these parameters" in result["reason"]
    assert preview_composition_output(params={"part_name": "P"}, **values)["status"] == "calculated"
