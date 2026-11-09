from __future__ import annotations

"""Contract-level regression coverage for generalized ordered composition."""

import asyncio
from copy import deepcopy
import functools
import math
from types import SimpleNamespace

import pytest

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
from cais_spade_llm.recovery_framework.geometry import compose
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


def test_a_part_s_current_process_is_its_own_whichever_resource_acts(contract):
    """Study 2, 2026-11-08: Sol's Part slippage outlines state gear_small's current_process in ur5e-3's insertion.

    The outline stage accepted it (it already held); composition refused the event because the robot's model does
    not list the field. The part's unfinished job is the part's own, as its process record is: it is compared, never
    supplied.
    """
    contract["part_context"]["current_process"] = None
    contract["task"]["expected_start_state"]["current_process"] = None
    contract["task"]["expected_end_state"]["current_process"] = None
    result = validate_recovery_primitive_composition(**contract)
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    assert result["derived_end_state"]["current_process"] is None

    contract["task"]["expected_start_state"]["current_process"] = {"machine": "M2", "process": "trim"}
    assert validate_recovery_primitive_composition(**contract)["feasibility_status"] != "FEASIBLE"


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
        contract["primitive_steps"][1]["params"]["other"] = {"context_ref": "/limit"}
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


def test_one_finding_lists_every_number_a_program_writes_before_any_step_resolves(contract):
    from cais_spade_llm.resources.recovery_composition import typed_number_refusal

    # Every number in params is a reference (D1); strings, booleans and references stay as they are.
    contract["primitive_steps"] = [
        {"primitive": "reset", "params": {"speed": 0.45, "label": "fast", "check": True}},
        {"primitive": "qualify", "params": {"count": 3, "pose": [1.0, {"context_ref": "/limit"}],
                                            "where": {"z": -2.6}, "unresolved": {"context_ref": "/missing"}}}]
    typed = [{"step_index": 0, "primitive": "reset", "path": "params.speed", "value": 0.45},
             {"step_index": 1, "primitive": "qualify", "path": "params.count", "value": 3},
             {"step_index": 1, "primitive": "qualify", "path": "params.pose[0]", "value": 1.0},
             {"step_index": 1, "primitive": "qualify", "path": "params.where.z", "value": -2.6}]
    assert typed_number_refusal(contract["primitive_steps"])["typed_numbers"] == typed
    result = validate_recovery_primitive_composition(**contract)
    assert result["feasibility_status"] == "INFEASIBLE" and result["trace"] == []
    assert result["reason"] == (
        'The program writes 4 numbers in params, where every number is a {"context_ref": "/path"} into the served '
        "context: step 0 reset params.speed 0.45; step 1 qualify params.count 3; step 1 qualify params.pose[0] 1.0; "
        "step 1 qualify params.where.z -2.6")
    assert result["findings"] == [{"constraint_owner": "resource", "step_index": None,
                                   "constraint_code": "composition_infeasible", "reason": result["reason"],
                                   "typed_numbers": typed}]
    # A program that writes no number goes on to its steps, where the unresolved reference is found.
    contract["primitive_steps"] = [{"primitive": "reset", "params": {}},
                                   {"primitive": "qualify", "params": {"count": {"context_ref": "/missing"}}}]
    assert typed_number_refusal(contract["primitive_steps"]) is None
    assert typed_number_refusal([{"primitive": "reset", "params": {"label": "fast", "check": False}}]) is None
    result = validate_recovery_primitive_composition(**contract)
    assert result["feasibility_status"] == "NEEDS_CONTEXT" and result["findings"][0]["step_index"] == 1


def test_composition_s_output_contract_states_that_every_number_is_a_reference():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _render_primitive_generation_prompt,
    )

    event = {"outline_id": "e", "resource_jid": "r", "event_name": "go"}
    payload = {"primitive_active_outline_event_token": event, "primitive_active_outline_event": event}
    composed = _render_primitive_generation_prompt({**payload, "session_state": {"composition_validation_version": 1}})
    contract = composed.split("Output Contract\n")[1]
    assert ('- Each step MUST include params. Every number in params is a {"context_ref": ...}; strings and booleans '
            "may be literals from served context.") in contract.splitlines()
    assert sum(line.startswith("- ") for line in contract.splitlines()) == 8
    # The older path, which has no typed-number check, keeps its own line.
    assert "or a literal from served context" in _render_primitive_generation_prompt(payload)


def test_a_base_move_keeps_the_shape_of_the_recorded_pose(contract):
    # The KMR's move_base declares [x, y, yaw]; its recorded base_pose is position and quaternion.
    contract["primitive_catalog"].append({
        "name": "travel", "params": {"target_pose": {"type": "array", "items": {"type": "number"}}},
        "required_params": ["target_pose"], "preconditions": {},
        "effects": {"base_pose": {"pose_absolute_from_target_pose": ["x", "y", "yaw"]}}})
    contract["primitive_steps"].append({"primitive": "travel", "params": {"target_pose": {"context_ref": "/drive"}}})
    contract["grounding_context"]["drive"] = [1.0, 2.0, 0.0]
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
    steps = [*contract["primitive_steps"], {"primitive": "travel", "params": {"target_pose": {"context_ref": "/drive"}}}]
    result = validate_owner_composition(owner, task=contract["task"], primitive_steps=steps,
                                        prepared={"recovery_resources": {"owner": entry}},
                                        recovery_snapshot=contract["recovery_snapshot"],
                                        part_context=contract["part_context"],
                                        grounding_context={**contract["grounding_context"], "drive": [1.0, 2.0, 0.0]})
    assert result["feasibility_status"] == "FEASIBLE"
    assert result["trace"][2]["projected_snapshot"]["base_pose"] == [1.0, 2.0, 0.1, 0.0, 0.0, 0.0, 1.0]
    # The prefix's own fact wins over the recorded one.
    assert result["trace"][0]["start_snapshot"]["mode"] == "fault"
    # The evidence binds the prefix, which is what the final program check restores, not the facts added to it.
    assert result["start_fingerprint"] == recovery_validation_fingerprint(
        [contract["recovery_snapshot"], contract["part_context"]])


def test_a_stated_base_pose_is_read_back_from_the_recorded_pose(contract):
    # A KMR's pose is its base on x, y and yaw, recorded as position and quaternion; a turn past pi reads back negative.
    contract["recovery_des_model"]["state_variables"]["pose"] = {
        "scope": "resource", "fields": ["x", "y", "yaw"], "effect": "base_pose", "locates_resource": True}
    contract["primitive_catalog"].append({
        "name": "travel", "params": {"target_pose": {"type": "array", "items": {"type": "number"}}},
        "required_params": ["target_pose"], "preconditions": {},
        "effects": {"base_pose": {"pose_absolute_from_target_pose": ["x", "y", "yaw"]}}})
    contract["primitive_steps"].append({"primitive": "travel", "params": {"target_pose": {"context_ref": "/drive"}}})
    contract["grounding_context"]["drive"] = [-5.47, 1.1, 3.1416]
    contract["recovery_snapshot"]["base_pose"] = [-5.54, 2.13, 0.0, 0.0, 0.0, 0.7071, 0.7071]
    contract["task"]["expected_end_state"]["pose"] = {"x": -5.47, "y": 1.1, "yaw": 3.1416}
    assert validate_recovery_primitive_composition(**contract)["feasibility_status"] == "FEASIBLE"
    contract["task"]["expected_end_state"]["pose"]["yaw"] = 1.5708
    refused = validate_recovery_primitive_composition(**contract)
    assert refused["feasibility_status"] == "INFEASIBLE" and "Postcondition 'pose'" in refused["reason"]


_KMR_TOOL_DOWN = [1.0, 0.0, 0.0, 0.0]


def _kmr_scene(transport: list[float]) -> dict:
    """The KMR settings its composition reads, with the tool position of its transport pose in the base frame."""
    return {"KMR": {"task_execution": {"position_tolerance_m": 0.006, "orientation_tolerance_rad": 0.02,
                                       "cartesian_waypoints": {"transport_tcp_position_in_base_frame_m": transport}}}}


def _kmr_step(evidence, targets, primitive, *, facts=None, base=None, **params):
    from cais_spade_llm.recovery_framework.kmr_composition import advance_kmr_program

    return advance_kmr_program(evidence, targets, task={"part_name": "P"}, primitive=primitive, params=params,
                               recovery_snapshot={"base_pose": base}, checked={"evidence": facts or {}})


def test_a_kmr_program_starts_each_step_where_the_last_one_left_it():
    down = [1.0, 0.0, 0.0, 0.0]
    evidence = {"inputs": {"scene": _kmr_scene([1.0, 0.0, 1.0])}, "snapshot": {"current_pose": [1.0, 0.0, 1.0, *down]},
                "observed_parts": {"P": {"current_pose": [3.0, 1.0, 0.9, 0.0, 0.0, 0.0, 1.0]}}}
    targets, here = {}, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # The arm rides on the base: a tool 1 m ahead of a base that drives to (2, 0) and turns left is 1 m to its left.
    assert _kmr_step(evidence, targets, "move_base", target_pose=[2.0, 0.0, math.pi / 2], base=here) is None
    assert evidence["snapshot"]["base_pose"][:2] == [2.0, 0.0]
    assert evidence["snapshot"]["current_pose"][:3] == pytest.approx([2.0, 1.0, 1.0])
    # An arm motion leaves the arm at the joints MoveIt's path ended on, and the TCP there.
    grasp = [3.0, 1.0, 0.925, *down]
    assert _kmr_step(evidence, targets, "move_cartesian", target=grasp, facts=_arm_moved(grasp, joints=[0.5] * 7)) is None
    assert evidence["snapshot"]["joint_positions"] == [0.5] * 7 and evidence["snapshot"]["current_pose"] == grasp
    # A grasp attaches the part by the transform its check found at the actual TCP; no calculated target is needed.
    attachment = {"attached": True, "part_name": "P", "held_part": "P", "resting_on_support": True,
                  "grasp_transform": [0.0, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]}
    assert _kmr_step(evidence, targets, "grasp_part", part_name="P", facts={"grasp": {"attachment": attachment}}) is None
    assert evidence["attachment"] == attachment
    # The part leaves its support with the next arm motion, and no calculated retreat binds that motion.
    lower = [3.0, 1.0, 0.995, *down]
    assert _kmr_step(evidence, targets, "move_cartesian", target=lower, facts=_arm_moved(lower)) is None
    assert "resting_on_support" not in evidence["attachment"] and evidence["attachment"]["held_part"] == "P"
    # A base move may follow at once: its own check finds whether the base can drive with the arm where it is.
    assert _kmr_step(evidence, targets, "move_base", target_pose=[2.0, 1.0, 0.0],
                     base=evidence["snapshot"]["base_pose"]) is None
    # A release leaves the gripper empty and the part where its check found it resting.
    rest = [2.5, 1.5, 0.9, 0.0, 0.0, 0.0, 1.0]
    assert _kmr_step(evidence, targets, "release_part",
                     facts={"release": {"part_name": "P", "pose": rest, "place": "Rack"}}) is None
    assert evidence["attachment"]["attached"] is False
    row = evidence["observed_parts"]["P"]
    assert row["current_pose"] == rest and row["part_location"] == "Rack" and row["source"]["kind"] == "projected"


def test_a_composed_kmr_release_may_take_a_part_pose_of_its_own():
    # The worker's compute_place_targets takes an explicit part_pose; its capability check judges the pose.
    calculated = {"target": [0.0, 0.0, 1.025, 1.0, 0.0, 0.0, 0.0], "destination": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}
    targets = {}
    assert _kmr_step({"snapshot": {}}, targets, "compute_place_targets", part_pose=[0.0, 0.0, 1.0],
                     facts={"calculated_targets": calculated}) is None
    assert targets["compute_place_targets"] == calculated


def _kmr_completion(*, trace, stated, snapshot, evidence=None, part_context=None, physical=None, start=None,
                    neighbors=None):
    from cais_spade_llm.recovery_framework.kmr_composition import project_kmr_completion
    from cais_spade_llm.recovery_framework.kmr_recovery import transport_pose

    scene = {**_kmr_scene([-0.25, 0.5, 1.16]), "Storage": {"KMR_docking_pose": [-5.54, 2.13, 0, 0, 0, 1.57]},
             "machines": [{"resource_id": "M1", "KMR_docking_pose": [-4.7, 2.2, 0, 0, 0, 1.57]}]}
    recorded = {"inputs": {"scene": scene}, "snapshot": {}, **(evidence or {})}
    # Unless a test says otherwise the base stays where it is, with the arm parked.
    base = recorded["snapshot"].setdefault("base_pose", [-5.54, 2.13, 0, 0, 0, 0.70682518, 0.70738827])
    recorded["snapshot"].setdefault("current_pose", transport_pose(scene, base, [0.0, 0.0, 0.0, *_KMR_TOOL_DOWN]))
    return project_kmr_completion(task={"resource_jid": "kmr", "part_name": "P", "expected_end_state": stated,
                                        "neighbor_end_state": neighbors},
        trace=trace, recovery_snapshot=start or ({} if evidence else {"base_pose": base}), projected_snapshot=snapshot,
        part_context=part_context or {}, grounding_context={}, resource_evidence=recorded,
        physical_evidence=physical or {})


@pytest.mark.parametrize("base, stated, reached", [
    ([-4.7, 2.2], {"resource_location": "M1"}, "M1"), ([-3.3, 1.3], {"resource_location": None}, None),
    # A stated place the base did not reach takes where it ended, so the postcondition refuses it.
    ([-3.3, 1.3], {"resource_location": "M1"}, None), ([-4.7, 2.2], {"resource_location": None}, "M1")])
def test_a_kmr_location_the_event_states_is_compared_with_the_dock_its_base_reached(base, stated, reached):
    # Facing as the docks declare, yaw 1.57.
    snapshot = {"resource_location": "Storage", "current_location": "Storage",
                "resource_core": {"current_location": "Storage"}, "occupancy": {"location": "Storage"},
                "base_pose": [*base, 0, 0, 0, 0.70682518, 0.70738827]}
    result = _kmr_completion(trace=[{"step_index": 0, "primitive": "move_base", "resolved_params": {}}], stated=stated,
                             snapshot=snapshot, evidence={"snapshot": {"base_pose": snapshot["base_pose"]}})
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    projected = result["projected_snapshot"]
    assert projected["resource_location"] == reached and projected["current_location"] == reached
    # Written as a declared effect writes it: into the mirrors and the occupancy record too.
    assert projected["resource_core"]["current_location"] == reached and projected["occupancy"]["location"] == reached
    assert result["resource_evidence"]["snapshot"]["resource_location"] == reached


@pytest.mark.parametrize("base, stated, location, status, reason", [
    ([-3.3, 1.3], {}, "Storage", "INFEASIBLE", "The program ends at no named place and the event states no resource_location"),
    ([-4.7, 2.2], {}, "Storage", "INFEASIBLE", "The program ends at M1 and the event states no resource_location"),
    # From one unnamed spot to another the names match, but the base moved all the same.
    ([-3.3, 1.3], {}, None, "INFEASIBLE", "The program moves the base and the event states neither resource_location nor pose"),
    ([-3.3, 1.3], {"pose": {"x": -3.3, "y": 1.3, "yaw": 0.0}}, None, "FEASIBLE", None),
    ([-5.54, 2.13], {}, "Storage", "FEASIBLE", None)])
def test_a_kmr_move_the_event_leaves_out_writes_no_location(base, stated, location, status, reason):
    snapshot = {"resource_location": location, "base_pose": [*base, 0, 0, 0, 0.70682518, 0.70738827]}
    start = {"base_pose": snapshot["base_pose"] if base == [-5.54, 2.13] else [-1.0, 1.0, 0, 0, 0, 0, 1]}
    result = _kmr_completion(trace=[{"step_index": 0, "primitive": "move_base", "resolved_params": {}}],
                             stated={"resource_state": "positioned", **stated}, snapshot=snapshot, start=start,
                             evidence={"snapshot": {"base_pose": snapshot["base_pose"]}})
    assert result["feasibility_status"] == status and result.get("reason") == reason
    if status == "FEASIBLE":
        assert result["projected_snapshot"]["resource_location"] == location


def test_a_named_move_leaves_the_tool_at_its_recorded_posture():
    import json
    from pathlib import Path

    from cais_spade_llm.resources.robot.recovery_composition import _named_pose

    arm = json.loads((Path(__file__).parent / "fixtures/recorded_arm_chains.json").read_text())["arms"]["ur5e-3"]
    physical = {"jids": {"ur5e-3": "arm"}, "resource_models": {"ur5e-3": {
        "configuration": arm["configuration"], "named_joint_poses": arm["named_joint_poses"]}}}
    pose = _named_pose("arm", "home", physical)
    assert set(pose) == {"x", "y", "z", "qx", "qy", "qz", "qw"}
    assert sum(pose[axis] ** 2 for axis in ("qx", "qy", "qz", "qw")) == pytest.approx(1.0)
    # A posture or resource with no recorded joints leaves the pose unknown, as the primitive declares.
    assert _named_pose("arm", "ready", physical) is None and _named_pose("other", "home", physical) is None


def test_a_composition_prompt_keeps_what_repeats_ahead_of_what_changes():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _render_primitive_generation_prompt,
    )

    event = {"outline_id": "R1", "resource_jid": "kmr"}
    model_ref = "/resources/kmr/recovery_des_model"
    prompt = _render_primitive_generation_prompt({
        "primitive_active_outline_event_token": event, "primitive_active_outline_event": event,
        "primitive_visible_catalog": [{"name": "move_base"}],
        "primitive_served_context": {"/primitive_contracts/move_base": {"name": "move_base"}, model_ref: {"events": []}}})
    order = [prompt.index(name) for name in (
        "Task and Role", "Visible Primitive Catalog", "Served Context: resource model",
        "Served Context (retrieved earlier in this event)", "Active DES Transition (token)", "Output Contract")]
    assert order == sorted(order)
    # The model has its own section, so a lookup asked for first or last does not move it.
    assert prompt.count(model_ref) == 1 and prompt.index(model_ref) < order[3]


def test_a_composition_retry_lists_every_finding_its_event_has_had():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _record_composition_findings,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _render_primitive_generation_prompt,
    )

    def finding(outline_id, reason, step):
        return {"outline_id": outline_id, "resource_jid": "arm", "constraint_code": "composition_infeasible",
                "reason": reason, "step_index": step}
    override, line = finding("R5", "pick override", 1), finding("R5", "straight line", 7)
    session = {}
    for findings in ([override], [line], [override], [finding("R4", "other event", 2)]):
        _record_composition_findings(session, findings)
    # Each distinct finding once, the latest last.
    assert [row["reason"] for row in session["composition_rejection_history"]] == [
        "straight line", "pick override", "other event"]
    event = {"outline_id": "R5", "resource_jid": "arm", "event_name": "move"}
    session["primitive_rejection_feedback"] = [override]
    prompt = _render_primitive_generation_prompt({"session_state": session, "primitive_active_outline_event_token": event,
                                                  "primitive_active_outline_event": event})
    feedback = prompt.split("Previous Validation Feedback (read first)\n")[1]
    assert feedback.index("straight line") < feedback.index("pick override")
    assert feedback.count("pick override") == 1 and "other event" not in feedback


def test_a_served_resource_model_has_no_duplicate_catalog_and_takes_one_line():
    import json

    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _render_primitive_generation_prompt,
    )
    from cais_spade_llm.resources.recovery_composition import composition_grounding_context

    model = {"events": [{"event_name": "pick_grasp", "updates": {"part_location": {"set": "arm"}}}],
             "primitive_catalog": [{"name": "open_gripper", "primitive_kind": "gripper"}], "current_valuation": {}}
    prepared = {"recovery_resources": {"arm": {"resource_type": "robot", "recovery_des_model": deepcopy(model),
                                               "resource_evidence": {"observed_parts": {}}}}}
    context = composition_grounding_context(session={"accepted_primitive_program": []}, prepared=prepared,
                                            task={"resource_jid": "arm"})
    served = context["resources"]["arm"]["recovery_des_model"]
    assert "primitive_catalog" not in served and served["events"] == model["events"]
    # The owner checks read the prepared model, which keeps its catalog.
    assert prepared["recovery_resources"]["arm"]["recovery_des_model"]["primitive_catalog"] == model["primitive_catalog"]
    event = {"outline_id": "R1", "resource_jid": "arm"}
    prompt = _render_primitive_generation_prompt({"primitive_active_outline_event_token": event,
        "primitive_active_outline_event": event,
        "primitive_served_context": {"/resources/arm/recovery_des_model": served}})
    block = prompt.split("Served Context: resource model\n")[1].split("\n\n")[0]
    assert "\n" not in block and json.loads(block)["/resources/arm/recovery_des_model"] == served


def test_claude_caches_each_stable_part_of_a_composition_prompt_and_nothing_after_it():
    from cais_spade_llm.agents.shared_information.llm_agent import _cached_user_content

    model = "\n\nServed Context: resource model\n{}"
    event = "\n\nServed Context (retrieved earlier in this event)\n{}"
    feedback = "\n\nPrevious Validation Feedback (read first)\nfix it"
    active = "\n\nActive DES Transition (token)\n{}\n\nOutput Contract\n- rules\n"

    def cached(prompt):
        blocks = _cached_user_content(prompt)
        assert "".join(block["text"] for block in blocks) == prompt
        return [block["text"] for block in blocks if "cache_control" in block]

    assert cached("Task" + model + event + active) == ["Task", model, event]
    # With only a model served, or nothing yet, the same fixed text is still its own entry.
    assert cached("Task" + model + active) == ["Task", model]
    assert cached("Task" + active) == ["Task"]
    # Feedback changes between turns, so it stays out of what is cached.
    assert cached("Task" + model + feedback + active) == ["Task", model]
    assert _cached_user_content("a prompt with none of these sections") == "a prompt with none of these sections"


def test_the_composition_prompt_points_a_job_primitive_at_the_job_record():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _job_composition_guidance,
    )

    def prepared(*names):
        return {"composition_validation_version": 1,
                "recovery_resources": {"m2": {"execution_primitive_catalog": [{"name": name} for name in names]}}}

    guidance = _job_composition_guidance(prepared("pause_job", "resume_job"), "m2")
    assert "/resources/m2/resource_evidence/processing_observations/jobs" in guidance
    assert "start_job opens a new job, so it takes no job_id" in guidance
    assert _job_composition_guidance(prepared("move_cartesian"), "m2") == ""


def _kmr_place(part_xy: list[float]) -> tuple[dict, dict, dict]:
    """Calculate a release on the real scene as compute_place_targets does, for a KMR parked on the M2 dock."""
    import json

    from cais_spade_llm.recovery_framework import SCENE_PATH
    from cais_spade_llm.recovery_framework.kmr_recovery import release_targets, transport_pose

    scene = json.loads(SCENE_PATH.read_text())
    base = [-2.7, 2.2, 0.0, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    tool = transport_pose(scene, base, [0.0, 0.0, 0.0, *_KMR_TOOL_DOWN])
    calculated = release_targets(scene, [0.0, 0.0, 0.025, *_KMR_TOOL_DOWN], tool, base,
                                 [*part_xy, 1.0, 0.0, 0.0, 0.0, 1.0])
    return {"inputs": {"scene": scene}, "snapshot": {"base_pose": base, "current_pose": tool}}, {}, calculated


def _arm_moved(end: list[float], *, joints: list[float] | None = None) -> dict:
    """What composition's MoveIt check leaves for a KMR arm motion whose path ends with the TCP at ``end``."""
    return {"composed_motion": {"end_joints": list(joints or [0.0] * 7), "end_pose": list(end)}}


def test_a_kmr_release_is_judged_by_its_path_not_by_the_routes_calculated_for_it():
    evidence, targets, calculated = _kmr_place([-2.7, 1.35])
    # The calculation still publishes its routes, for a program to bind where it chooses.
    assert calculated["retreat"] == calculated["approach"]
    assert calculated["transfer_waypoints"] and calculated["withdraw_waypoints"]
    assert _kmr_step(evidence, targets, "compute_place_targets", facts={"calculated_targets": calculated}) is None
    assert targets["compute_place_targets"] == calculated
    # Straight to the approach, a named pose or a relative move: each step is judged by MoveIt's check of the path it
    # flies (composed_motion_check), never against the calculated routes.
    for primitive, params in (("move_cartesian", {"target": calculated["approach"]}),
                              ("move_cartesian", {"target": calculated["target"]}),
                              ("move_to_named_pose", {"pose_name": "home"}),
                              ("move_relative", {"offset": [0.0, 0.0, 0.1]})):
        assert _kmr_step(evidence, targets, primitive, facts=_arm_moved(calculated["transport"]), **params) is None
    assert evidence["snapshot"]["current_pose"] == calculated["transport"]


def test_a_kmr_pick_may_come_by_any_path_its_check_accepts():
    import json

    from cais_spade_llm.recovery_framework import SCENE_PATH
    from cais_spade_llm.recovery_framework.kmr_recovery import pick_targets, transport_pose

    scene = json.loads(SCENE_PATH.read_text())
    m1 = next(row for row in scene["machines"] if row["resource_id"] == "M1")
    dock, fixture = m1["KMR_docking_pose"], m1["workholding_pose"]
    base = [dock[0], dock[1], 0.0, 0.0, 0.0, math.sin(dock[5] / 2), math.cos(dock[5] / 2)]
    tool = transport_pose(scene, base, [0.0, 0.0, 0.0, *_KMR_TOOL_DOWN])
    part = [*fixture[:3], 0.0, 0.0, 0.0, 1.0]
    calculated = pick_targets(scene, "KET4_Square_4mm", part, tool, base, location="M1")
    evidence = {"inputs": {"scene": scene}, "snapshot": {"base_pose": base, "current_pose": tool},
                "observed_parts": {"KET4_Square_4mm": {"current_pose": part}}}
    targets = {}
    assert calculated["approach_waypoints"]
    assert _kmr_step(evidence, targets, "compute_pick_targets", facts={"calculated_targets": calculated}) is None
    # A straight approach to a part behind the arm is no longer refused by a rule: MoveIt checks the path it flies.
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["approach"],
                     facts=_arm_moved(calculated["approach"])) is None
    assert evidence["snapshot"]["current_pose"] == calculated["approach"]


def test_a_kmr_release_needs_no_visit_to_its_calculated_approach():
    evidence, targets, calculated = _kmr_place([-2.7, 1.35])
    assert _kmr_step(evidence, targets, "compute_place_targets", facts={"calculated_targets": calculated}) is None
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["target"],
                     facts=_arm_moved(calculated["target"])) is None
    # Where the part comes to rest is the release check's fact, not whether the approach was visited.
    rest = [-2.7, 1.35, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert _kmr_step(evidence, targets, "release_part",
                     facts={"release": {"part_name": "P", "pose": rest, "place": "M2 staging tray"}}) is None


@pytest.mark.parametrize("left_out", [{}, {"waypoints": None}, {"waypoints": []}])
def test_a_kmr_turn_under_the_minimum_angle_calculates_no_waypoints(left_out):
    evidence, targets, calculated = _kmr_place([-3.3, 2.0])
    assert calculated["transfer_waypoints"] == [] and calculated["withdraw_waypoints"] == []
    assert _kmr_step(evidence, targets, "compute_place_targets", facts={"calculated_targets": calculated}) is None
    for pose in (calculated["approach"], calculated["target"], calculated["retreat"], calculated["transport"]):
        assert _kmr_step(evidence, targets, "move_cartesian", target=pose, facts=_arm_moved(pose), **left_out) is None


def test_a_kmr_lift_is_judged_by_its_path_not_by_the_route_calculated_for_it():
    from cais_spade_llm.recovery_framework.kmr_recovery import pick_targets

    evidence, targets, _ = _kmr_place([-2.7, 1.35])
    state = evidence["snapshot"]
    part = [-2.7, 1.35, 1.0, 0.0, 0.0, 0.0, 1.0]
    evidence["observed_parts"] = {"P": {"current_pose": part}}
    calculated = pick_targets(evidence["inputs"]["scene"], "P", part, state["current_pose"], state["base_pose"],
                              location=None)
    assert calculated["lift"] == calculated["retreat"] and calculated["lift_waypoints"]
    assert _kmr_step(evidence, targets, "compute_pick_targets", facts={"calculated_targets": calculated}) is None
    attachment = {"attached": True, "part_name": "P", "held_part": "P", "resting_on_support": True,
                  "grasp_transform": [0.0, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]}
    assert _kmr_step(evidence, targets, "grasp_part", part_name="P", facts={"grasp": {"attachment": attachment}}) is None
    # A named pose or a straight move to the lift pose carries no rule of its own; the first one lifts the part.
    assert _kmr_step(deepcopy(evidence), deepcopy(targets), "move_to_named_pose", pose_name="transport",
                     facts=_arm_moved(calculated["lift"])) is None
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["lift"],
                     facts=_arm_moved(calculated["lift"])) is None
    assert "resting_on_support" not in evidence["attachment"]


@functools.lru_cache(maxsize=1)
def _ur5e1_breakdown_capture() -> dict | None:
    """The recorded ur5e-1 breakdown cell (capture_4f9097aa), read once; None when it is not on this machine."""
    from cais_spade_llm.recovery_framework.capture_files import read_capture
    from cais_spade_llm.recovery_framework.recovery_observations import RECORDS

    path = RECORDS / "capture_4f9097aa46bf450ca0717840262806a4" / "observations.json"
    return read_capture(path) if path.exists() else None


def _planar(x: float, y: float, yaw: float) -> list[float]:
    return [x, y, 0.0, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


def _kmr_cell(*, base: list[float] | None = None, joints: list[float] | None = None) -> tuple[dict, object, dict]:
    """The recorded KMR of the ur5e-1 breakdown cell for its step checks: its evidence, event scene and floor map.

    The base stands on ``base`` (its captured pose by default), the arm at ``joints`` (the captured ones).
    """
    from cais_spade_llm.recovery_framework.outline_draft import _navigation_floor
    from cais_spade_llm.resources.robot.recovery_composition import RobotEventScene

    observations = _ur5e1_breakdown_capture()
    if observations is None:
        pytest.skip("the recorded ur5e-1 breakdown capture is not on this machine")
    physical = observations["physical_evidence"]
    captured = physical["snapshot"]["resources"]["KMR"]
    evidence = {"inputs": observations["inputs"],
                "snapshot": {"base_pose": list(base or captured["base_pose"]),
                             "joint_positions": list(joints or captured["joint_positions"])}}
    scene = RobotEventScene(task={"resource_jid": observations["jids"]["KMR"], "part_name": "KET4_Square_4mm"},
                            program=[{"primitive": name} for name in ("move_base", "grasp_part", "release_part")],
                            resource_id="KMR", grounding_context={}, physical_evidence=physical,
                            recovery_des_model={"events": []}, motion_model=lambda: None)
    return evidence, scene, _navigation_floor()


def _kmr_checked(evidence: dict, scene: object, floor: dict, index: int, primitive: str, params: dict,
                 snapshot: dict) -> dict:
    from cais_spade_llm.recovery_framework.kmr_composition import check_kmr_step

    return check_kmr_step(evidence, scene, floor, index=index, primitive=primitive, params=params,
                          recovery_snapshot=snapshot, checked={"feasibility_status": "FEASIBLE", "allowed": True})


def test_a_kmr_base_moves_only_where_its_controller_and_a_route_allow_it():
    north = math.pi / 2
    captured = _kmr_cell()[0]["snapshot"]["joint_positions"]

    def drive(target, *, held=None, gripper="open", base=None, joints=None, floor=None):
        evidence, scene, recorded = _kmr_cell(base=base, joints=joints)
        if held:
            evidence["attachment"] = {"attached": True, "part_name": held, "held_part": held,
                                      "grasp_transform": [0.0, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]}
        return _kmr_checked(evidence, scene, recorded if floor is None else floor, 0, "move_base",
                            {"target_pose": target}, {"held_part": held, "gripper_state": gripper})

    # Empty, from the captured base at Storage to M1's dock: a route exists with the arm at its captured joints.
    empty = drive([-4.7, 2.2, north])
    assert empty["feasibility_status"] == "FEASIBLE", empty.get("reason")
    statement = empty["evidence"]["drive"]["statement"]
    assert "an existence check" in statement and "the floor map is the map script's" in statement
    # The base controller drives an empty KMR only with its gripper open.
    closed = drive([-4.7, 2.2, north], gripper="closed")
    assert closed["feasibility_status"] == "INFEASIBLE" and "only with its gripper open" in closed["reason"]
    # Loaded, with the TCP at the transport pose of its base, as the worker's confirm_carrying needs it.
    loaded = drive([-2.7, 2.2, north], held="KET4_Square_4mm", base=_planar(-4.7, 2.2, north))
    assert loaded["feasibility_status"] == "FEASIBLE", loaded.get("reason")
    assert loaded["evidence"]["drive"]["held_part"] == "KET4_Square_4mm"
    # With the arm turned off that pose, or the tool tilted, the worker refuses to carry.
    out = drive([-2.7, 2.2, north], held="KET4_Square_4mm", base=_planar(-4.7, 2.2, north),
                joints=[captured[0] + 0.6, *captured[1:]])
    assert out["feasibility_status"] == "INFEASIBLE"
    assert out["reason"].startswith("move_base step 0 (KMR): KMR carrying TCP has not reached the configured transport "
                                    "posture: the TCP at")
    assert "the worker's confirm_carrying allows 0.015 m (its code constant, not a recorded fact)" in out["reason"]
    tilted = drive([-2.7, 2.2, north], held="KET4_Square_4mm", base=_planar(-4.7, 2.2, north),
                   joints=[*captured[:5], captured[5] + 0.3, captured[6]])
    assert tilted["feasibility_status"] == "INFEASIBLE" and "requires a downward observed gripper" in tilted["reason"]
    # Beside M2's staging tray facing north, no way in keeps the arm clear: the search says what it met.
    blocked = drive([-3.25, 1.45, north], base=_planar(-2.7, 2.2, north))
    assert blocked["feasibility_status"] == "INFEASIBLE"
    assert "at its composed joints" in blocked["reason"] and blocked["reason"].endswith("not a recorded fact")
    # A goal whose padded footprint lies on an obstacle of the floor map is one the base controller refuses.
    inside = drive([-4.13, 1.82, north])
    assert inside["feasibility_status"] == "INFEASIBLE" and "on the floor map" in inside["reason"]
    # Without the floor map there is nothing to search a route on.
    assert drive([-4.7, 2.2, north], floor={})["feasibility_status"] == "NEEDS_CONTEXT"


def test_a_kmr_grasp_is_checked_where_the_worker_checks_it():
    from cais_spade_llm.recovery_framework.kmr_composition import _tool_pose

    evidence, scene, floor = _kmr_cell(base=_planar(-4.7, 2.2, math.pi / 2))
    tool = _tool_pose(evidence, scene, evidence["snapshot"]["joint_positions"])

    def grasp(below: float, **params) -> tuple[dict, list[float]]:
        part = [tool[0], tool[1], tool[2] - below, 0.0, 0.0, 0.0, 1.0]
        evidence["observed_parts"] = {"KET4_Square_4mm": {"current_pose": part}}
        return _kmr_checked(evidence, scene, floor, 1, "grasp_part", {"part_name": "KET4_Square_4mm", **params},
                            {"held_part": None}), part

    held, part = grasp(0.025)
    assert held["feasibility_status"] == "FEASIBLE", held.get("reason")
    attachment = held["evidence"]["grasp"]["attachment"]
    # The part is attached where it is, by the transform from the TCP the program reached.
    assert compose(tool, attachment["grasp_transform"]) == pytest.approx(part, abs=1e-9)
    assert attachment["resting_on_support"] is True
    assert held["evidence"]["grasp"]["tcp_to_part_m"] == pytest.approx(0.025)
    far, _ = grasp(0.07)
    assert far["feasibility_status"] == "INFEASIBLE"
    assert far["reason"].startswith("grasp_part step 1 (KMR): KMR grasp requires the part at the gripper: the TCP at")
    assert "is 0.0700 m from KET4_Square_4mm's observed origin" in far["reason"] and "allows 0.06 m" in far["reason"]
    moved, _ = grasp(0.025, initial=[tool[0] + 0.02, tool[1], tool[2] - 0.025])
    assert moved["feasibility_status"] == "INFEASIBLE" and "Part moved before grasp" in moved["reason"]
    evidence["observed_parts"] = {}
    unseen = _kmr_checked(evidence, scene, floor, 1, "grasp_part", {"part_name": "KET4_Square_4mm"}, {"held_part": None})
    assert unseen["feasibility_status"] == "NEEDS_CONTEXT" and "needs its recorded pose" in unseen["reason"]


def test_a_kmr_release_rests_where_a_recorded_support_holds_it():
    from cais_spade_llm.recovery_framework.kmr_composition import _inverse, _tool_pose

    evidence, scene, floor = _kmr_cell(base=_planar(-2.7, 2.2, math.pi / 2))
    tool = _tool_pose(evidence, scene, evidence["snapshot"]["joint_positions"])
    turned = [0.0, 0.0, math.sqrt(0.5), -math.sqrt(0.5)]

    def release(z: float) -> dict:
        # Held so that the released part stands at (-2.6, 1.4, z), over the M2 staging tray.
        evidence["attachment"] = {"attached": True, "part_name": "KET4_Square_4mm", "held_part": "KET4_Square_4mm",
                                  "grasp_transform": compose(_inverse(tool), [-2.6, 1.4, z, *turned])}
        return _kmr_checked(evidence, scene, floor, 2, "release_part", {}, {"held_part": "KET4_Square_4mm"})

    placed = release(1.0)
    assert placed["feasibility_status"] == "FEASIBLE", placed.get("reason")
    rest = placed["evidence"]["release"]
    assert (rest["place"], rest["area"]) == ("M2 staging tray", "M2.staging")
    assert rest["support_collisions"] == ["M2 staging tray::link::surface"]
    dropped = release(1.02)
    assert dropped["feasibility_status"] == "INFEASIBLE"
    assert dropped["reason"].startswith("release_part step 2 (KMR): KET4_Square_4mm released at (-2.6000, 1.4000, "
                                        "1.0200) rests on no recorded support")


@pytest.mark.parametrize("offset", [0.0, 0.005, 0.007, 0.2])
def test_a_kmr_program_may_end_with_its_arm_where_its_steps_left_it(offset):
    from cais_spade_llm.recovery_framework.kmr_recovery import transport_pose

    base = [-5.54, 2.13, 0, 0, 0, 0.70682518, 0.70738827]
    parked = transport_pose(_kmr_scene([-0.25, 0.5, 1.16]), base, [0.0, 0.0, 0.0, *_KMR_TOOL_DOWN])
    tool = [parked[0] + offset, *parked[1:]]
    # The arm's end posture is no completion rule: a base move that follows is checked where the arm then is.
    result = _kmr_completion(trace=[], stated={"resource_state": "positioned"}, snapshot={"base_pose": base},
                             evidence={"snapshot": {"base_pose": base, "current_pose": tool}}, start={"base_pose": base})
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")


def test_a_kmr_tool_left_tilted_is_refused_where_the_worker_refuses_it():
    base = [-5.54, 2.13, 0, 0, 0, 0.70682518, 0.70738827]
    tool = [-6.0, 1.9, 1.16, 0.0, 0.0, 0.0, 1.0]
    # The completion takes the tilted tool as the steps left it; the loaded drive after it is refused with the
    # worker's own text (test_a_kmr_base_moves_only_where_its_controller_and_a_route_allow_it).
    result = _kmr_completion(trace=[], stated={"resource_state": "positioned"}, snapshot={"base_pose": base},
                             evidence={"snapshot": {"base_pose": base, "current_pose": tool}}, start={"base_pose": base})
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    assert result["resource_evidence"]["snapshot"]["current_pose"] == tool


def test_the_composition_prompt_names_the_kmr_facts_and_gives_no_recipe():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _kmr_composition_guidance,
    )

    guidance = _kmr_composition_guidance("kmr")
    for name in ("lift_waypoints", "transfer_waypoints", "withdraw_waypoints"):
        assert name in guidance
    # The place calculation reads a given part_pose, which composition passes to it.
    assert "reads the place from a given part_pose, else from the event's part_location" in guidance
    for recipe in ("Bind ", "bind ", "Whatever resource_state", "Custody completion", "adapt its"):
        assert recipe not in guidance


def test_a_kmr_grasp_gives_it_custody():
    grasp = {"step_index": 0, "primitive": "grasp_part", "resolved_params": {"part_name": "P"}}
    stated = {"held_part": "P", "part_location": "kmr"}
    # No retreat has to follow the grasp: the part leaves its support with the next motion, checked there.
    result = _kmr_completion(trace=[grasp], stated=stated, snapshot={"held_part": "P"}, part_context={"part_location": "M1"})
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    assert result["projected_part_context"]["part_location"] == "kmr"
    assert result["projected_part_context"]["part_holder_resource_jid"] == "kmr"
    # A grasp of another part than the event's is refused with both names.
    other = _kmr_completion(trace=[{**grasp, "resolved_params": {"part_name": "Q"}}], stated=stated,
                            snapshot={"held_part": "Q"})
    assert other == {"feasibility_status": "INFEASIBLE", "reason": "grasp_part step 0 names Q, and the event's part is P"}


def test_a_kmr_grasp_out_of_a_machine_leaves_what_the_machine_s_model_declares():
    slot = {"reference": "part_name", "scope": "resource"}
    unload = {"event_id": 1, "event_name": "pick_grasp", "participants": ["robot", "M9"],
              "parameter_bindings": {"resource_id": {"equals": "robot"}, "origin_resource_location": {"equals": "M9"}},
              "guards": {"resource_state": {"equals": "completed"}, "part_name": {"equals_from_param": "part_name"}},
              "updates": {"resource_state": {"set": "idle"}, "part_name": {"set": None}}}
    physical = {"jids": {"M9": "m9"}, "environment_models": {"M9": {
        "state_variables": {"part_name": slot, "resource_state": {"scope": "resource"}}, "events": [unload]}},
        "environment_valuation": {"M9": {"part_name": "P", "resource_state": "completed"}}}
    grasp = {"step_index": 0, "primitive": "grasp_part", "resolved_params": {"part_name": "P"}}
    motion = {"step_index": 1, "primitive": "move_relative", "resolved_params": {}}

    def pick(neighbors):
        return _kmr_completion(trace=[grasp, motion], stated={"held_part": "P", "part_location": "kmr"},
                               snapshot={"held_part": "P"}, part_context={"part_location": "M9"}, physical=physical,
                               neighbors=neighbors)

    # The machine's model declares what taking its part leaves it, whichever resource takes it (a robot's unload here).
    result = pick({"m9": {"part_name": None, "resource_state": "idle"}})
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    assert result["projected_resources"]["m9"] == {"part_name": None, "resource_state": "idle"}
    # A value the event leaves out is not written.
    assert pick({"m9": {"part_name": None}})["projected_resources"]["m9"] == {"part_name": None}


def test_a_kmr_release_gives_the_part_to_the_slot_that_receives_it():
    part = {"reference": "part_name", "scope": "resource"}
    physical = {"product_name": "board", "jids": {"Rack": "rack"},
                "environment_models": {"Rack": {"state_variables": {"zone_1_part": part, "zone_2_part": part},
                    "assignments": {"zones": [{"zone": 1, "pose": [0.0, 0.5, 1.0]}, {"zone": 2, "pose": [0.2, 0.5, 1.0]}]}}},
                "environment_valuation": {"Rack": {"zone_1_part": None, "zone_2_part": None}},
                "geometry": {"regions": _RACK_ZONES}}
    pose = [0.2, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0]
    # The release check found the part resting on the rack, where the KMR's tool and grasp put it.
    release = _released({"step_index": 0, "primitive": "release_part", "resolved_params": {}}, pose=pose, place="Rack")
    motion = {"step_index": 1, "primitive": "move_cartesian", "resolved_params": {}}
    result = _kmr_completion(trace=[release, motion], stated={"held_part": None, "part_location": "Rack"},
                             snapshot={"held_part": None}, physical=physical, neighbors={"rack": {"zone_2_part": "P"}})
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    assert result["projected_resources"]["rack"] == {"zone_1_part": None, "zone_2_part": "P"}
    # A slot the event leaves out is not written; the RA refuses such a place (unstated_slot).
    unstated = _kmr_completion(trace=[release, motion], stated={"held_part": None, "part_location": "Rack"},
                               snapshot={"held_part": None}, physical=physical)
    assert "rack" not in unstated.get("projected_resources", {})
    assert result["projected_part_context"]["part_location"] == "Rack"
    assert result["projected_part_context"]["pose_source"] == "validated_composition_projection"
    assert result["resource_evidence"]["observed_parts"]["P"]["current_pose"] == pose
    assert result["resource_evidence"]["observed_parts"]["P"]["part_location"] == "Rack"
    # A zone holding another part takes no more.
    physical["environment_valuation"]["Rack"]["zone_2_part"] = "Q"
    occupied = _kmr_completion(trace=[release, motion], stated={"held_part": None, "part_location": "Rack"},
                               snapshot={"held_part": None}, physical=physical)
    assert occupied == {"feasibility_status": "INFEASIBLE", "reason": "Rack's zone_2_part holds Q"}
    # A release with no checked resting place, or of another part than the event's, is refused with that fact.
    unchecked = _kmr_completion(trace=[{"step_index": 0, "primitive": "release_part", "resolved_params": {}}],
                                stated={"part_location": "Rack"}, snapshot={"held_part": None}, physical=physical)
    assert unchecked["reason"] == "release_part step 0 has no checked resting place"
    other = _kmr_completion(trace=[_released(release, pose=pose, place="Rack", part_name="Q")],
                            stated={"part_location": "Rack"}, snapshot={"held_part": None}, physical=physical)
    assert other["reason"] == "release_part step 0 puts down Q, and the event's part is P"


def test_a_part_an_accepted_program_put_down_is_there_for_every_resource():
    from cais_spade_llm.resources.recovery_composition import composition_grounding_context

    placed = {"current_pose": [1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0], "source": {"kind": "projected", "contract_fingerprint": "f"}}
    prepared = {"recovery_resources": {
        "kmr": {"resource_type": "kmr", "resource_evidence": {"inputs": {"scene": {
            "Storage": {"KMR_docking_pose": [-5.54, 2.13, 0, 0, 0, 1.57]}}}, "observed_parts": {}}},
        "arm": {"resource_type": "robot", "resource_evidence": {"observed_parts": {
            "P": {"current_pose": [9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]}, "Q": {"current_pose": [8.0, 8.0, 8.0]}}}}}}
    accepted = {"outline_id": "put", "resource_jid": "kmr", "part_name": "P", "composition_validation": {
        "feasibility_status": "FEASIBLE", "projected_snapshot": {"resource_location": "M1"}, "projected_part_context": {},
        "resource_evidence": {"observed_parts": {"P": placed, "Q": {"current_pose": [0.0, 0.0, 0.0]}}}}}
    context = composition_grounding_context(session={"accepted_primitive_program": [accepted]}, prepared=prepared,
                                            task={"resource_jid": "arm", "part_name": "P"})
    arm = context["resources"]["arm"]["resource_evidence"]["observed_parts"]
    assert arm["P"] == placed
    # What the placing resource only recorded is not news to the others.
    assert arm["Q"] == {"current_pose": [8.0, 8.0, 8.0]}
    # The KMR's docks are a lookup, as [x, y, yaw].
    assert context["resources"]["kmr"]["docks"] == {"Storage": [-5.54, 2.13, 1.57]}


def test_the_served_model_is_the_same_text_at_every_event_of_a_resource(contract):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _serve_context_requests,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _compact_json

    prepared = _prepared(contract)
    model = prepared["recovery_resources"]["owner"]["recovery_des_model"]
    model["current_valuation"] = {"mode": "fault"}

    def block(session):
        served, errors = _serve_context_requests(session_state=session, prepared_recovery_request=prepared,
            outline_event=contract["task"], context_requests=["/resources/owner/recovery_des_model"])
        assert errors == []
        return _compact_json(served)

    # This block carries a prompt cache point, so an accepted program must not change a byte of it.
    assert block({}) == block({"composition_prefix": [_row(contract)]})
    assert "current_valuation" not in block({})
    # The owner check reads the prepared model, which keeps the capture's valuation.
    assert model["current_valuation"] == {"mode": "fault"}


def test_a_projected_snapshot_is_the_resources_state_at_the_active_event(contract):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _resolve_context_ref,
    )

    prepared = _prepared(contract)
    prepared["recovery_resources"]["machine"] = {"recovery_snapshot": {"part_name": None, "resource_state": "idle"}}

    def accepted(outline_id, jid, snapshot, **evidence):
        return {"outline_id": outline_id, "resource_jid": jid, "part_name": None, "composition_validation": {
            "feasibility_status": "FEASIBLE", "projected_snapshot": snapshot, **evidence}}

    # The machine's own last program paused it with R, and a later robot event loaded K on it. Each event ran in
    # its own batch session, so these rows are the prefix and this session has accepted nothing itself.
    session = {"accepted_primitive_program": [], "composition_prefix": [
        accepted("pause", "machine", {"part_name": "R", "resource_state": "paused"}),
        accepted("load", "owner", {"mode": "idle"},
                 projected_resources={"machine": {"part_name": "K", "resource_state": "loaded"}})]}

    def serve(ref):
        return _resolve_context_ref(ref=ref, session_state=session, prepared_recovery_request=prepared,
                                    outline_event=contract["task"])

    assert serve("/projected_snapshot/machine") == ({"part_name": "K", "resource_state": "loaded"}, None)
    assert serve("/projected_snapshot/machine") == serve("/resources/machine/snapshot")
    # With no name it is the active event's own resource.
    assert serve("/projected_snapshot") == serve("/resources/owner/snapshot") == ({"mode": "idle"}, None)
    # A capture field no program updates is not served, where it would contradict the projected held_part.
    session["composition_prefix"].append(accepted("release", "machine", {
        "held_part": None, "occupancy": {"held_parts": ["K"]}, "resource_core": {"occupancy": {"held_parts": ["K"]}},
        "held_part_location": "machine", "task_ctx.part_name": "K"}))
    assert serve("/projected_snapshot/machine") == serve("/resources/machine/snapshot")
    served = serve("/projected_snapshot/machine")[0]
    assert served["held_part"] is None
    assert not {"occupancy", "resource_core", "held_part_location", "task_ctx.part_name"} & set(served)


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
    base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    contract["recovery_snapshot"]["base_pose"] = base
    prepared = _prepared(contract)
    entry = prepared["recovery_resources"]["owner"]
    entry.update(resource_type="kmr", execution_primitive_catalog=[
        *contract["primitive_catalog"],
        {"name": sorted(INTERNAL_GRIPPER_PRIMITIVES)[0], "params": {}, "preconditions": {}, "effects": {}}],
        resource_evidence={"inputs": {"scene": _kmr_scene([-0.25, 0.5, 1.16])},
                           "snapshot": {"base_pose": base, "current_pose": [-0.25, 0.5, 1.16, *_KMR_TOOL_DOWN]}})
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


# What the grasp check reads: the TCP on the part's origin, for a grasp 0.1 m above it.
_TCP = {"tcp_offset_z": -0.1}
_GRIP_EVIDENCE = {"controller_config": {"move_group": {"position_tolerance_m": .005}}}
_PART_GEOMETRY = {"geometry": {"parts": {"P": {"local_bounds": [[-.01, .01], [-.01, .01], [-.025, .025]]}}}}
# The recorded areas of a rack's two zones, named as the RA names a zone's area.
_RACK_ZONES = {"Rack.zone_1": {"bounds": [[-0.06, 0.06], [0.4, 0.6], [1.0, 1.05]]},
               "Rack.zone_2": {"bounds": [[0.14, 0.26], [0.4, 0.6], [1.0, 1.05]]}}
_AXES = ("x", "y", "z", "qx", "qy", "qz", "qw")


def _moved(row: dict, end: dict) -> dict:
    """A motion's trace row as composition's MoveIt check leaves it: with the tool pose the motion ended at."""
    return {**row, "projected_snapshot": {"current_pose": end},
            "capability_evidence": {"evidence": {"composed_motion": {"end_pose": [end[axis] for axis in _AXES]}}}}


def _released(row: dict, **release) -> dict:
    """A release's trace row as composition's release check leaves it: where the part comes to rest."""
    return {**row, "capability_evidence": {"evidence": {"release": {"part_name": "P", **release}}}}


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
              "observed_pose": {**grasp, "z": 0.2}, **_TCP}
    trace = [{"step_index": 0, "primitive": "compute_pick_targets", "params": {}, "resolved_params": {"part_name": "P"},
              "preview_output": target, "start_snapshot": {}},
             {"step_index": 1, "primitive": "grasp_part", "params": {}, "resolved_params": {"part_name": "P"},
              "start_snapshot": {"current_pose": grasp}},
             {"step_index": 2, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
              "start_snapshot": {"current_pose": grasp}, "projected_snapshot": {"current_pose": retreat}}]
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "pick", "event_name": "pick", "resource_jid": "robot", "part_name": "P",
              "expected_end_state": {"resource_location": "Exit"}}, trace=trace,
        recovery_des_model=model, recovery_snapshot={"resource_state": "at_pick"},
        projected_snapshot={"resource_state": "at_pick", "current_pose": retreat}, part_context={"part_location": "Exit"},
        grounding_context={}, primitive_catalog=[], resource_evidence=_GRIP_EVIDENCE, physical_evidence=_PART_GEOMETRY)
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
    target = {"target_pose": grasp, "retreat_pose": retreat, "model_name": "P_model", "observed_pose": {**grasp, "z": 0.2}, **_TCP}
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
        part_context={"part_location": None}, grounding_context={}, primitive_catalog=[], resource_evidence=_GRIP_EVIDENCE,
        physical_evidence=_PART_GEOMETRY)
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")


def _arm_scene(physical_extra: dict | None = None, program: list | None = None):
    """A ur5e-3 step-check scene on its recorded chain, standing at its named home joints."""
    import json
    from pathlib import Path

    from cais_spade_llm.resources.robot.recovery_composition import RobotEventScene

    arm = json.loads((Path(__file__).parent / "fixtures/recorded_arm_chains.json").read_text())["arms"]["ur5e-3"]
    joints = arm["named_joint_poses"]["home"]["positions"]
    physical = {"jids": {"ur5e-3": "arm"}, "resource_models": {"ur5e-3": {"configuration": arm["configuration"]}},
                "snapshot": {"resources": {"ur5e-3": {"joint_positions": joints}}},
                "resources": {"ur5e-3": {"tcp_offset_z": -0.218}}, **(physical_extra or {})}
    return RobotEventScene(task={"resource_jid": "arm", "part_name": "P"}, program=program or [{"primitive": "grasp_part"}],
                           resource_id="ur5e-3", grounding_context={}, physical_evidence=physical,
                           recovery_des_model={"events": [{"event_name": "pick_grasp"}]}, motion_model=lambda: None)


def test_a_grasp_that_misses_its_part_is_refused_with_the_numbers():
    from cais_spade_llm.recovery_framework.geometry import compose, rotate
    from cais_spade_llm.resources.robot.recovery_composition import _tool_pose, check_robot_step

    geometry = {"geometry": {"parts": {"P": {"local_bounds": [[-.002, .002], [-.002, .002], [0., .05]]}}}}
    scene = _arm_scene(geometry)
    # The grasp is checked at the tool pose the program reached: here the carried joints of the robot's evidence.
    tool = _tool_pose(scene, scene.physical_evidence["snapshot"]["resources"]["ur5e-3"]["joint_positions"])
    tcp = [tool[index] + value for index, value in enumerate(rotate(tool[3:], [0., 0., 0.218]))]

    def grasp(part_x=0., part_z=-.025, physical=None, evidence=None):
        part = [tcp[0] + part_x, tcp[1], tcp[2] + part_z, 0., 0., 0., 1.]
        evidence = evidence if evidence is not None else {
            **_GRIP_EVIDENCE, "observed_parts": {"P": {"current_pose": part}},
            "inputs": {"geometry": {"parts": {"model_map": {"P": "P_model"}}}}}
        used = scene if physical is None else _arm_scene(physical)
        return check_robot_step(evidence, {}, used, index=0, primitive="grasp_part", params={"part_name": "P"},
                                recovery_snapshot={}, checked={"feasibility_status": "FEASIBLE", "allowed": True}), part

    centred, part = grasp()
    assert centred["feasibility_status"] == "FEASIBLE"
    attachment = centred["evidence"]["grasp"]["attachment"]
    # The part is attached where it is, by the transform from the tool pose the program reached.
    assert compose(tool, attachment["grasp_transform"]) == pytest.approx(part, abs=1e-9)
    assert attachment["model_name"] == "P_model" and attachment["resting_on_support"] is True
    # Within the recorded tolerance of the part's top, as the controller's payload check has it.
    assert grasp(part_z=-.054)[0]["feasibility_status"] == "FEASIBLE"
    above, _ = grasp(part_z=-.07)
    assert above["feasibility_status"] == "INFEASIBLE"
    assert above["reason"].startswith("grasp_part step 0 (ur5e-3): The grasp's TCP at")
    assert "is 0.0200 m from P" in above["reason"] and "tolerance 0.005 m" in above["reason"]
    assert "size [0.004, 0.004, 0.05]" in above["reason"]
    beside, _ = grasp(part_x=.01)
    assert beside["feasibility_status"] == "INFEASIBLE" and "is 0.0080 m from P" in beside["reason"]
    # Evidence the check needs is asked for by name, not passed over.
    unbounded, _ = grasp(physical={})
    assert unbounded["feasibility_status"] == "NEEDS_CONTEXT" and "local_bounds" in unbounded["reason"]
    unseen, _ = grasp(evidence=_GRIP_EVIDENCE)
    assert unseen["feasibility_status"] == "NEEDS_CONTEXT" and "pose (observed_parts)" in unseen["reason"]


def _part_slippage_release(rest: list, *, slot: dict | None = None, targets: dict | None = None, part: str = "KET4_Square_4mm"):
    """ur5e-3 of the recorded Part slippage cell releasing ``part`` with the part at ``rest``, through the step check.

    The arm stands at its captured joints; the grasp that puts the part at ``rest`` there is the one it holds by.
    """
    from cais_spade_llm.recovery_framework.capture_files import read_capture
    from cais_spade_llm.recovery_framework.recovery_observations import RECORDS
    from cais_spade_llm.resources.robot.recovery_composition import (
        RobotEventScene,
        _inverse,
        _tool_pose,
        check_robot_step,
    )

    path = RECORDS / "capture_be3e6112cf09475fa6a2d0fb5073af3a" / "observations.json"
    if not path.exists():
        pytest.skip("the recorded Part slippage capture is not on this machine")
    observations = read_capture(path)
    physical = observations["physical_evidence"]
    params = {"part_name": part, **({"assembly_slot": slot} if slot else {})}
    scene = RobotEventScene(task={"resource_jid": physical["jids"]["ur5e-3"], "part_name": part},
                            program=[{"primitive": "release_part", "params": params}], resource_id="ur5e-3",
                            grounding_context={}, physical_evidence=physical, recovery_des_model={"events": []},
                            motion_model=lambda: None)
    tool = _tool_pose(scene, physical["snapshot"]["resources"]["ur5e-3"]["joint_positions"])
    evidence = {"attachment": {"part_name": part, "grasp_transform": compose(_inverse(tool), rest)},
                "inputs": observations["inputs"]}
    return check_robot_step(evidence, targets or {}, scene, index=0, primitive="release_part", params=params,
                            recovery_snapshot={"held_part": part}, checked={"feasibility_status": "FEASIBLE", "allowed": True})


def test_a_released_part_rests_where_a_recorded_support_holds_it():
    # Where flown Part slippage SEQ1 set KET4 down: flush on the belt of buffer zone 1.
    rest = [-0.68, 0.5, 1.017, 0., 0., 1.020801401310305e-05, 0.9999999999478983]
    placed = _part_slippage_release(rest)
    assert placed["feasibility_status"] == "FEASIBLE", placed.get("reason")
    release = placed["evidence"]["release"]
    assert (release["place"], release["area"]) == ("Buffer For Machined parts", "Buffer For Machined parts.zone_1")
    assert release["support_collisions"] == ["Buffer For Machined parts::link::zone_1_belt"]
    assert release["pose"] == pytest.approx(rest, abs=1e-9)
    # Released 2 cm above the belt, the part would drop: no recorded support holds it there.
    dropped = _part_slippage_release([*rest[:2], rest[2] + 0.02, *rest[3:]])
    assert dropped["feasibility_status"] == "INFEASIBLE"
    assert dropped["reason"].startswith("release_part step 0 (ur5e-3): KET4_Square_4mm released at (-0.6800, 0.5000, "
                                        "1.0370) rests on no recorded support")


def test_a_release_into_an_assembly_slot_is_compared_with_the_recorded_slot():
    from scipy.spatial.transform import Rotation

    slot = {"slot_x": 0.0294496, "slot_y": 0.1448242, "part_height": 0.02, "board_top_z": 1.0289916,
            "part_origin_z": 1.0389916, "destination_location": "assembly_board-v1"}
    turned = Rotation.from_euler("z", math.pi / 2).as_quat().tolist()
    seated = [slot["slot_x"], slot["slot_y"], slot["part_origin_z"], *turned]
    contact = {"target_collision_object": "Gear_Plate/Gear_Shaft_1/collision", "axis_tolerance_m": 0.0005,
               "max_contact_depth_m": 0.0005, "target_yaw_rad": math.pi / 2, "yaw_tolerance_rad": 0.005}
    mated = {"compute_place_targets": {"gear_small": {"simulation_mating_contact": contact}}}
    inserted = _part_slippage_release(seated, slot=slot, targets=mated, part="gear_small")
    assert inserted["feasibility_status"] == "FEASIBLE", inserted.get("reason")
    release = inserted["evidence"]["release"]
    assert release["place"] == "assembly_board-v1"
    assert release["assembly_slot"]["target"] == "Gear_Plate/Gear_Shaft_1"
    # With its own place calculation's mating context the controller keeps the seated part's orientation.
    assert release["pose"] == pytest.approx(seated, abs=1e-9)
    # Without one it sets the part in its slot with no turn.
    plain = _part_slippage_release(seated, slot=slot, part="gear_small")
    assert plain["evidence"]["release"]["pose"] == pytest.approx([*seated[:3], 0., 0., 0., 1.], abs=1e-12)
    # The slot the program states is the recorded one, field by field.
    shifted = _part_slippage_release(seated, slot={**slot, "slot_x": 0.03}, targets=mated, part="gear_small")
    assert shifted["feasibility_status"] == "INFEASIBLE"
    assert shifted["reason"] == ("release_part step 0 (ur5e-3): release_part assembly_slot differs from gear_small's "
                                 "calculated slot: slot_x 0.03 differs from 0.0294496")
    # Released off its corridor, the controller refuses the release.
    off = _part_slippage_release([seated[0] + 0.002, *seated[1:]], slot=slot, targets=mated, part="gear_small")
    assert off["feasibility_status"] == "INFEASIBLE" and "lies outside its mating corridor" in off["reason"]


def test_an_accepted_step_carries_the_arm_the_attachment_and_the_rest_into_the_evidence():
    from cais_spade_llm.resources.robot.recovery_composition import advance_robot_program

    attachment = {"part_name": "P", "grasp_transform": [0., 0., .2, 1., 0., 0., 0.], "resting_on_support": True}
    evidence = {"snapshot": {"joint_positions": [0.] * 6}}
    pick = {"target_pose": {"x": 1.}, "travel_z": 1.3}
    advance_robot_program(evidence, {"compute_pick_targets": {"P": pick}}, task={}, primitive="grasp_part", params={},
                          recovery_snapshot={}, checked={"evidence": {"grasp": {"attachment": attachment}}})
    # The grasp attaches the part as found at the tool pose, with the pick context of the event's own calculation.
    assert evidence["attachment"] == attachment and evidence["state"]["_task_ctx"] == pick
    advance_robot_program(evidence, {}, task={}, primitive="move_cartesian", params={}, recovery_snapshot={},
                          checked={"evidence": {"composed_motion": {"end_joints": [.1] * 6}}})
    # The arm is where MoveIt's path ended; the part has left the support it rested on.
    assert evidence["snapshot"]["joint_positions"] == [.1] * 6
    assert "resting_on_support" not in evidence["attachment"]
    advance_robot_program(evidence, {}, task={}, primitive="release_part", params={}, recovery_snapshot={},
                          checked={"evidence": {"release": {"part_name": "P", "pose": [1., 2., 1., 0., 0., 0., 1.],
                                                            "place": "Rack"}}})
    assert evidence["attachment"] == {}
    row = evidence["observed_parts"]["P"]
    assert row["current_pose"] == dict(zip(_AXES, [1., 2., 1., 0., 0., 0., 1.]))
    assert row["source"]["kind"] == "projected" and row["part_location"] == "Rack"


def test_a_cylinder_s_departure_bound_is_its_exact_lowest_point():
    from cais_spade_llm.resources.continuous_motion import interval
    from cais_spade_llm.resources.recovery_support_departure import _cylinder_minimum

    # An affine of the plane z = 1 for an identity pose: distance = z - 1.
    affine = [interval(0.), interval(0.), interval(1.), interval(-1.)]
    upright = {"center": [0., 0., 1.025], "axis": [0., 0., 1.], "radius": .002, "length": .05}
    low, high = _cylinder_minimum(affine, upright)
    # The bound encloses the exact value; the square root across the axis widens it by its rounding only.
    assert low <= 0. <= high and high - low < 1e-9
    lying = {"center": [0., 0., 1.0019], "axis": [1., 0., 0.], "radius": .002, "length": .05}
    low, high = _cylinder_minimum(affine, lying)
    # Lying on its side, the cylinder sinks 0.1 mm into the plane: its rim, not its axis, is the lowest point.
    assert low <= -1e-4 <= high and high - low < 1e-9


@pytest.mark.parametrize("last", ["move_cartesian", "move_to_named_pose"])
def test_a_move_without_a_transfer_leaves_the_robot_where_its_pose_is(monkeypatch, last):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    above = {"x": 0.0, "y": 0.1, "z": 1.1, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    trace = [_moved({"step_index": 0, "primitive": last, "params": {}, "resolved_params": {"pose_name": "home"},
                     "start_snapshot": {}}, above)]
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
    pick = {"target_pose": grasp, "retreat_pose": lift, "model_name": "P_model", "observed_pose": {**grasp, "z": 1.0}, **_TCP}
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
    trace = [_moved(row, row["projected_snapshot"]["current_pose"]) if row["primitive"] == "move_cartesian" else row
             for row in trace]
    # The release check found the part resting on the rack, in zone 2's recorded area.
    trace[5] = _released(trace[5], pose=[0.2, 0.5, 1.0, 0., 0., 0., 1.], place="Rack", area="Rack.zone_2")
    slot = {"reference": "part_name", "scope": "resource"}
    physical = {"product_name": "board", "jids": {"Rack": "rack"},
                "environment_models": {"Rack": {"state_variables": {"zone_1_part": slot, "zone_2_part": slot},
                    "assignments": {"zones": [{"zone": 1, "pose": [0.0, 0.5, 1.0]}, {"zone": 2, "pose": [0.2, 0.5, 1.0]}]}}},
                "environment_valuation": {"Rack": {"zone_1_part": "P", "zone_2_part": None}},
                "geometry": {"regions": _RACK_ZONES}}
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    def complete(end, neighbors=None):
        return project_composition_completion(
            task={"outline_id": "move", "event_name": "move", "resource_jid": "robot", "part_name": "P",
                  "expected_end_state": end, "neighbor_end_state": neighbors}, trace=trace,
            recovery_des_model=model, recovery_snapshot={"resource_state": "idle"},
            projected_snapshot={"resource_state": "placed", "current_pose": away}, part_context={"part_location": "Rack"},
            grounding_context={"resources": {"rack": {"zone_1_part": "P", "zone_2_part": None}}}, primitive_catalog=[],
            resource_evidence=_GRIP_EVIDENCE, physical_evidence={**physical, "geometry": {
                **physical["geometry"], **_PART_GEOMETRY["geometry"]}})

    stated = {"rack": {"zone_1_part": None, "zone_2_part": "P"}}
    result = complete({"resource_location": "Rack", "resource_state": "placed"}, stated)
    # Wherever the motion after the grasp ends, no calculated retreat pose is required (user decision, 2026-11-06):
    # the step checks decide whether the part leaves its support.
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    # The part leaves its source slot and is received by the slot whose area holds it, in one record.
    assert result["projected_resources"]["rack"] == {"zone_1_part": None, "zone_2_part": "P"}
    # A source slot the event leaves out keeps its value: code writes none, and the RA refuses such an event.
    unstated = complete({"resource_location": "Rack", "resource_state": "placed"}, {"rack": {"zone_2_part": "P"}})
    assert unstated["projected_resources"]["rack"]["zone_1_part"] == "P"
    assert result["projected_part_context"]["part_location"] == "Rack"
    assert result["projected_part_context"]["observed_pose"] == dict(zip(_AXES, [0.2, 0.5, 1.0, 0., 0., 0., 1.]))
    assert result["projected_snapshot"]["resource_location"] == "Rack"
    assert result["projected_snapshot"]["resource_state"] == "placed"
    assert result["resource_evidence"]["attachment"] == {}
    # A later pick reads where the part now rests from the row this place leaves.
    assert result["resource_evidence"]["observed_parts"]["P"]["part_location"] == "Rack"
    # The release's declared effect leaves the robot at the rack, a value the event has to state.
    left_out = complete({})
    assert left_out == {"feasibility_status": "INFEASIBLE",
                        "reason": "The program ends at Rack and the event states no resource_location"}
    # A stated location the place does not end at takes where it ends, so the postcondition refuses it.
    assert complete({"resource_location": "home"})["projected_snapshot"]["resource_location"] == "Rack"

@pytest.mark.parametrize("stated", [True, False])
def test_an_insert_takes_a_stated_record_from_its_runtime_model_and_adds_none_left_out(monkeypatch, stated):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    retreat_step = {"op": "move_relative", "params": {"dx": 0., "dy": 0., "dz": {"$sub": [
        {"$state": "_task_ctx", "path": ["travel_z"]}, {"$state": "_position", "path": ["z"]}]}}}
    place_insert = {"event_name": "place_insert", "updates": {"part_location": {"set_from_param": "destination_location"}},
                    "composition_contract": {"steps": [{"op": "release_part", "params": {}}, retreat_step]}}
    # As in a robot's recovery model, nothing here declares the process record.
    model = {"events": [place_insert], "state_variables": {"part_location": {"scope": "part", "domain": ["Board", "robot"]}}}
    down = {"qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    place, away = {"x": 0.2, "y": 0.5, "z": 1.1, **down}, {"x": 0.2, "y": 0.5, "z": 1.3, **down}
    slot = {"slot_x": 0.2, "slot_y": 0.5, "part_height": 0.02, "board_top_z": 1.0}
    target = {**slot, "target_pose": place, "retreat_pose": away, "destination_location": "Board", "target": "Plate/P",
              "place_part_origin_z": 1.01, "target_reference": {"surface_role": "assembly_slot"},
              "target_origin_pose": {"x": 0.2, "y": 0.5, "z": 1.01}}
    stated_slot = {**slot, "destination_location": "Board", "part_origin_z": 1.01}
    trace = [{"step_index": 0, "primitive": "compute_place_targets", "params": {}, "resolved_params": {"part_name": "P"},
              "preview_output": target, "start_snapshot": {}},
             # The release check found the stated slot equal to the recorded one; the part is set there.
             _released({"step_index": 1, "primitive": "release_part", "params": {},
                         "resolved_params": {"part_name": "P", "assembly_slot": stated_slot},
                         "start_snapshot": {"current_pose": place}},
                        pose=[0.2, 0.5, 1.01, 0., 0., 0., 1.], place="Board",
                        assembly_slot={**stated_slot, "target": "Plate/P"}),
             _moved({"step_index": 2, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
                     "start_snapshot": {"current_pose": place}}, away)]
    runtime = {"events": [{"parameter_bindings": {"destination_location": {"equals": "Board"}}, "product_effects": {
        "processCompleted": [{"process": "assembly", "target": {"set_from_param": "target"}}]}}]}
    printed, assembled = {"process": "print_part"}, {"process": "assembly", "target": "Plate/P"}
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "insert", "event_name": "insert", "resource_jid": "robot", "part_name": "P",
              "expected_end_state": {"processCompleted": [printed, assembled]} if stated else {}}, trace=trace,
        recovery_des_model=model, recovery_snapshot={}, projected_snapshot={"current_pose": away},
        part_context={"part_location": "robot", "processCompleted": [printed]}, grounding_context={}, primitive_catalog=[],
        resource_evidence={}, physical_evidence={"jids": {"ur5e-3": "robot"}, "environment_models": {"ur5e-3": runtime}})
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    assert result["projected_part_context"]["processCompleted"] == ([printed, assembled] if stated else [printed])
    # The part rests where the controller sets it in its slot.
    assert result["projected_part_context"]["observed_pose"] == dict(zip(_AXES, [0.2, 0.5, 1.01, 0., 0., 0., 1.]))
    assert result["projected_part_context"]["part_location"] == "Board"


def test_a_part_s_record_is_compared_when_the_acting_resource_s_model_does_not_list_it():
    start = {"name": "start_job", "params": {"job_id": {"type": "string"}}, "required_params": ["job_id"],
             "preconditions": {"resource_state": {"equals": "loaded"}}, "effects": {"resource_state": {"set": "running"}}}
    declarations = {"resource_state": {"scope": "resource"}, "part_name": {"scope": "resource"}}
    done = [{"process": "print_part"}]
    task = {"outline_id": "start", "event_name": "start", "resource_jid": "owner", "part_name": "P",
            "expected_start_state": {"resource_state": "loaded", "processCompleted": done},
            "expected_end_state": {"resource_state": "running", "processCompleted": done}}

    def validate(record):
        return validate_recovery_primitive_composition(
            task=task, primitive_steps=[{"primitive": "start_job", "params": {"job_id": "job-2"}}],
            recovery_des_model={"state_variables": declarations},
            recovery_snapshot={"resource_state": "loaded", "part_name": "P"}, part_context={"processCompleted": record},
            grounding_context={}, primitive_catalog=[start], validate_primitive=feasible)

    assert validate(done)["feasibility_status"] == "FEASIBLE"
    assert validate([])["feasibility_status"] != "FEASIBLE"


def test_a_supported_placement_is_received_by_the_slot_whose_custody_locates_the_part():
    from cais_spade_llm.resources.robot.recovery_composition import _receiving_slot

    part = {"reference": "part_name", "scope": "resource"}
    physical = {"product_name": "board", "geometry": {"regions": _RACK_ZONES}, "environment_models": {
        "Rack": {"state_variables": {"zone_1_part": part, "zone_2_part": part},
                 "assignments": {"zones": [{"zone": 1, "pose": [0.0, 0.5, 1.0]}, {"zone": 2, "pose": [0.2, 0.5, 1.0]}]}},
        "M9": {"state_variables": {"part_name": part, "staging_part": part}},
        "arm": {"state_variables": {"held_part": part}}}}
    # The zone whose recorded area holds the part, or another resource's slot for a place it owns.
    assert _receiving_slot("P", "Rack", {"x": 0.2, "y": 0.5}, physical) == ("Rack", "zone_2_part")
    # Anywhere in the zone's footprint, not only at its recorded pose; its height bounds nothing that rests there.
    assert _receiving_slot("P", "Rack", {"x": 0.23, "y": 0.44, "z": 1.2}, physical) == ("Rack", "zone_2_part")
    assert _receiving_slot("P", "M9 staging tray", {"x": 3.0, "y": 1.0}, physical) == ("M9", "staging_part")
    assert _receiving_slot("P", "Rack", {"x": 0.1, "y": 0.5}, physical) is None
    assert _receiving_slot("P", "Rack", {"x": 9.0, "y": 9.0}, physical) is None
    assert _receiving_slot("P", "Shelf", {"x": 0.0, "y": 0.0}, physical) is None




def test_a_pick_empties_the_stated_values_that_record_the_part_as_the_ra_finds_them():
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
    stated = {"m9": {"staging_part": None}, "belt": {"part_location.P": None, "part_order.P": None},
              "store": {"inventory.P": False}}
    records = _source_records("P", "arm", {"resources": {}}, physical, stated)
    # A staging slot, a belt slot with its order (0 still records the part), and a storage flag (false: no null).
    assert sorted(records) == [("belt", "part_location.P", None), ("belt", "part_order.P", None),
                               ("m9", "staging_part", None), ("store", "inventory.P", False)]
    # Code writes no value the event leaves out, so the ones it does not state are not returned.
    assert _source_records("P", "arm", {"resources": {}}, physical, {"m9": {"staging_part": None}}) == [
        ("m9", "staging_part", None)]


def test_a_destination_receives_into_its_own_slot():
    from cais_spade_llm.resources.robot.recovery_composition import _receiving_slot

    part = {"reference": "part_name", "scope": "resource"}
    physical = {"product_name": "board", "environment_models": {"M9": {"state_variables": {
        "part_name": part, "staging_part": part}}}}
    # A machine's workholding slot locates the part at the machine; its staging slot at its staging tray.
    assert _receiving_slot("P", "M9", {"x": 0.0, "y": 0.0}, physical) == ("M9", "part_name")
    assert _receiving_slot("P", "M9 staging tray", {"x": 0.0, "y": 0.0}, physical) == ("M9", "staging_part")


@pytest.mark.parametrize("stated, declared", [("loaded", False), ("paused", True)])
def test_a_neighbour_state_neither_its_primitives_nor_a_received_part_make_is_refused(stated, declared):
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
    assert result["feasibility_status"] == "INFEASIBLE"
    assert f"m9's resource_state: achieved 'idle', required '{stated}'" in result["reason"]


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


@pytest.mark.parametrize("after, end, expected", [
    ("move_cartesian", {"x": 2.0, "y": -1.0, "z": 1.3}, "M1"), ("move_cartesian", {"x": 0.5, "y": 0.5, "z": 1.3}, "Rack"),
    ("move_to_named_pose", {"x": 0.5, "y": 0.5, "z": 1.3}, "home")])
def test_a_robot_is_at_the_last_place_it_reached(monkeypatch, after, end, expected):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    pick_grasp = {"event_name": "pick_grasp", "updates": {
        "resource_location": {"set_from_param": "origin_resource_location"}, "part_location": {"set": "robot"}}}
    home = {"event_name": "move_home", "updates": {"resource_location": {"set": "home"}, "resource_state": {"set": "idle"}},
            "composition_contract": {"steps": [{"op": "move_to_named_pose", "params": {"pose_name": "home"}}]}}
    model = {"events": [pick_grasp, home], "state_variables": {
        "part_location": {"scope": "part", "domain": ["Rack", "robot"]},
        "resource_location": {"scope": "resource", "domain": ["Rack", "M1", "home"]}}}
    down = {"qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    grasp, lift, away = {"x": 0.0, "y": 0.5, "z": 1.1, **down}, {"x": 0.0, "y": 0.5, "z": 1.3, **down}, {**end, **down}
    trace = [{"step_index": 0, "primitive": "grasp_part", "params": {}, "resolved_params": {"part_name": "P"},
              "start_snapshot": {"current_pose": grasp}, "projected_snapshot": {"current_pose": grasp}},
             _moved({"step_index": 1, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
                     "start_snapshot": {"current_pose": grasp}}, lift),
             _moved({"step_index": 2, "primitive": after, "params": {}, "resolved_params": {"pose_name": "home"},
                     "start_snapshot": {"current_pose": lift}}, away)]
    seen = []

    def locate(pose, stated=None):
        seen.append(stated)
        # Only M1's recorded area holds a gripper: far along x. A pose over the rack's area at travel height is in none.
        return "M1" if pose["x"] > 1.0 else None

    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "take", "event_name": "take", "resource_jid": "robot", "part_name": "P",
              "expected_end_state": {"resource_location": expected}}, trace=trace,
        recovery_des_model=model, recovery_snapshot={"resource_location": "Rack"},
        projected_snapshot={"resource_location": "Rack", "current_pose": away}, part_context={"part_location": "Rack"},
        grounding_context={}, primitive_catalog=[], resource_evidence=_GRIP_EVIDENCE, physical_evidence=_PART_GEOMETRY,
        locate_pose=locate)
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    # The grasp's declared effect puts the robot at the rack; a motion ending in no recorded area leaves it there; a
    # motion ending in M1's area, or a named pose with a declared place, moves it on (D2).
    assert result["projected_snapshot"]["resource_location"] == expected
    # Each motion other than a named pose is located, preferring the place the event states.
    assert seen == ([expected, expected] if after == "move_cartesian" else [expected])


# A stated other place takes where the program ends too, so the postcondition refuses it.
@pytest.mark.parametrize("end, status", [({}, "INFEASIBLE"), ({"resource_location": "home"}, "FEASIBLE"),
                                         ({"resource_location": "Bench"}, "FEASIBLE")])
def test_a_program_that_ends_at_a_named_pose_needs_the_event_to_state_it(monkeypatch, end, status):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    move_home = {"event_name": "move_home", "updates": {"resource_location": {"set": "home"}, "resource_state": {"set": "idle"}},
                 "composition_contract": {"steps": [{"op": "move_to_named_pose", "params": {"pose_name": "home"}}]}}
    model = {"events": [move_home], "state_variables": {"resource_location": {"scope": "resource", "domain": ["Bench", "home"]},
                                                        "resource_state": {"scope": "resource", "domain": ["idle", "busy"]}}}
    pose = {"x": 0.0, "y": 0.5, "z": 1.3, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    trace = [{"step_index": 0, "primitive": "move_to_named_pose", "params": {}, "resolved_params": {"pose_name": "home"},
              "start_snapshot": {"current_pose": pose}, "projected_snapshot": {"current_pose": pose}}]
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "home", "event_name": "home", "resource_jid": "robot", "expected_end_state": end}, trace=trace,
        recovery_des_model=model, recovery_snapshot={"resource_location": "Bench"},
        projected_snapshot={"resource_location": "Bench", "current_pose": pose}, part_context={}, grounding_context={},
        primitive_catalog=[], resource_evidence={}, physical_evidence={})
    assert result["feasibility_status"] == status, result.get("reason")
    if status == "FEASIBLE":
        assert result["projected_snapshot"]["resource_location"] == "home"
    else:
        assert result["reason"] == "The program ends at home and the event states no resource_location"


@pytest.mark.parametrize("end, status", [({}, "INFEASIBLE"), ({"resource_location": "Rack"}, "FEASIBLE")])
def test_a_program_that_moves_the_robot_needs_the_event_to_state_where(monkeypatch, end, status):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    model = {"events": [], "state_variables": {"resource_location": {"scope": "resource", "domain": ["Bench", "Rack"]}}}
    pose = {"x": 0.0, "y": 0.5, "z": 1.3, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    trace = [_moved({"step_index": 0, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
                     "start_snapshot": {"current_pose": pose}}, pose)]
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "move", "event_name": "move", "resource_jid": "robot", "expected_end_state": end}, trace=trace,
        recovery_des_model=model, recovery_snapshot={"resource_location": "Bench"},
        projected_snapshot={"resource_location": "Bench", "current_pose": pose}, part_context={}, grounding_context={},
        primitive_catalog=[], resource_evidence={}, physical_evidence={}, locate_pose=lambda _, stated=None: stated or "Rack")
    assert result["feasibility_status"] == status, result.get("reason")
    if status == "INFEASIBLE":
        assert result["reason"] == "The program ends at Rack and the event states no resource_location"


@pytest.mark.parametrize("part_at, end, status, expected", [
    # The grasp's declared effect leaves the robot at the part's place, or at none for a part at none, and an event
    # that leaves resource_location out is refused for it (D2).
    ("Rack", {}, "INFEASIBLE", "The program ends at Rack and the event states no resource_location"),
    (None, {}, "INFEASIBLE", "The program ends at no named place and the event states no resource_location"),
    # A place the location does not declare says nothing about the robot.
    ("other@localhost", {}, "FEASIBLE", "Bench"), ("other", {}, "FEASIBLE", "Bench"),
    ("Rack tray", {}, "FEASIBLE", "Bench"), (None, {"resource_location": None}, "FEASIBLE", None),
    ("Rack tray", {"resource_location": "Bench"}, "FEASIBLE", "Bench"), ("Rack", {"resource_location": "Rack"}, "FEASIBLE", "Rack"),
    # A stated location the pick does not end at takes where it ends, so the postcondition refuses it.
    (None, {"resource_location": "Bench"}, "FEASIBLE", None), ("Rack", {"resource_location": "Bench"}, "FEASIBLE", "Rack")])
def test_a_pick_leaves_the_robot_where_the_event_states(monkeypatch, part_at, end, status, expected):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    pick_grasp = {"event_name": "pick_grasp", "updates": {
        "resource_location": {"set_from_param": "origin_resource_location"}, "part_location": {"set": "robot"}}}
    model = {"events": [pick_grasp], "state_variables": {
        "part_location": {"scope": "part", "domain": ["Rack", "Rack tray", "other", "other@localhost", "robot"]},
        "resource_location": {"scope": "resource", "domain": ["Rack", "Bench"]}}}
    down = {"qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    grasp, lift = {"x": 0.0, "y": 0.5, "z": 1.1, **down}, {"x": 0.0, "y": 0.5, "z": 1.3, **down}
    trace = [{"step_index": 0, "primitive": "grasp_part", "params": {}, "resolved_params": {"part_name": "P"},
              "start_snapshot": {"current_pose": grasp}, "projected_snapshot": {"current_pose": grasp}},
             _moved({"step_index": 1, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
                     "start_snapshot": {"current_pose": grasp}}, lift)]
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "take", "event_name": "take", "resource_jid": "robot", "part_name": "P",
              "expected_end_state": end}, trace=trace,
        recovery_des_model=model, recovery_snapshot={"resource_location": "Bench"},
        projected_snapshot={"resource_location": "Bench", "current_pose": lift}, part_context={"part_location": part_at},
        grounding_context={"resources": {"other@localhost": {}}}, primitive_catalog=[], resource_evidence=_GRIP_EVIDENCE,
        # The lift ends in no recorded area, so it leaves the robot where the grasp left it.
        physical_evidence=_PART_GEOMETRY, locate_pose=lambda pose, stated=None: None)
    assert result["feasibility_status"] == status, result.get("reason")
    if status == "FEASIBLE":
        assert result["projected_snapshot"]["resource_location"] == expected
    else:
        assert result["reason"] == expected


def test_a_restated_location_the_robot_keeps_holds_without_locating_its_pose(monkeypatch):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    above = {"x": 0.0, "y": 0.5, "z": 1.4, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    trace = [_moved({"step_index": 0, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
                     "start_snapshot": {}}, above)]
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "hover", "event_name": "hover", "resource_jid": "robot", "part_name": None,
              "expected_end_state": {"resource_location": "Rack"}}, trace=trace,
        recovery_des_model={"events": [], "state_variables": {}}, recovery_snapshot={"resource_location": "Rack"},
        projected_snapshot={"resource_location": "Rack", "current_pose": above}, part_context={},
        grounding_context={}, primitive_catalog=[], resource_evidence={}, physical_evidence={},
        locate_pose=lambda pose, stated=None: None)
    # A motion that ends in no recorded area leaves the robot at the last place it reached (D2).
    assert result["projected_snapshot"]["resource_location"] == "Rack"

_TOOL_DOWN = {"qx": 0.0, "qy": 1.0, "qz": 0.0, "qw": 0.0}
_RECORDED_CELL = {
    "diagnostic_regions": {
        "M1.staging": {"bounds": [[-4.71, -4.49], [0.99, 1.21], [1.0, 1.12]],
                       "source": {"surface_collision": "M1 staging tray::link::surface"}},
        "M1.workholding": {"bounds": [[-4.25, -4.01], [1.7, 1.94], [0.94, 1.18]], "source": {}},
        "Conveyor.loading_position_1": {"bounds": [[-4.2, -3.9], [0.35, 0.65], [1.015, 1.065]], "source": {}}},
    "physical_evidence": {"jids": {"ur5e-1": "robot"}, "resources": {"ur5e-1": {"tcp_offset_z": -0.218}}},
    "recovery_resources": {"robot": {"static_capabilities": {"reachability": ["M1", "M1 staging tray", "Conveyor"]}}}}


@pytest.mark.parametrize("wrist, stated, place", [
    # The gripper is over the tray: one pose is at one place, whatever the event states.
    ((-4.6, 1.1, 1.30), "M1 staging tray", "M1 staging tray"),
    ((-4.6, 1.1, 1.30), None, "M1 staging tray"),
    ((-4.6, 1.1, 1.30), "M1", "M1 staging tray"),
    # The wrist is inside the tray's area with the fingertips through the tray, which is no place to be.
    ((-4.6, 1.1, 1.10), "M1 staging tray", None),
    ((-4.6, 1.1, 1.10), "M1", None),
    ((-4.6, 1.1, 1.10), None, None),
    # The same through the belt, where the name's prefix is the place.
    ((-4.05, 0.5, 1.05), "Conveyor", None),
    ((-4.05, 0.5, 1.26), "Conveyor", "Conveyor"),
    # The gripper above an area's top is out of it.
    ((-4.6, 1.1, 1.35), "M1 staging tray", None),
    ((-4.13, 1.82, 1.30), "M1", "M1"),
    ((-4.13, 1.82, 1.30), None, "M1"),
])
def test_a_robot_is_at_the_place_whose_own_area_holds_its_gripper(wrist, stated, place):
    from cais_spade_llm.resources.recovery_composition_owner import _locate

    pose = {**dict(zip("xyz", wrist, strict=True)), **_TOOL_DOWN}
    assert _locate(pose, _RECORDED_CELL, "robot", stated) == place


@pytest.mark.parametrize("pose, jid", [
    # Read at the wrist these are inside the tray's area, so only a gripper point with no default leaves them out.
    ({"x": -4.6, "y": 1.1, "z": 1.10}, "robot"),
    ({"x": -4.6, "y": 1.1, "z": 1.10, **_TOOL_DOWN}, "other"),
])
def test_a_robot_with_no_recorded_offset_or_orientation_is_at_no_place(pose, jid):
    from cais_spade_llm.resources.recovery_composition_owner import _locate

    assert _locate(pose, _RECORDED_CELL, jid, "M1 staging tray") is None


def _stated_pose_case(stated_pose, steps, monkeypatch, facts=None):
    from cais_spade_llm.resources.recovery_composition_owner import validate_owner_composition
    from cais_spade_llm.resources.robot import recovery_composition as robot_composition

    axes = ("x", "y", "z", "qx", "qy", "qz", "qw")
    catalog = [{"name": "move_cartesian", "params": {axis: {"type": "number"} for axis in axes},
                "required_params": ["x", "y", "z"], "preconditions": {},
                "effects": {"current_pose": {"pose_absolute_from_params": ["x", "y", "z"]}}}]
    declarations = {"resource_location": {"scope": "resource"}, "resource_state": {"scope": "resource"},
                    "pose": {"scope": "resource", "type": "object", "fields": ["x", "y", "z"], "effect": "current_pose"}}
    prepared = {**_RECORDED_CELL, "recovery_resources": {"robot": {
        **_RECORDED_CELL["recovery_resources"]["robot"], "resource_type": "robot", "execution_primitive_catalog": catalog,
        "recovery_des_model": {"state_variables": declarations, "events": []}}}}
    task = {"outline_id": "leave", "event_name": "leave", "resource_jid": "robot", "part_name": None,
            "expected_start_state": {"resource_location": "M1"},
            "expected_end_state": {"resource_location": "M1 staging tray", "pose": stated_pose}}
    start = {"resource_location": "M1", "held_part": None, "current_pose": {"x": -4.13, "y": 1.82, "z": 1.35, **_TOOL_DOWN}}

    def check(**_):
        return {"allowed": True, "feasibility_status": "FEASIBLE", "evidence": {}}

    def reached(evidence, scene, *, index, primitive, params, recovery_snapshot, targets=None):
        # MoveIt is not what these cases test: the motion reaches the pose its parameters state.
        end = [params[axis] for axis in "xyz"] + [params.get(axis, start["current_pose"][axis]) for axis in _TOOL_DOWN]
        return {"composed_motion": {"end_joints": [], "end_pose": end}}

    monkeypatch.setattr(robot_composition, "_motion_check", reached)
    return validate_owner_composition(SimpleNamespace(jid="robot", check_recovery_primitive_feasibility=check),
        task=task, primitive_steps=steps, prepared=prepared, recovery_snapshot=start, part_context={},
        grounding_context={"outline_event": task, "resources": {"robot": dict(start)}, **(facts or {})})


_STATED_POSE_STEP = {"primitive": "move_cartesian", "params": {
    **{axis: {"context_ref": f"/outline_event/expected_end_state/pose/{axis}"} for axis in "xyz"},
    **{axis: {"context_ref": f"/resources/robot/current_pose/{axis}"} for axis in _TOOL_DOWN}}}


def test_a_program_binds_the_stated_pose_and_the_robot_ends_at_the_stated_place(monkeypatch):
    result = _stated_pose_case({"x": -4.6, "y": 1.1, "z": 1.30}, [_STATED_POSE_STEP], monkeypatch)
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    assert result["projected_snapshot"]["resource_location"] == "M1 staging tray"
    assert result["trace"][0]["resolved_params"] == {"x": -4.6, "y": 1.1, "z": 1.30, **_TOOL_DOWN}


def test_a_program_that_ends_short_of_the_stated_pose_is_refused(monkeypatch):
    elsewhere = {"primitive": "move_cartesian", "params": {axis: {"context_ref": f"/elsewhere/{axis}"} for axis in "xyz"}}
    result = _stated_pose_case({"x": -4.6, "y": 1.1, "z": 1.30}, [elsewhere], monkeypatch,
                               facts={"elsewhere": {"x": -4.6, "y": 1.1, "z": 1.28}})
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reason"].startswith("Postcondition 'pose': achieved")


def test_a_stated_pose_whose_gripper_is_out_of_the_stated_place_does_not_end_there(monkeypatch):
    result = _stated_pose_case({"x": -4.6, "y": 1.1, "z": 1.35}, [_STATED_POSE_STEP], monkeypatch)
    assert result["feasibility_status"] == "INFEASIBLE"
    # Its gripper ends above the tray's area, in no recorded area, so the robot is still where it last was (D2).
    assert result["reason"] == "Postcondition 'resource_location': achieved 'M1', required 'M1 staging tray'"


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
        task=task, primitive_steps=[{"primitive": "move_cartesian", "params": {
            axis: {"context_ref": f"/outline_event/expected_end_state/pose/{axis}"} for axis in "xyz"}}],
        recovery_des_model={"state_variables": declarations},
        recovery_snapshot={"current_pose": {"x": 0.3, "y": 0.0, "z": 1.3}}, part_context={},
        grounding_context={"outline_event": task},
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


_PAUSED_ROUND_PART = {"process": "trim", "result": "circle", "status": "paused", "machine": "M2"}


def _job_access_case(primitive, params, snapshot, robot_z, part_process=_PAUSED_ROUND_PART):
    """Run one job step through the owner, with the other resources' state in ``grounding_context``."""
    import json

    from cais_spade_llm.recovery_framework import SCENE_PATH
    from cais_spade_llm.resources.machine.machine_jobs import MachineJobs, machine_primitive_contracts
    from cais_spade_llm.resources.recovery_composition_owner import validate_owner_composition

    scene = json.loads(SCENE_PATH.read_text())
    machine = scene["machines"][1]
    part = "RGOCG4-50_Round_4mm"
    x, y, z = machine["workholding_pose"][:3]
    jobs = MachineJobs(machine, "run")
    jobs.command("start_job", {"job_id": "circle-job", "task_id": "circle-job", "part_name": part, "process": "trim",
                               "result": "circle"}, command_id="circle-job", now=100,
                 observations={"part_name": part, "correct_part_present": True, "robot_access_clear": True})
    jobs.command("pause_job", {"job_id": "circle-job"}, command_id="pause", now=101,
                 observations={"part_name": part, "correct_part_present": True, "robot_access_clear": True})
    scene["KMR"]["task_execution"]["clearance_link_bounds"] = {"tool": [{"center": [0.0, 0.0, 0.0], "radius": 0.01}]}
    chain = {"root": "base", "reference_link": "tool", "root_xyz": [0.0, 0.0, 0.0], "root_quaternion": [0.0, 0.0, 0.0, 1.0],
             "joint_names": ["a1"], "joints": [{"name": "a1", "parent": "base", "child": "tool", "type": "revolute",
                                                "axis": [0.0, 0.0, 1.0], "xyz": [0.0, 0.0, 0.5], "rpy": [0.0, 0.0, 0.0]}]}
    kmr = {"base_pose": [x + 2.0, y, 0.0, 0.0, 0.0, 0.0, 1.0], "joint_positions": [0.0],
           "current_pose": [x + 2.0, y, 0.5, 0.0, 0.0, 0.0, 1.0]}
    contracts = machine_primitive_contracts(machine)
    catalog = [{"name": primitive, "params": {name: {} for name in params}, "required_params": list(params),
                **contracts[primitive][0]}]
    evidence = {"resource_id": "M2", "inputs": {"scene": scene}, "processing_observations": jobs.snapshot()}
    prepared = {"physical_evidence": {"jids": {"ur5e-2": "robot", "KMR": "kmr"},
                                      "resource_models": {"KMR": {"configuration": chain}}},
                "recovery_resources": {"owner": {"resource_type": "resource", "execution_primitive_catalog": catalog,
                                                 "resource_evidence": evidence, "recovery_des_model": {
                                                     "state_variables": {**_JOB_DECLARATIONS,
                                                                         "selected_program": {"scope": "resource"}}}}}}
    seen = []

    def check(*, physical_evidence, **_):
        seen.append(physical_evidence.get("machine_access_observations", {}).get("M2"))
        return {"allowed": True, "feasibility_status": "FEASIBLE", "evidence": {}}

    process = {"process": "trim", "result": "circle", "status": "running", "machine": "M2"}
    task = {"outline_id": "job", "event_name": "job", "resource_jid": "owner", "part_name": part,
            "expected_start_state": {"resource_state": snapshot["resource_state"]},
            "expected_end_state": {"resource_state": "running", "current_process": process}}
    context = {"parts": {part: {"observed_pose": {"x": x, "y": y, "z": z}}},
               "resources": {"robot": {"held_part": None, "current_pose": {"x": x, "y": y, "z": robot_z}},
                             "kmr": {"held_part": None, "resource_evidence": {"snapshot": kmr}}}}
    result = validate_owner_composition(SimpleNamespace(jid="owner", check_recovery_primitive_feasibility=check),
        task=task, primitive_steps=[{"primitive": primitive, "params": params}], prepared=prepared,
        recovery_snapshot=snapshot, part_context={"part_name": part, "current_process": part_process},
        grounding_context=context)
    return result, seen


@pytest.mark.parametrize("primitive, params, state, part_process", [
    ("resume_job", {"job_id": "circle-job"}, "paused", _PAUSED_ROUND_PART),
    ("start_job", {"part_name": "RGOCG4-50_Round_4mm", "process": "trim", "result": "circle"}, "loaded", None),
])
def test_a_job_step_reads_the_access_worked_out_from_the_composition_state(primitive, params, state, part_process):
    snapshot = {"resource_state": state, "part_name": "RGOCG4-50_Round_4mm", "selected_program": "circle"}
    result, seen = _job_access_case(primitive, params, snapshot, robot_z=1.34, part_process=part_process)
    assert result["feasibility_status"] == "FEASIBLE", result["reason"]
    # A paused job's stored observation is not reused: access is worked out for this step.
    assert seen and seen[0]["source"]["region"] == "M2.workholding"
    assert seen[0]["part_name"] == "RGOCG4-50_Round_4mm"


def test_an_unknown_job_is_left_to_the_machine_which_names_it():
    result, seen = _job_access_case("resume_job", {"job_id": "no-such-job"},
                                    {"resource_state": "paused", "part_name": "RGOCG4-50_Round_4mm",
                                     "selected_program": "circle"}, robot_z=1.34)
    assert seen == [None]
    assert "the position of" not in str(result["reason"])


def test_a_refused_access_never_reaches_the_machines_own_check():
    result, seen = _job_access_case("resume_job", {"job_id": "circle-job"},
                                    {"resource_state": "paused", "part_name": "RGOCG4-50_Round_4mm",
                                     "selected_program": "circle"}, robot_z=1.1)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert "has not left the declared region M2.workholding" in result["reason"]
    assert seen == []


def test_machine_commands_leave_the_job_record_their_later_steps_and_events_read():
    machine, result, seen = _machine_job_case(
        [{"primitive": "pause_job", "params": {"job_id": "circle-job"}},
         {"primitive": "select_program", "params": {"program": "square"}}],
        {"resource_state": "running", "part_name": "RGOCG4-50_Round_4mm", "selected_program": "circle"},
        {"current_process": {"process": "trim", "result": "circle", "status": "running", "machine": "M2"},
         "processCompleted": []},
        # The pause changes the part's job record, so the event states it, as the RA requires.
        {"resource_state": "paused", "selected_program": "square",
         "current_process": {"process": "trim", "result": "circle", "status": "paused", "machine": "M2"}})
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


@pytest.mark.parametrize("named", ["standby", "idle"])
def test_a_resource_state_comes_only_from_the_primitives_in_the_program(named):
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
    # No primitive declares standby, and idle comes only from move_to_named_pose, which the program does not run.
    assert result["feasibility_status"] == "INFEASIBLE"
    assert f"achieved 'failed', required '{named}'" in result["reason"]

def test_a_program_that_changes_a_value_the_event_leaves_out_is_refused():
    # As the RA's left_by: the home move sets idle, so an event that leaves resource_state out would leave the
    # outline at placed while the program ends idle.
    declarations = {"resource_state": {"scope": "resource"}, "gripper_state": {"scope": "resource"}}
    catalog = [{"name": "open_gripper", "params": {}, "required_params": [], "effects": {"gripper_state": {"set": "open"}}},
               {"name": "move_to_named_pose", "params": {}, "required_params": [],
                "effects": {"current_state": {"set": "idle"}}}]

    def compose(steps, end):
        task = {"outline_id": "home", "event_name": "home", "resource_jid": "owner", "part_name": None,
                "expected_start_state": {"resource_state": "placed"}, "expected_end_state": end}
        return validate_recovery_primitive_composition(
            task=task, primitive_steps=steps, recovery_des_model={"state_variables": declarations},
            recovery_snapshot={"resource_state": "placed", "gripper_state": "closed"}, part_context={},
            grounding_context={}, primitive_catalog=catalog, validate_primitive=feasible)

    both = [{"primitive": "open_gripper", "params": {}}, {"primitive": "move_to_named_pose", "params": {}}]
    refused = compose(both, {"gripper_state": "open"})
    assert refused["feasibility_status"] == "INFEASIBLE"
    assert refused["reason"] == "The program changes resource_state from 'placed' to 'idle', and the event does not state it"
    assert compose(both, {"gripper_state": "open", "resource_state": "idle"})["feasibility_status"] == "FEASIBLE"
    assert compose(both[:1], {"gripper_state": "open"})["feasibility_status"] == "FEASIBLE"


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
            task=task, primitive_steps=[{"primitive": "move_cartesian", "params": {
                axis: {"context_ref": f"/target/{axis}"} for axis in target}}],
            recovery_des_model={"state_variables": declarations}, recovery_snapshot={"current_pose": dict(start)},
            part_context={}, grounding_context={"target": target}, primitive_catalog=[move], validate_primitive=feasible)

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
        return {"primitive_events": [row], "turns": [], "feedback": [{"reason": "wrong order"}],
                "session_state": {"primitive_authoring_memo": [{"outline_id": "restore"}]}}
    monkeypatch.setattr(multi_turn, "_run_resource_primitive_batch", reject)
    decision, _, _ = asyncio.run(generate_next_composition_event(planner=SimpleNamespace(), session_state=session,
        prepared_recovery_request=prepared))
    assert decision == "primitive_blocked"
    assert session["accepted_primitive_program"] == []
    assert session["primitive_rejection_feedback"][0]["reason"] == "wrong order"
    assert "primitive_authoring_memo" not in session


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

    formats = []

    async def recorded(fn, **kwargs):
        formats.append(kwargs["response_format"])
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
    # Composition's response format offers no block: its checks or its turn limits end it.
    assert all("primitive_blocked" not in row["schema"]["properties"]["decision"]["enum"] for row in formats)


def test_the_robot_guidance_says_where_a_location_comes_from_and_gives_no_recipe(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_primitive_generation as generation
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_prompts as prompts

    calls = []

    async def ask(prompt, **kwargs):
        calls.append(prompt)
        return {"decision": "primitive_steps_ready"}

    async def recorded(fn, **kwargs):
        return await fn(kwargs["prompt"])

    async def reject(**kwargs):
        kwargs["session_state"]["primitive_rejection_feedback"] = []
        return "primitive_blocked", {}

    monkeypatch.setattr(prompts, "build_multi_turn_phase_prompt_input", lambda **kwargs: {})
    monkeypatch.setattr(prompts, "render_multi_turn_phase_prompt", lambda value: "author")
    monkeypatch.setattr(generation, "build_primitive_generation_prompt_context", lambda **kwargs: {})
    monkeypatch.setattr(generation, "recorded_structured_call", recorded)
    monkeypatch.setattr(generation, "_handle_primitive_generation_phase", reject)
    monkeypatch.setattr(generation, "_write_primitive_subturn_artifact", lambda **kwargs: {})
    asyncio.run(generation.generate_primitive_batch_with_llm_agent(
        llm_agent=SimpleNamespace(ask_llm_structured=ask),
        prepared_recovery_request={"composition_validation_version": 1, "recovery_resources": {"owner": {"resource_type": "robot"}}},
        assigned_outline_events=[{"outline_id": "leave", "resource_jid": "owner"}]))

    # D2: the robot is at the last place it reached; the program's end alone does not decide it.
    assert ("A robot's resource_location is the last place it reached: a transfer's declared effect, a named pose's "
            "declared place, or the place whose recorded area holds the gripper where a motion ends; a motion that ends "
            "in no recorded area leaves it as it was.") in calls[0]
    assert "is read from where the program ends" not in calls[0]
    # D1: the contract fact, and where the numbers of the nominal steps are in the served context.
    assert "Every number in params is such a reference, and a number written into a program is refused" in calls[0]
    assert "/resources/<jid>/recovery_des_model/events/<index>/composition_contract/steps/<index>/params" in calls[0]
    assert "bind" not in calls[0].split("RA composition validation:")[1].split("Use exact primitive identities")[1].split(
        "Missing facts require")[1]


@pytest.mark.parametrize("reached, expected", [("Bench", "Bench"), (None, "home")])
def test_move_home_completion_cannot_override_a_later_motion(reached, expected):
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    model = {"events": [{"event_name": "move_home", "composition_contract": {
        "steps": [{"op": "move_to_named_pose", "params": {"pose_name": "home"}}]},
        "updates": {"resource_location": {"set": "home"}}}], "state_variables": {}}
    trace = [{"primitive": "move_to_named_pose", "params": {"pose_name": "home"},
              "resolved_params": {"pose_name": "home"}, "step_index": 0},
             _moved({"primitive": "move_cartesian", "params": {"x": 2., "y": 0., "z": 1.},
                     "resolved_params": {"x": 2., "y": 0., "z": 1.}, "step_index": 1},
                    {"x": 2., "y": 0., "z": 1., "qx": 0., "qy": 1., "qz": 0., "qw": 0.})]
    result = project_composition_completion(task={"resource_jid": "robot", "expected_end_state": {
        "resource_location": expected}}, trace=trace, recovery_des_model=model, recovery_snapshot={},
        projected_snapshot={"resource_location": "elsewhere"}, part_context={}, grounding_context={},
        primitive_catalog=[], resource_evidence={}, physical_evidence={},
        locate_pose=lambda pose, stated=None: reached)
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    # The named pose's declared place holds until a later motion ends in a recorded area of another place.
    assert result["projected_snapshot"]["resource_location"] == expected
    assert result["completed_contracts"][-1]["step_index"] == (1 if reached else 0)


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


@pytest.mark.parametrize("travel_z, retreat_z", [(1.2855, 1.3455), (1.5, 1.5)])
def test_a_place_retreat_rises_when_the_place_sits_above_the_picks_travel_height(monkeypatch, travel_z, retreat_z):
    from cais_spade_llm.resources.robot import recovery_capability
    from cais_spade_llm.resources.robot.recovery_composition import preview_composition_output

    insert = {"x": 0., "y": 0., "z": 1.2955, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **kwargs: {
        "status": "calculated", "destination_location": "M2", "insert_pose": insert, "target_reference": {"surface_role": "support"}})
    evidence = {"state": {"_task_ctx": {"part_name": "P", "travel_z": travel_z}},
                "controller_config": {"parts_tuning": {"insertion_depth_m": .0025}}}
    result = preview_composition_output(primitive="compute_place_targets",
        params={"part_name": "P", "destination_location": "M2"}, recovery_snapshot={}, part_context={},
        task={"part_name": "P", "resource_jid": "robot"}, grounding_context={}, resource_evidence=evidence, physical_evidence={})
    assert result["output"]["approach_pose"]["z"] == pytest.approx(1.3455)
    assert result["output"]["retreat_pose"]["z"] == pytest.approx(retreat_z)


def test_a_place_after_its_own_grasp_takes_the_pick_context_that_grasp_recorded(monkeypatch):
    from cais_spade_llm.resources.robot import recovery_capability
    from cais_spade_llm.resources.robot.recovery_composition import preview_composition_output

    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **kwargs: {
        "status": "calculated", "destination_location": "M2", "target_pose": {"x": 0., "y": 0., "z": 1.}})
    picked = {"target_pose": {"x": 1.0}, "model_name": "P_model"}
    earlier = {"part_name": "earlier"}
    held = {"attachment": {"part_name": "P", "source": {"kind": "projected"}}, "state": {"_task_ctx": picked}}

    def place(evidence, pick_ctx):
        return preview_composition_output(primitive="compute_place_targets",
            params={"part_name": "P", "destination_location": "M2", "pick_ctx": pick_ctx}, recovery_snapshot={},
            part_context={}, task={"part_name": "P", "resource_jid": "robot"},
            grounding_context={"resources": {"robot": {"resource_evidence": {"state": {"_task_ctx": earlier}}}}},
            resource_evidence=evidence, physical_evidence={})
    assert place(held, picked)["status"] == "calculated"
    assert "pick_ctx" in place(held, earlier)["reason"]
    # Without a projected grasp of the part, the recorded context is still the one an explicit pick_ctx must equal.
    assert place({}, earlier)["status"] == "calculated"
    assert "pick_ctx" in place({}, picked)["reason"]


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
    # One finding states every given input the composition's calculation does not read as given.
    assert result["reason"] == (
        "compute_pick_targets: product_geometry {} differs from None, the input the composition's calculation reads; "
        "compute_pick_targets: target_pose {'x': 0.0, 'y': 0.0, 'z': 1.0} differs from None, the input the "
        "composition's calculation reads; compute_pick_targets: target_pose_source 'observed_pose' differs from '', "
        "the input the composition's calculation reads")
    assert preview_composition_output(params={"part_name": "P"}, **values)["status"] == "calculated"
    live = preview_composition_output(params={"part_name": "P", "prefer_live_detection": True}, **values)
    assert live["reason"] == "compute_pick_targets: prefer_live_detection reads a live detection the capture does not record"
    other = preview_composition_output(params={"part_name": "Q", "speed": 1.}, **values)
    assert other["reason"] == ("compute_pick_targets part_name 'Q' differs from the event's part 'P'; "
                               "compute_pick_targets: the composition's calculation does not model speed")
    # What the robot applies to an observed part stays the owner's: a program can restate it and choose nothing else.
    values["resource_evidence"] = {"motion_config": {"recovery_observed_pick_approach_height_m": .06,
                                                     "recovery_observed_pick_surface_clearance_m": .005}}
    assert preview_composition_output(params={
        "part_name": "P", "use_global_min_pick_tcp_z": False, "surface_clearance_override_m": .005,
        "approach_height_override_m": .06, "ignore_current_height_for_travel_z": True,
        "apply_pick_z_adjustments": False}, **values)["status"] == "calculated"
    for field, value in (("use_global_min_pick_tcp_z", True), ("surface_clearance_override_m", 0.),
                         ("approach_height_override_m", .2), ("ignore_current_height_for_travel_z", False),
                         ("apply_pick_z_adjustments", True), ("min_pick_tcp_z_override_m", 1.)):
        refused = preview_composition_output(params={"part_name": "P", field: value}, **values)
        assert refused["status"] == "NEEDS_CONTEXT" and field in refused["reason"]


def test_a_place_calculation_s_destination_is_the_place_the_event_states(monkeypatch):
    from cais_spade_llm.resources.robot import recovery_capability
    from cais_spade_llm.resources.robot.recovery_composition import preview_composition_output

    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **kwargs: {
        "status": "calculated", "destination_location": "M2", "target_pose": {"x": 0., "y": 0., "z": 1.}})
    values = dict(primitive="compute_place_targets", recovery_snapshot={}, part_context={}, grounding_context={},
                  task={"part_name": "P", "resource_jid": "robot", "expected_end_state": {"part_location": "M2"}},
                  resource_evidence={}, physical_evidence={})
    refused = preview_composition_output(params={"part_name": "P", "destination_location": "M1"}, **values)
    assert refused == {"status": "NEEDS_CONTEXT", "reason":
                       "compute_place_targets destination_location 'M1' differs from the event's part_location 'M2'"}
    # Left out, or given as stated, the calculation reads the place the event states.
    assert preview_composition_output(params={"part_name": "P"}, **values)["status"] == "calculated"
    assert preview_composition_output(params={"destination_location": "M2"}, **values)["status"] == "calculated"


def test_detect_parts_at_a_part_an_accepted_event_put_down_is_an_execution_obligation():
    from cais_spade_llm.resources.robot.recovery_capability import check_robot_primitive_capability

    pose = [-2.05, 0.5, 1.015, 0.0, 0.0, 0.0, 1.0]
    def check(row):
        return check_robot_primitive_capability(primitive="detect_parts", params={"part_name": "P"},
            task={"resource_jid": "robot", "part_name": "P"}, recovery_snapshot={}, static_capabilities={},
            resource_evidence={"observed_parts": {"P": row}}, primitive_catalog=[], resource_jid="robot")
    placed = check({"current_pose": pose, "frame": "world", "source": {"kind": "projected", "contract_fingerprint": "f"}})
    assert placed["feasibility_status"] == "FEASIBLE", placed
    # The pose compute_pick_targets reads stands in; the detection is confirmed when the step runs.
    assert placed["evidence"]["observed_pose"] == dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), pose))
    assert placed["evidence"]["composition_obligations"] == ["Confirm detection when the primitive executes"]
    incomplete = check({"current_pose": pose[:2], "frame": "world", "source": {"kind": "projected"}})
    assert incomplete["feasibility_status"] == "NEEDS_CONTEXT", incomplete


def test_detect_parts_previews_the_observation_the_owner_recorded():
    from cais_spade_llm.resources.robot.recovery_composition import preview_composition_output

    pose = {"x": 0.1, "y": 0.2, "z": 1.0, "qx": 0., "qy": 0., "qz": 0., "qw": 1.}
    values = dict(primitive="detect_parts", params={"part_name": "P"}, recovery_snapshot={},
                  part_context={"observed_pose": None}, grounding_context={},
                  task={"part_name": "P", "resource_jid": "robot"}, physical_evidence={})
    # The row the capability check and compute_pick_targets read; failure grounding measured no pose for P.
    seen = preview_composition_output(resource_evidence={"observed_parts": {"P": {"observed_pose": pose}}}, **values)
    assert seen["status"] == "calculated" and seen["output"]["pose"] == pose
    listed = {"observed_parts": {"P": {"current_pose": list(pose.values())}}}
    assert preview_composition_output(resource_evidence=listed, **values)["output"]["pose"] == pose
    assert preview_composition_output(resource_evidence={}, **values) == {
        "status": "NEEDS_CONTEXT", "reason": "Recorded part observation is missing"}


_STEP_EFFECT = {"part_pose.x": {"set_from_step": {"primitive": "calculate", "output": "target_x"}}}


def _step_effect_result(steps, *, output=None):
    catalog = [{"name": "calculate", "params": {}, "effects": {}},
               {"name": "move_relative", "params": {}, "effects": _STEP_EFFECT}]
    return validate_recovery_primitive_composition(
        task={"outline_id": "R1", "resource_jid": "belt", "expected_start_state": {}, "expected_end_state": {"resource_state": "idle"}},
        primitive_steps=[{"primitive": name, "params": {}} for name in steps],
        recovery_des_model={"state_variables": {"resource_state": {"scope": "resource", "domain": ["idle"]}}},
        recovery_snapshot={"resource_state": "idle"}, part_context={}, grounding_context={}, primitive_catalog=catalog,
        validate_primitive=lambda **_values: {"feasibility_status": "FEASIBLE", "allowed": True},
        preview_primitive=lambda *, primitive, **_values: (
            {"event_fact_key": "transport_targets", "output": output, "status": "calculated"}
            if primitive == "calculate" and output else {}))


def test_a_step_bound_effect_takes_what_the_earlier_step_calculated():
    result = _step_effect_result(["calculate", "move_relative"], output={"target_x": {"P": 2.5}})
    assert result["feasibility_status"] == "FEASIBLE", result
    assert result["trace"][1]["step_effects"] == {"part_pose.x": {"P": 2.5}}


@pytest.mark.parametrize("steps", [["move_relative"], ["move_relative", "calculate"]])
def test_a_step_bound_effect_needs_its_calculation_before_it(steps):
    result = _step_effect_result(steps, output={"target_x": {"P": 2.5}})
    assert result["feasibility_status"] == "INFEASIBLE"
    assert "needs an earlier calculate step" in result["reason"]


def test_the_composition_prompt_states_facts_and_no_recipe():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _PUBLISHED_REF_SCHEMA,
        _visible_primitive_catalog_names,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _render_primitive_generation_prompt,
    )

    event = {"outline_id": "e", "resource_jid": "r", "event_name": "go"}
    prompt = _render_primitive_generation_prompt({"primitive_active_outline_event_token": event,
                                                  "primitive_active_outline_event": event})
    for recipe in ("Prefer adapting", "Author from scratch", "worth pulling", "retrieve the closest", "fragile-but-valid",
                   "flipped-SG", "store_as", "get_current_pose"):
        assert recipe not in prompt
    assert "show how this resource's nominal functions compose primitives" in prompt
    contract = prompt.split("Output Contract\n")[1]
    assert sum(line.startswith("- ") for line in contract.splitlines()) == 8
    assert _visible_primitive_catalog_names([{"name": "grasp_part", "primitive_kind": "pick"}]) == [{"name": "grasp_part"}]
    assert not any("memo" in row["ref"] for row in _PUBLISHED_REF_SCHEMA)


def test_static_capabilities_is_described_by_what_composition_serves():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _PUBLISHED_REF_SCHEMA,
    )

    rows = {row["ref"]: row["description"] for row in _PUBLISHED_REF_SCHEMA}
    assert not any(promise in rows["/resources/<jid>/static_capabilities"]
                   for promise in ("named poses", "workspace bounds", "staging areas"))


_DECLARED_MOTION = {
    "ur5e-1": {"transit_waypoints": [[[-4.6, 1.55, 1.35], [-4.6, 0.7, 1.35]]], "rotation_waypoint": [-4.6, 0.7, 1.35]},
    "ur5e-3": {"transit_waypoints": [], "rotation_waypoint": [0.0, 0.05, 1.5],
               "pick_transit_waypoints": [[0.4, 0.15, 1.5], [0.0, 0.05, 1.5], [-0.4, 0.15, 1.5]]},
    "ur5e-4": {"transit_waypoints": [], "rotation_waypoint": [0.3, -0.35, 1.5], "pick_rotation_at_current_pose": True},
}


def _approach_step(prepared, function_name):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _capability_decomposition_for_resource,
    )

    decomposition = _capability_decomposition_for_resource(
        prepared_recovery_request=prepared, resource_jid="arm", function_name=function_name)
    return next(step for step in decomposition["recovery_visible_steps"] if step["primitive"] == "move_cartesian")


@pytest.mark.parametrize("robot, route", [("ur5e-1", "transit_waypoints/0"), ("ur5e-3", "pick_transit_waypoints"),
                                          ("ur5e-4", "rotation_waypoint")])
def test_a_nominal_approach_carries_no_route_note_and_the_route_stays_a_fact(robot, route):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _capability_decomposition_for_resource,
    )
    from cais_spade_llm.resources.robot.robot_task_registry import robot_task_capability_decompositions

    evidence = {"controller_config": {"cartesian_motion": _DECLARED_MOTION[robot]}}
    entry = {"resource_type": "robot", "resource_evidence": evidence}
    # The note read as how to fly the move, so composition serves the step templates as the task definitions have them
    # (user decision, 2026-11-06); a straight move the arm cannot fly is refused by composition's MoveIt check of
    # its path.
    for prepared in ({"composition_validation_version": 1, "recovery_resources": {"arm": entry}},
                     {"recovery_resources": {"arm": entry}}):
        for function_name in ("pick_approach", "place_approach"):
            assert "note" not in _approach_step(prepared, function_name)
        every = _capability_decomposition_for_resource(prepared_recovery_request=prepared, resource_jid="arm")
        for function in ("pick_approach", "place_approach"):
            steps = every[function]["recovery_visible_steps"]
            assert not any("note" in step for step in steps if step["primitive"] == "move_cartesian")
            assert not any("cartesian_motion" in step.get("note", "") for step in steps)
    for function in ("pick_approach", "place_approach"):
        steps = robot_task_capability_decompositions(function_name=function)["recovery_visible_steps"]
        assert not any("note" in step for step in steps if step["primitive"] == "move_cartesian")
    # The declared points stay a fact the program can bind under the arm's cartesian_motion lookup.
    ref = f"/resources/arm/resource_evidence/controller_config/cartesian_motion/{route}"
    context = {"resources": {"arm": {"resource_evidence": evidence}}}
    declared = _DECLARED_MOTION[robot]
    for key in route.split("/"):
        declared = declared[int(key)] if key.isdigit() else declared[key]
    assert resolve_composition_value({"context_ref": ref}, context) == declared


def test_an_arms_straight_move_is_left_to_the_moveit_check_of_its_path():
    from cais_spade_llm.resources.robot.recovery_capability import check_robot_primitive_capability

    # The arm's own capability no longer samples one point of a straight line near its base axis: where the arm can
    # go is MoveIt's Cartesian path on the recorded model and scene, which composition checks at the step
    # (test_composed_motion_check: test_a_straight_move_past_the_base_axis_is_refused_at_its_step).
    start = {"x": -0.68, "y": 0.5, "z": 1.5, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    checked = check_robot_primitive_capability(
        primitive="move_cartesian", params={"x": 0.25, "y": 0.45, "z": 1.3}, task={}, primitive_catalog=[],
        recovery_snapshot={"current_pose": start}, resource_evidence={"controller_config": {"move_group": {
            "frame_id": "world"}}}, static_capabilities={}, resource_jid="arm", physical_evidence={})
    assert checked["feasibility_status"] == "FEASIBLE"
    assert checked["reason"].endswith("composition checks the motion's path with MoveIt on the recorded model and scene")
    assert checked["evidence"]["target"] == {"x": 0.25, "y": 0.45, "z": 1.3}
    unstated = check_robot_primitive_capability(
        primitive="move_cartesian", params={"x": 0.25, "y": 0.45}, task={}, primitive_catalog=[],
        recovery_snapshot={"current_pose": start}, resource_evidence={}, static_capabilities={}, resource_jid="arm")
    assert unstated["feasibility_status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("part_yaw", [0.0, 0.6, -1.2])
def test_a_pick_for_a_mating_slot_leaves_the_fingers_across_the_row(part_yaw):
    from scipy.spatial.transform import Rotation

    from cais_spade_llm.resources.robot.recovery_capability import _mating_pick_orientation

    axes = ("qx", "qy", "qz", "qw")
    geometry = {"assembly_board": {"slots": {"near": [0.03, 0.14], "next": [0.0, 0.14], "far": [-0.05, 0.14]},
                                   "mating_contact_by_part": {"near": {
                                       "target_yaw_rad": math.pi / 2,
                                       "ignore_tooth_contact_with": ["next/link/collision", "far/link/collision"]}}}}
    down = dict(zip(axes, Rotation.from_euler("ZY", [-math.pi / 2, math.pi]).as_quat().tolist(), strict=True))
    observed = dict(zip(axes, Rotation.from_euler("z", part_yaw).as_quat().tolist(), strict=True))
    picked = _mating_pick_orientation("near", observed, down, geometry)
    tool = Rotation.from_quat([picked[axis] for axis in axes])
    grasp = tool.inv() * Rotation.from_quat([observed[axis] for axis in axes])
    # As the controller places it: the slot's yaw, with the grasp the pick made.
    at_slot = Rotation.from_euler("z", math.pi / 2) * grasp.inv()
    assert abs(at_slot.apply([1.0, 0.0, 0.0])[0]) < 1e-9
    assert tool.apply([0.0, 0.0, 1.0])[2] == pytest.approx(-1.0)
    # With no mating contact the orientation stays as it was given.
    assert _mating_pick_orientation("other", observed, down, geometry) is down


def test_an_arms_contracts_list_what_the_catalog_does_not():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _composition_contract,
    )

    release = {"name": "release_part", "params": {"assembly_slot": {"description": "Configured assembly target."}}}
    served = _composition_contract(release, "robot")
    for field in ("slot_x", "slot_y", "part_height", "board_top_z", "place_part_origin_z", "destination_location"):
        assert field in served["params"]["assembly_slot"]["description"]
    assert release["params"]["assembly_slot"]["description"] == "Configured assembly target."
    assert "retreat_pose" in _composition_contract({"name": "compute_pick_targets"}, "robot")["publishes"]
    assert "insert_pose" in _composition_contract({"name": "compute_place_targets"}, "robot")["publishes"]
    assert "publishes" not in _composition_contract({"name": "compute_pick_targets"}, "kmr")


def test_an_arms_place_calculation_says_it_reads_the_place_the_event_states_not_a_registered_token():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _composition_contract,
    )

    recorded = {"name": "compute_place_targets", "params": {"destination_location": {
        "type": "string", "description": "Optional registered controller destination token. A descriptive ontology "
                                         "label does not establish a resolvable token."}}}
    served = _composition_contract(recorded, "robot")["params"]["destination_location"]

    assert served == {"type": "string", "description": "The place the event states (its part_location, else its "
                                                       "resource_location); the composition's calculation reads it "
                                                       "from the event."}
    assert "registered controller" in recorded["params"]["destination_location"]["description"]


def test_every_aligned_motion_on_the_way_to_a_slot_release_carries_the_insertion_corridor(monkeypatch):
    """A program may reach its slot release through more than one motion (Part slippage v8, 2026-11-07).

    The flown program descended from the approach to the seated pose and released; the model's program added a move
    to the seated pose (a zero-length motion) or split the descent at a pre-insert pose. Only the last motion carried
    the recorded corridor, so the same insertion path was refused at the gear's contact with its shaft. Each earlier
    motion the corridor fits now carries it; one it does not fit gets no allowance.
    """
    from cais_spade_llm.recovery_framework import composed_motion_check as motion
    from cais_spade_llm.resources.robot import recovery_composition as robot

    slot = {"primitive": "release_part", "params": {"part_name": "gear_small", "assembly_slot": {"slot_x": 0.03}}}
    move = {"primitive": "move_cartesian", "params": {}}
    program = [{"primitive": "compute_place_targets", "params": {}}, move, move, move, slot, move]
    assert [robot._next_slot_release(program, index) is slot for index in (1, 2, 3)] == [True, True, True]
    assert robot._next_slot_release(program, 4) is None
    assert robot._next_slot_release([move, {"primitive": "move_base", "params": {}}, move, slot], 0) is None
    assert robot._next_slot_release([move, move, {"primitive": "release_part", "params": {}}], 0) is None

    recorded = []
    monkeypatch.setattr(motion, "motion_end_pose", lambda model, state, rid, step: [0.03, 0.14, 1.267, 0., 0., 0., 1.])
    monkeypatch.setattr(robot, "_slot_origin", lambda evidence, part: {"x": 0.03, "y": 0.14, "z": 1.039})

    def corridor(model, state, rid, evidence, *, part_name, target_origin_pose, target_tool_pose):
        recorded.append((part_name, target_tool_pose[2]))
        return {"status": "prepared", "part_name": part_name}

    monkeypatch.setattr(motion, "mating_contact_record", corridor)
    scene = SimpleNamespace(program=program, resource_id="ur5e-3")
    for index in (1, 2, 3):
        contacts, mating = robot._release_contacts(None, {}, scene, {}, {"step_index": index}, held="gear_small")
        assert contacts == [] and mating == {"status": "prepared", "part_name": "gear_small"}
    assert recorded == [("gear_small", 1.267)] * 3

    def not_fitting(*args, **kwargs):
        raise motion.MissingMotionContext("Mating evidence requires the owner's aligned vertical insertion stage")

    monkeypatch.setattr(motion, "mating_contact_record", not_fitting)
    # An earlier motion the corridor does not fit is checked with no contact allowed ...
    assert robot._release_contacts(None, {}, scene, {}, {"step_index": 2}, held="gear_small") == ([], None)
    # ... while the insertion itself still needs its corridor, as before.
    with pytest.raises(motion.MissingMotionContext):
        robot._release_contacts(None, {}, scene, {}, {"step_index": 3}, held="gear_small")
    assert robot._release_contacts(None, {}, scene, {}, {"step_index": 2}, held=None) == ([], None)


def test_a_place_descent_may_touch_the_support_its_place_calculation_names(monkeypatch):
    """Study 2, 2026-11-08: Terra's tray descent was refused as an unrelated collision with the M2 staging tray.

    The round part kept the 8e-4 rad tilt it settled with in M2, the allowance asked the support projection whether
    the part would rest at the descent's end, and its refusal was dropped. The support the place calculation names
    is allowed now, as MoveIt's place allows its support surface; the release check judges where the part rests.
    """
    from cais_spade_llm.recovery_framework import composed_motion_check as motion
    from cais_spade_llm.resources.robot import recovery_composition as robot

    part, tray = "RGOCG4-50_Round_4mm", "M2 staging tray::link::surface"
    move = {"primitive": "move_cartesian", "params": {}}
    release = {"primitive": "release_part", "params": {"part_name": part}}
    scene = SimpleNamespace(program=[{"primitive": "compute_place_targets", "params": {}}, move, release],
                            resource_id="ur5e-2")
    monkeypatch.setattr(motion, "motion_end_pose", lambda model, state, rid, step: [-2.6, 1.4, 1.243, 0., 1., 0., 0.])
    monkeypatch.setattr(motion, "projected_snapshot",
                        lambda model, state: {"resources": {"ur5e-2": {"grasp_transform": [0., 0., .243, 0., 0., 0., 1.]}}})
    monkeypatch.setattr(motion, "part_rest_support",
                        lambda model, held, pose: {"status": "NEEDS_CONTEXT",
                                                   "reason": "Tilted cylinder settling has no recorded projection model"})
    monkeypatch.setattr(motion, "place_arrival_contact",
                        lambda model, held, ids: {"part_name": held, "support_collision_ids": list(ids)})
    targets = {"compute_place_targets": {part: {"support_placement": {"support_collision_id": tray}}}}

    contacts, mating = robot._release_contacts(None, {}, scene, {}, {"step_index": 1}, held=part, targets=targets)
    assert contacts == [{"part_name": part, "support_collision_ids": [tray]}] and mating is None
    # A program with no place calculation for the part still takes the support found under it at the descent's end.
    assert robot._release_contacts(None, {}, scene, {}, {"step_index": 1}, held=part, targets={}) == ([], None)


def test_composition_serves_the_recorded_regions_and_scene_geometry_as_the_outline_stage_does():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _resolve_context_ref,
    )

    tray = {"bounds": [[-4.75, -4.6], [1.3, 1.5], [1.0, 1.2]]}
    prepared = {"composition_validation_version": 1, "diagnostic_regions": {"M1 staging tray": tray},
                "diagnostic_scene_geometry": {"floor_extent_xy": [-10.5, 1.5, -4.0, 4.8]}}

    served, error = _resolve_context_ref(ref="/regions/M1 staging tray", session_state={},
                                         prepared_recovery_request=prepared, outline_event={})
    assert error is None and served["value"] == tray
    whole, error = _resolve_context_ref(ref="/geometry", session_state={}, prepared_recovery_request=prepared,
                                        outline_event={})
    assert error is None and whole["value"]["floor_extent_xy"] == [-10.5, 1.5, -4.0, 4.8]
    missing, error = _resolve_context_ref(ref="/regions/Exit tray", session_state={}, prepared_recovery_request=prepared,
                                          outline_event={})
    assert missing is None and "is not recorded" in error



def test_a_composing_llm_cannot_end_its_composition_by_declaring_an_event_blocked(contract):
    """Study 2, 2026-11-08: Sol and Terra ended runs by declaring events blocked, with no program ever checked.

    Composition ends on its checks or its turn limits: its response format offers no block, and a block that still
    arrives is a reported gap, so the event goes on and the no-progress limit still applies.
    """
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_primitive_generation as generation,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        multi_turn_phase_response_schema,
    )

    assert "primitive_blocked" in multi_turn_phase_response_schema("primitive_generation")["schema"]["properties"][
        "decision"]["enum"]
    session = {"accepted_outline_prefix": [deepcopy(contract["task"])], "accepted_primitive_program": []}
    reply = {"decision": "primitive_blocked", "rationale": "No effect sets the part location.",
             "outline_id": "restore", "resource_jid": "owner"}
    decision, entry = asyncio.run(generation._handle_primitive_generation_phase(
        session_state=session, parsed_response=reply, prepared_recovery_request=_prepared(contract), planner=None))

    assert decision == "need_primitive_revision" and session["status"] == "paused_after_primitive_turn"
    assert entry["primitive_response"]["legacy_decision"] == "primitive_blocked"
    assert entry["primitive_rejection_feedback"][0]["reason"] == (
        "LLM reported primitive-surface gap: No effect sets the part location.")


def test_a_composed_program_may_reference_every_path_a_lookup_serves(contract, monkeypatch):
    """Study 2, 2026-11-08: Terra bound numbers to paths its lookups had served, and each was unresolved.

    The prompt says every number is a reference into the served context; a program's references now resolve in what
    /projected_snapshot, /parts/<name>/snapshot, /regions and /geometry serve, at the path shown. A path not shown
    stays unresolved.
    """
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_primitive_generation as generation,
    )
    from cais_spade_llm.resources.recovery_composition import resolve_composition_value

    zone = {"bounds": [[-4.75, -4.6], [1.3, 1.5], [1.0, 1.2]]}
    prepared = {**_prepared(contract), "diagnostic_regions": {"Buffer.zone_4": zone},
                "diagnostic_scene_geometry": {"floor_extent_xy": [-10.5, 1.5, -4.0, 4.8]}}
    task = {**contract["task"], "resource_jid": "owner"}
    context = generation._composition_parameter_context(session_state={}, prepared_recovery_request=prepared,
                                                        outline_event=task)

    def value(ref: str):
        return resolve_composition_value({"context_ref": ref}, context)

    assert value("/projected_snapshot/owner/mode") == value("/resources/owner/snapshot/mode") == "fault"
    assert value("/parts/P/snapshot/quality") == value("/parts/P/quality") == "pending"
    assert value("/regions/Buffer.zone_4/value/bounds/0/0") == value("/regions/value/Buffer.zone_4/bounds/0/0") == -4.75
    assert value("/geometry/value/floor_extent_xy/1") == value("/geometry/floor_extent_xy/value/1") == 1.5
    with pytest.raises(KeyError):
        value("/regions/Buffer.zone_4/bounds/0/0")


def test_a_machine_command_says_its_part_name_condition_reads_the_events_part():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _composition_contract,
    )

    pause = {"name": "pause_job", "description": "Recovery primitive (pre: availability, part_name, resource_state; "
                                                  "effects: resource_state)",
             "params": {"job_id": {"type": "string"}}, "required_params": ["job_id"],
             "preconditions": {"resource_state": {"equals": "running"}, "part_name": {"equals_from_param": "part_name"}}}
    served = _composition_contract(pause, "resource")

    assert served["description"].endswith("Its part_name condition reads the event's part_name; part_name is not a "
                                          "parameter of this command.")
    assert served["preconditions"] == pause["preconditions"] and served["params"] == pause["params"]
    start = {"name": "start_job", "params": {"part_name": {"type": "string"}},
             "preconditions": {"part_name": {"equals_from_param": "part_name"}}}
    assert "description" not in _composition_contract(start, "resource")
