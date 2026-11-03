from __future__ import annotations

"""Contract-level regression coverage for generalized ordered composition."""

import asyncio
from copy import deepcopy
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
    contract["primitive_steps"].append({"primitive": "travel", "params": {"target_pose": [-5.47, 1.1, 3.1416]}})
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
    # A transfer needs the owner's calculated target, and has to happen there.
    assert "preceding owner target calculation" in _kmr_step(evidence, targets, "grasp_part", part_name="P")
    grasp = [3.0, 1.0, 0.925, *down]
    lift = [3.0, 1.0, 1.1, *down]
    clear = [3.0, 1.0, 0.995, *down]
    calculated = {"target": grasp, "retreat": lift, "lift": lift, "lift_waypoints": [clear]}
    assert _kmr_step(evidence, targets, "compute_pick_targets", facts={"calculated_targets": calculated}) is None
    assert "calculated target pose" in _kmr_step(evidence, targets, "grasp_part", part_name="P")
    assert _kmr_step(evidence, targets, "move_cartesian", target=grasp, facts={"joint_positions": [0.5] * 7}) is None
    assert evidence["snapshot"]["joint_positions"] == [0.5] * 7
    assert _kmr_step(evidence, targets, "grasp_part", part_name="P") is None
    # The grip is where the part sat in the tool, so a release puts it back through the same transform.
    attachment = evidence["attachment"]
    assert attachment["attached"] is True and attachment["held_part"] == "P"
    assert compose(grasp, attachment["grasp_transform"]) == pytest.approx([3.0, 1.0, 0.9, 0.0, 0.0, 0.0, 1.0])
    # As for an arm, the next motion has to reach the calculated retreat, so the program chooses no height.
    assert "retreat" in targets
    low = [3.0, 1.0, 0.995, *down]
    assert "calculated event_facts retreat" in _kmr_step(deepcopy(evidence), dict(targets), "move_cartesian", target=low)
    away = [2.0, 1.0, 0.0]
    assert "calculated event_facts retreat" in _kmr_step(deepcopy(evidence), dict(targets), "move_base", target_pose=away,
                                                         base=evidence["snapshot"]["base_pose"])
    assert _kmr_step(evidence, targets, "move_cartesian", target=lift, waypoints=[clear]) is None
    assert "retreat" not in targets


def test_a_composed_kmr_release_takes_no_part_pose_of_its_own():
    # A place the event states and a pose that lands elsewhere would give the slot to a part that is not in it.
    reason = _kmr_step({"snapshot": {}}, {}, "compute_place_targets", part_pose=[0.0, 0.0, 1.0],
                       facts={"calculated_targets": {}})
    assert reason == "A release composes at the place the event states, so compute_place_targets takes no part_pose"


def _kmr_completion(*, trace, stated, snapshot, evidence=None, part_context=None, targets=None, physical=None,
                    start=None, neighbors=None):
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
        part_context=part_context or {}, grounding_context={},
        targets=targets or {}, resource_evidence=recorded, physical_evidence=physical or {})


@pytest.mark.parametrize("base, stated, reached", [
    ([-4.7, 2.2], {"resource_location": "M1"}, "M1"), ([-3.3, 1.3], {"resource_location": None}, None),
    # A stated place the base did not reach takes where it ended, so the postcondition refuses it.
    ([-3.3, 1.3], {"resource_location": "M1"}, None), ([-4.7, 2.2], {"resource_location": None}, "M1")])
def test_a_kmr_location_the_event_states_is_compared_with_the_dock_its_base_reached(base, stated, reached):
    # Facing as the docks declare, yaw 1.57.
    snapshot = {"resource_location": "Storage", "current_location": "Storage",
                "base_pose": [*base, 0, 0, 0, 0.70682518, 0.70738827]}
    result = _kmr_completion(trace=[{"step_index": 0, "primitive": "move_base", "resolved_params": {}}], stated=stated,
                             snapshot=snapshot, evidence={"snapshot": {"base_pose": snapshot["base_pose"]}})
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    assert result["projected_snapshot"]["resource_location"] == reached
    assert result["projected_snapshot"]["current_location"] == reached


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


def test_a_kmr_release_follows_the_routes_calculated_for_it():
    evidence, targets, calculated = _kmr_place([-2.7, 1.35])
    # release_heights puts the retreat where the approach stood, so a check on the pose alone would refuse the retreat.
    assert calculated["retreat"] == calculated["approach"]
    assert calculated["transfer_waypoints"] and calculated["withdraw_waypoints"]

    def tried(primitive, **params):
        return _kmr_step(deepcopy(evidence), deepcopy(targets), primitive, **params)

    assert _kmr_step(evidence, targets, "compute_place_targets", facts={"calculated_targets": calculated}) is None
    assert "transfer_waypoints" in tried("move_cartesian", target=calculated["approach"])
    assert "transfer_waypoints" in tried("move_cartesian", target=calculated["approach"],
                                         waypoints=[[0.0, 0.0, 1.0]] * len(calculated["transfer_waypoints"]))
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["approach"],
                     waypoints=calculated["transfer_waypoints"]) is None
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["target"]) is None
    assert _kmr_step(evidence, targets, "release_part") is None
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["retreat"]) is None
    assert "withdraw_waypoints" in tried("move_cartesian", target=calculated["transport"])
    # A named pose or a relative move reaches the same place by the straight line the waypoints are there to avoid.
    assert "withdraw_waypoints" in tried("move_to_named_pose", pose_name="home",
                                         facts={"target": [*calculated["transport"][:3], *calculated["retreat"][3:]]})
    assert "withdraw_waypoints" in tried("move_relative", offset=[
        want - have for want, have in zip(calculated["transport"][:3], calculated["retreat"][:3], strict=True)])
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["transport"],
                     waypoints=calculated["withdraw_waypoints"]) is None


def test_a_kmr_release_comes_through_its_calculated_approach():
    evidence, targets, calculated = _kmr_place([-2.7, 1.35])
    assert _kmr_step(evidence, targets, "compute_place_targets", facts={"calculated_targets": calculated}) is None
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["target"]) is None
    assert "calculated approach" in _kmr_step(evidence, targets, "release_part")


@pytest.mark.parametrize("left_out", [{}, {"waypoints": None}, {"waypoints": []}])
def test_a_kmr_turn_under_the_minimum_angle_has_no_waypoints_to_bind(left_out):
    evidence, targets, calculated = _kmr_place([-3.3, 2.0])
    assert calculated["transfer_waypoints"] == [] and calculated["withdraw_waypoints"] == []
    assert _kmr_step(evidence, targets, "compute_place_targets", facts={"calculated_targets": calculated}) is None
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["approach"], **left_out) is None
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["target"]) is None
    assert _kmr_step(evidence, targets, "release_part") is None
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["retreat"]) is None
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["transport"], **left_out) is None


def test_a_kmr_lift_follows_the_route_calculated_for_it():
    from cais_spade_llm.recovery_framework.kmr_recovery import pick_targets

    evidence, targets, _ = _kmr_place([-2.7, 1.35])
    state = evidence["snapshot"]
    part = [-2.7, 1.35, 1.0, 0.0, 0.0, 0.0, 1.0]
    evidence["observed_parts"] = {"P": {"current_pose": part}}
    calculated = pick_targets(evidence["inputs"]["scene"], "P", part, state["current_pose"], state["base_pose"],
                              location=None)
    assert calculated["lift"] == calculated["retreat"] and calculated["lift_waypoints"]
    assert _kmr_step(evidence, targets, "compute_pick_targets", facts={"calculated_targets": calculated}) is None
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["target"]) is None
    assert _kmr_step(evidence, targets, "grasp_part", part_name="P") is None

    def tried(primitive, **params):
        return _kmr_step(deepcopy(evidence), deepcopy(targets), primitive, **params)

    assert "lift_waypoints" in tried("move_cartesian", target=calculated["lift"])
    assert "lift_waypoints" in tried("move_to_named_pose", pose_name="transport", facts={"target": calculated["lift"]})
    assert _kmr_step(evidence, targets, "move_cartesian", target=calculated["lift"],
                     waypoints=calculated["lift_waypoints"]) is None


def test_a_kmr_base_moves_only_with_its_arm_at_the_transport_pose():
    base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    scene = _kmr_scene([-0.25, 0.5, 1.16])
    out = {"inputs": {"scene": scene}, "snapshot": {"base_pose": base, "current_pose": [0.5, 0.0, 1.16, *_KMR_TOOL_DOWN]}}
    reason = _kmr_step(out, {}, "move_base", target_pose=[1.0, 0.0, 0.0], base=base)
    assert "0.901 m from the transport pose" in reason
    parked = {"inputs": {"scene": scene},
              "snapshot": {"base_pose": base, "current_pose": [-0.25, 0.5, 1.16, *_KMR_TOOL_DOWN]}}
    assert _kmr_step(parked, {}, "move_base", target_pose=[1.0, 0.0, 0.0], base=base) is None


@pytest.mark.parametrize("offset, status", [
    (0.0, "FEASIBLE"), (0.005, "FEASIBLE"), (0.007, "INFEASIBLE"), (0.2, "INFEASIBLE")])
def test_a_kmr_program_ends_with_the_arm_at_the_transport_pose_of_its_base(offset, status):
    from cais_spade_llm.recovery_framework.kmr_recovery import transport_pose

    base = [-5.54, 2.13, 0, 0, 0, 0.70682518, 0.70738827]
    parked = transport_pose(_kmr_scene([-0.25, 0.5, 1.16]), base, [0.0, 0.0, 0.0, *_KMR_TOOL_DOWN])
    tool = [parked[0] + offset, *parked[1:]]
    result = _kmr_completion(trace=[], stated={"resource_state": "positioned"}, snapshot={"base_pose": base},
                             evidence={"snapshot": {"base_pose": base, "current_pose": tool}}, start={"base_pose": base})
    assert result["feasibility_status"] == status
    if status == "INFEASIBLE":
        assert f"{offset:.3f} m from the transport pose" in result["reason"]


def test_a_kmr_tool_left_tilted_is_refused_with_the_fact():
    base = [-5.54, 2.13, 0, 0, 0, 0.70682518, 0.70738827]
    tool = [-6.0, 1.9, 1.16, 0.0, 0.0, 0.0, 1.0]
    result = _kmr_completion(trace=[], stated={"resource_state": "positioned"}, snapshot={"base_pose": base},
                             evidence={"snapshot": {"base_pose": base, "current_pose": tool}}, start={"base_pose": base})
    assert result["feasibility_status"] == "INFEASIBLE" and "downward" in result["reason"]


def test_the_composition_prompt_names_the_kmr_facts_and_gives_no_recipe():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _kmr_composition_guidance,
    )

    guidance = _kmr_composition_guidance("kmr")
    for name in ("lift_waypoints", "transfer_waypoints", "withdraw_waypoints"):
        assert name in guidance
    for recipe in ("Bind ", "bind ", "Whatever resource_state", "Custody completion", "adapt its"):
        assert recipe not in guidance


def test_a_kmr_grasp_gives_it_custody_and_needs_its_retreat():
    grasp = {"step_index": 0, "primitive": "grasp_part", "resolved_params": {"part_name": "P"}}
    motion = {"step_index": 1, "primitive": "move_relative", "resolved_params": {}}
    stated = {"held_part": "P", "part_location": "kmr"}
    refused = _kmr_completion(trace=[grasp], stated=stated, snapshot={"held_part": "P"}, targets={"retreat": [0.0] * 7})
    assert refused["reason"] == "Owner completion requires a retreat after the custody transfer"
    result = _kmr_completion(trace=[grasp, motion], stated=stated, snapshot={"held_part": "P"},
                             part_context={"part_location": "M1"})
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    assert result["projected_part_context"]["part_location"] == "kmr"
    assert result["projected_part_context"]["part_holder_resource_jid"] == "kmr"


def test_a_kmr_release_gives_the_part_to_the_slot_that_receives_it():
    part = {"reference": "part_name", "scope": "resource"}
    physical = {"product_name": "board", "jids": {"Rack": "rack"},
                "environment_models": {"Rack": {"state_variables": {"zone_1_part": part, "zone_2_part": part},
                    "assignments": {"zones": [{"zone": 1, "pose": [0.0, 0.5, 1.0]}, {"zone": 2, "pose": [0.2, 0.5, 1.0]}]}}},
                "environment_valuation": {"Rack": {"zone_1_part": None, "zone_2_part": None}}}
    release = {"step_index": 0, "primitive": "release_part", "resolved_params": {}}
    motion = {"step_index": 1, "primitive": "move_cartesian", "resolved_params": {}}
    pose = [0.2, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0]
    result = _kmr_completion(trace=[release, motion], stated={"held_part": None, "part_location": "Rack"},
                             snapshot={"held_part": None}, physical=physical,
                             targets={"compute_place_targets": {"destination": pose}},
                             neighbors={"rack": {"zone_2_part": "P"}})
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    assert result["projected_resources"]["rack"] == {"zone_1_part": None, "zone_2_part": "P"}
    # A slot the event leaves out is not written; the RA refuses such a place (unstated_slot).
    unstated = _kmr_completion(trace=[release, motion], stated={"held_part": None, "part_location": "Rack"},
                               snapshot={"held_part": None}, physical=physical,
                               targets={"compute_place_targets": {"destination": pose}})
    assert "rack" not in unstated.get("projected_resources", {})
    assert result["projected_part_context"]["part_location"] == "Rack"
    assert result["projected_part_context"]["pose_source"] == "validated_composition_projection"
    assert result["resource_evidence"]["observed_parts"]["P"]["current_pose"] == pose
    assert result["resource_evidence"]["observed_parts"]["P"]["part_location"] == "Rack"
    # A zone holding another part takes no more.
    physical["environment_valuation"]["Rack"]["zone_2_part"] = "Q"
    occupied = _kmr_completion(trace=[release, motion], stated={"held_part": None, "part_location": "Rack"},
                               snapshot={"held_part": None}, physical=physical,
                               targets={"compute_place_targets": {"destination": pose}})
    assert occupied == {"feasibility_status": "INFEASIBLE", "reason": "The receiving zone is occupied"}


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


def test_a_grasp_that_misses_its_part_is_refused_with_the_numbers(monkeypatch):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    pick_grasp = {"event_name": "pick_grasp", "updates": {"part_location": {"set": "robot"}},
        "composition_contract": {"steps": [{"op": "grasp_part", "params": {}}, {"op": "move_relative", "params": {
            "dx": 0., "dy": 0., "dz": {"$sub": [{"$state": "_task_ctx", "path": ["travel_z"]},
                                                {"$state": "_position", "path": ["z"]}]}}}]}}
    model = {"events": [pick_grasp], "state_variables": {"part_location": {"scope": "part", "domain": ["robot"]}}}
    grasp = {"x": 0.3, "y": 0.5, "z": 1.3, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    lift = {**grasp, "z": 1.4}
    upright = {"qx": 0., "qy": 0., "qz": 0., "qw": 1.}
    physical = {"geometry": {"parts": {"P": {"local_bounds": [[-.002, .002], [-.002, .002], [0., .05]]}}}}

    def complete(tcp_z, part_x=0.3, physical=physical, evidence=_GRIP_EVIDENCE):
        target = {"target_pose": grasp, "retreat_pose": lift, "model_name": "P_model",
                  "observed_pose": {"x": part_x, "y": 0.5, "z": 1.0, **upright}, "tcp_offset_z": tcp_z - grasp["z"]}
        trace = [{"step_index": 0, "primitive": "compute_pick_targets", "params": {}, "resolved_params": {"part_name": "P"},
                  "preview_output": target, "start_snapshot": {}},
                 {"step_index": 1, "primitive": "grasp_part", "params": {}, "resolved_params": {"part_name": "P"},
                  "start_snapshot": {"current_pose": grasp}},
                 {"step_index": 2, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
                  "start_snapshot": {"current_pose": grasp}, "projected_snapshot": {"current_pose": lift}}]
        return project_composition_completion(
            task={"outline_id": "pick", "event_name": "pick", "resource_jid": "robot", "part_name": "P"}, trace=trace,
            recovery_des_model=model, recovery_snapshot={}, projected_snapshot={"current_pose": lift},
            part_context={"part_location": None}, grounding_context={}, primitive_catalog=[], resource_evidence=evidence,
            physical_evidence=physical)

    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    assert complete(1.0175)["feasibility_status"] == "FEASIBLE"
    # Within the recorded tolerance of the part's top, as the controller's payload check has it.
    assert complete(1.054)["feasibility_status"] == "FEASIBLE"
    above = complete(1.07)
    assert above["feasibility_status"] == "INFEASIBLE"
    assert "is 0.0200 m from P" in above["reason"] and "tolerance 0.005 m" in above["reason"]
    assert "size [0.004, 0.004, 0.05]" in above["reason"]
    # The TCP is where the grasp pose puts it, not where the part was observed.
    beside = complete(1.0175, part_x=0.31)
    assert beside["feasibility_status"] == "INFEASIBLE" and "is 0.0080 m from P" in beside["reason"]
    # Evidence the check needs is asked for, not passed over.
    assert complete(1.0175, physical={})["feasibility_status"] == "NEEDS_CONTEXT"
    assert complete(1.0175, evidence={})["feasibility_status"] == "NEEDS_CONTEXT"


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
    slot = {"reference": "part_name", "scope": "resource"}
    physical = {"product_name": "board", "jids": {"Rack": "rack"},
                "environment_models": {"Rack": {"state_variables": {"zone_1_part": slot, "zone_2_part": slot},
                    "assignments": {"zones": [{"zone": 1, "pose": [0.0, 0.5, 1.0]}, {"zone": 2, "pose": [0.2, 0.5, 1.0]}]}}},
                "environment_valuation": {"Rack": {"zone_1_part": "P", "zone_2_part": None}}}
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    def complete(end, neighbors=None):
        return project_composition_completion(
            task={"outline_id": "move", "event_name": "move", "resource_jid": "robot", "part_name": "P",
                  "expected_end_state": end, "neighbor_end_state": neighbors}, trace=trace,
            recovery_des_model=model, recovery_snapshot={"resource_state": "idle"},
            projected_snapshot={"resource_state": "idle", "current_pose": away}, part_context={"part_location": "Rack"},
            grounding_context={"resources": {"rack": {"zone_1_part": "P", "zone_2_part": None}}}, primitive_catalog=[],
            resource_evidence=_GRIP_EVIDENCE, physical_evidence={**physical, **_PART_GEOMETRY})

    stated = {"rack": {"zone_1_part": None, "zone_2_part": "P"}}
    result = complete({"resource_location": "Rack", "resource_state": "placed"}, stated)
    if not lifted:
        # The motion right after the grasp must reach the grasp's calculated retreat pose.
        assert result["feasibility_status"] == "INFEASIBLE" and "retreat_pose" in result["reason"]
        return
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    # The part leaves its source slot and is received by the slot under the place target, in one record.
    assert result["projected_resources"]["rack"] == {"zone_1_part": None, "zone_2_part": "P"}
    # A source slot the event leaves out keeps its value: code writes none, and the RA refuses such an event.
    unstated = complete({"resource_location": "Rack", "resource_state": "placed"}, {"rack": {"zone_2_part": "P"}})
    assert unstated["projected_resources"]["rack"]["zone_1_part"] == "P"
    assert result["projected_part_context"]["part_location"] == "Rack"
    assert result["projected_snapshot"]["resource_location"] == "Rack"
    assert result["projected_snapshot"]["resource_state"] == "placed"
    assert result["resource_evidence"]["attachment"] == {}
    # A later pick reads where the part now rests from the row this place leaves.
    assert result["resource_evidence"]["observed_parts"]["P"]["part_location"] == "Rack"
    # What the event leaves out, composition does not write for it.
    left_out = complete({})
    assert left_out["feasibility_status"] == "FEASIBLE", left_out.get("reason")
    assert "resource_location" not in left_out["projected_snapshot"]
    assert left_out["projected_snapshot"]["resource_state"] == "idle"
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
    trace = [{"step_index": 0, "primitive": "compute_place_targets", "params": {}, "resolved_params": {"part_name": "P"},
              "preview_output": target, "start_snapshot": {}},
             {"step_index": 1, "primitive": "release_part", "params": {},
              "resolved_params": {"part_name": "P", "assembly_slot": {
                  **slot, "destination_location": "Board", "part_origin_z": 1.01}},
              "start_snapshot": {"current_pose": place}},
             {"step_index": 2, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
              "start_snapshot": {"current_pose": place}, "projected_snapshot": {"current_pose": away}}]
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
    pick = {"target_pose": grasp, "retreat_pose": lift, "model_name": "P_model", "observed_pose": {**grasp, "z": 1.0}, **_TCP}
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
        grounding_context={}, primitive_catalog=[], resource_evidence=_GRIP_EVIDENCE, physical_evidence=_PART_GEOMETRY,
        locate_pose=locate)
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    # After its retreat the robot moved on, so its pickup place no longer holds: its final pose decides.
    located = result["projected_snapshot"]["resource_location"]
    assert located == ("M1" if after == "move_cartesian" else expected)
    assert seen == ([stated] if after == "move_cartesian" else [])


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
    trace = [{"step_index": 0, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
              "start_snapshot": {"current_pose": pose}, "projected_snapshot": {"current_pose": pose}}]
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "move", "event_name": "move", "resource_jid": "robot", "expected_end_state": end}, trace=trace,
        recovery_des_model=model, recovery_snapshot={"resource_location": "Bench"},
        projected_snapshot={"resource_location": "Bench", "current_pose": pose}, part_context={}, grounding_context={},
        primitive_catalog=[], resource_evidence={}, physical_evidence={}, locate_pose=lambda _, stated=None: stated or "Rack")
    assert result["feasibility_status"] == status, result.get("reason")
    if status == "INFEASIBLE":
        assert result["reason"] == "The program ends at Rack and the event states no resource_location"


@pytest.mark.parametrize("part_at, end, expected", [
    ("Rack", {}, "Bench"), (None, {}, "Bench"), ("other@localhost", {}, "Bench"), ("other", {}, "Bench"),
    ("Rack tray", {}, "Bench"), (None, {"resource_location": None}, None),
    ("Rack tray", {"resource_location": "Bench"}, "Bench"), ("Rack", {"resource_location": "Rack"}, "Rack"),
    # A stated location the pick does not end at takes where it ends, so the postcondition refuses it.
    (None, {"resource_location": "Bench"}, None), ("Rack", {"resource_location": "Bench"}, "Rack")])
def test_a_pick_leaves_the_robot_where_the_event_states(monkeypatch, part_at, end, expected):
    from cais_spade_llm.resources.robot import robot_primitives
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    retreat_step = {"op": "move_relative", "params": {"dx": 0., "dy": 0., "dz": {"$sub": [
        {"$state": "_task_ctx", "path": ["travel_z"]}, {"$state": "_position", "path": ["z"]}]}}}
    pick_grasp = {"event_name": "pick_grasp", "updates": {
        "resource_location": {"set_from_param": "origin_resource_location"}, "part_location": {"set": "robot"}},
        "composition_contract": {"steps": [{"op": "grasp_part", "params": {}}, retreat_step]}}
    model = {"events": [pick_grasp], "state_variables": {
        "part_location": {"scope": "part", "domain": ["Rack", "Rack tray", "other", "other@localhost", "robot"]},
        "resource_location": {"scope": "resource", "domain": ["Rack", "Bench"]}}}
    down = {"qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    grasp, lift = {"x": 0.0, "y": 0.5, "z": 1.1, **down}, {"x": 0.0, "y": 0.5, "z": 1.3, **down}
    pick = {"target_pose": grasp, "retreat_pose": lift, "model_name": "P_model", "observed_pose": {**grasp, "z": 1.0}, **_TCP}
    trace = [{"step_index": 0, "primitive": "compute_pick_targets", "params": {}, "resolved_params": {"part_name": "P"},
              "preview_output": pick, "start_snapshot": {}, "projected_snapshot": {}},
             {"step_index": 1, "primitive": "grasp_part", "params": {}, "resolved_params": {"part_name": "P"},
              "start_snapshot": {"current_pose": grasp}, "projected_snapshot": {"current_pose": grasp}},
             {"step_index": 2, "primitive": "move_cartesian", "params": {}, "resolved_params": {},
              "start_snapshot": {"current_pose": grasp}, "projected_snapshot": {"current_pose": lift}}]
    monkeypatch.setattr(robot_primitives, "robot_primitive_sequence_validator", lambda **_: [])
    result = project_composition_completion(
        task={"outline_id": "take", "event_name": "take", "resource_jid": "robot", "part_name": "P",
              "expected_end_state": end}, trace=trace,
        recovery_des_model=model, recovery_snapshot={"resource_location": "Bench"},
        projected_snapshot={"resource_location": "Bench", "current_pose": lift}, part_context={"part_location": part_at},
        grounding_context={"resources": {"other@localhost": {}}}, primitive_catalog=[], resource_evidence=_GRIP_EVIDENCE,
        physical_evidence=_PART_GEOMETRY, locate_pose=lambda pose, stated=None: stated)
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    assert result["projected_snapshot"]["resource_location"] == expected


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


def _stated_pose_case(stated_pose, steps):
    from cais_spade_llm.resources.recovery_composition_owner import validate_owner_composition

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

    return validate_owner_composition(SimpleNamespace(jid="robot", check_recovery_primitive_feasibility=check),
        task=task, primitive_steps=steps, prepared=prepared, recovery_snapshot=start, part_context={},
        grounding_context={"outline_event": task, "resources": {"robot": dict(start)}})


_STATED_POSE_STEP = {"primitive": "move_cartesian", "params": {
    **{axis: {"context_ref": f"/outline_event/expected_end_state/pose/{axis}"} for axis in "xyz"},
    **{axis: {"context_ref": f"/resources/robot/current_pose/{axis}"} for axis in _TOOL_DOWN}}}


def test_a_program_binds_the_stated_pose_and_the_robot_ends_at_the_stated_place():
    result = _stated_pose_case({"x": -4.6, "y": 1.1, "z": 1.30}, [_STATED_POSE_STEP])
    assert result["feasibility_status"] == "FEASIBLE", result.get("reason")
    assert result["projected_snapshot"]["resource_location"] == "M1 staging tray"
    assert result["trace"][0]["resolved_params"] == {"x": -4.6, "y": 1.1, "z": 1.30, **_TOOL_DOWN}


def test_a_program_that_ends_short_of_the_stated_pose_is_refused():
    elsewhere = {"primitive": "move_cartesian", "params": {"x": -4.6, "y": 1.1, "z": 1.28}}
    result = _stated_pose_case({"x": -4.6, "y": 1.1, "z": 1.30}, [elsewhere])
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reason"].startswith("Postcondition 'pose': achieved")


def test_a_stated_pose_whose_gripper_is_out_of_the_stated_place_does_not_end_there():
    result = _stated_pose_case({"x": -4.6, "y": 1.1, "z": 1.35}, [_STATED_POSE_STEP])
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reason"] == "Postcondition 'resource_location': achieved None, required 'M1 staging tray'"


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

    assert "a location no primitive declares is read from where the program ends" in calls[0]
    assert "bind" not in calls[0].split("RA composition validation:")[1].split("Use exact primitive identities")[1].split(
        "Missing facts require")[1]


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


@pytest.mark.parametrize("robot, function_name, route", [
    ("ur5e-1", "place_approach", "transit_waypoints/0"), ("ur5e-3", "pick_approach", "pick_transit_waypoints")])
def test_a_nominal_approach_names_the_route_its_robot_declares(robot, function_name, route):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _capability_decomposition_for_resource,
    )

    evidence = {"controller_config": {"cartesian_motion": _DECLARED_MOTION[robot]}}
    prepared = {"composition_validation_version": 1,
                "recovery_resources": {"arm": {"resource_type": "robot", "resource_evidence": evidence}}}
    ref = f"/resources/arm/resource_evidence/controller_config/cartesian_motion/{route}"
    note = _approach_step(prepared, function_name)["note"]
    assert ref in note
    # It says what the nominal function does; the program is the model's to write.
    assert "move_cartesian" not in note and "add" not in note.split()
    # The other approach has no route of its own, so it says nothing.
    other = "pick_approach" if function_name == "place_approach" else "place_approach"
    assert "note" not in _approach_step(prepared, other)
    # What the note says to bind, the owner check resolves to a declared point.
    context = {"resources": {"arm": {"resource_evidence": evidence}}}
    point = resolve_composition_value({"context_ref": ref}, context)[1]
    bound = {axis: {"context_ref": f"{ref}/1/{index}"} for index, axis in enumerate("xyz")}
    assert resolve_composition_value(bound, context) == dict(zip("xyz", point))
    # Asked for every function at once, it reads the same.
    every = _capability_decomposition_for_resource(prepared_recovery_request=prepared, resource_jid="arm")
    assert any(step.get("note") == note for step in every[function_name]["recovery_visible_steps"])


def test_a_robot_with_no_route_and_the_task_definitions_stay_as_they_are():
    from cais_spade_llm.resources.robot.robot_task_registry import robot_task_capability_decompositions

    evidence = {"controller_config": {"cartesian_motion": _DECLARED_MOTION["ur5e-4"]}}
    entry = {"resource_type": "robot", "resource_evidence": evidence}
    composition = {"composition_validation_version": 1, "recovery_resources": {"arm": entry}}
    for function_name in ("pick_approach", "place_approach"):
        assert "note" not in _approach_step(composition, function_name)
    # Only composition reads the note. The task definitions are every robot's, so they never carry one robot's route.
    evidence["controller_config"]["cartesian_motion"] = _DECLARED_MOTION["ur5e-1"]
    assert "note" not in _approach_step({"recovery_resources": {"arm": entry}}, "place_approach")
    assert "note" in _approach_step(composition, "place_approach")
    steps = robot_task_capability_decompositions(function_name="place_approach")["recovery_visible_steps"]
    assert not any("note" in step for step in steps)


def test_a_straight_move_finds_where_it_passes_nearest_an_arms_base_axis():
    from cais_spade_llm.resources.robot.recovery_capability import _nearest_base_axis

    start, end = {"x": -0.68, "y": 0.5, "z": 1.5}, {"x": 0.25, "y": 0.45, "z": 1.3}
    nearest = _nearest_base_axis(start, end, [0.0, 0.5, 1.02])
    assert nearest["x"] == pytest.approx(0.0, abs=2e-3) and nearest["z"] == pytest.approx(1.354, abs=2e-3)
    assert 0.03 < abs(nearest["y"] - 0.5) < 0.04
    # A move that only leaves the axis behind has no point of its own to check.
    assert _nearest_base_axis({"x": 0.3, "y": 0.5, "z": 1.5}, {"x": 0.6, "y": 0.5, "z": 1.5}, [0.0, 0.5, 1.02]) is None
    assert _nearest_base_axis({"x": 0.3, "y": 0.5, "z": 1.5}, {"x": 0.3, "y": 0.5, "z": 1.2}, [0.0, 0.5, 1.02]) is None


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
