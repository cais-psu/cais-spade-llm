"""RA feasibility of a proposed recovery event: backward derivation (Alg. 1) and the forward check (Eq. 12)."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from scipy.spatial.transform import Rotation

from cais_spade_llm.recovery_framework import SCENE_PATH
from cais_spade_llm.recovery_framework.ra_feasibility import RecordedCell, check_event
from cais_spade_llm.resources.resource_primitives import current_primitive_declarations

ARM = json.loads((Path(__file__).parent / "fixtures/recorded_arm_chains.json").read_text())["arms"]["ur5e-3"]
UR, BUFFER, M1, CONVEYOR = (f"recovery-resource-{index}@localhost" for index in (3, 10, 5, 9))
PARTS = [None, "P1", "P2"]


def _region(x, y, z, capacity=1):
    return {"frame": "world", "bounds": [x, y, z], "capacity": capacity}


@pytest.fixture(scope="module")
def cell():
    scene = json.loads(SCENE_PATH.read_text())
    configuration = ARM["configuration"]
    return RecordedCell(
        scene=scene,
        resources={"ur5e-3": {"joint_names": configuration["joint_names"], "joint_positions": ARM["joint_positions"],
                              "current_pose": ARM["reference_pose"], "tcp_offset_z": -0.218,
                              "tcp_link": "ur5e_3_rg2_gripper_tcp"}},
        models={"ur5e-3": {"configuration": configuration, "named_joint_poses": ARM["named_joint_poses"]}},
        regions={"Buffer For Machined parts.zone_1": _region([-0.74, -0.62], [0.38, 0.62], [1.017, 1.067]),
                 "Conveyor.loading_position_1": _region([-4.2, -3.9], [0.35, 0.65], [1.015, 1.065]),
                 "Conveyor.loading_position_2": _region([-2.2, -1.9], [0.35, 0.65], [1.015, 1.065])},
        bodies={}, parts={}, floor={}, declarations=current_primitive_declarations(scene),
        rids={UR: "ur5e-3", BUFFER: "Buffer For Machined parts", M1: "M1", CONVEYOR: "Conveyor"}, part_geometry={},
        contact_layer_m=0.001)


@pytest.fixture
def session():
    pose = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), ARM["reference_pose"], strict=True))
    resource = {"scope": "resource"}
    return {
        "recovery_des_models": {
            UR: {"state_variables": {
                "resource_state": resource, "held_part": {**resource, "domain": PARTS},
                "gripper_state": {**resource, "domain": ["open", "closed"]},
                "resource_location": {**resource, "domain": ["home", "Buffer For Machined parts"]},
                "pose": {**resource, "type": "object", "fields": ["x", "y", "z"]},
                "task_ctx.part_name": {**resource, "private": True}, "part_location": {"scope": "part", "domain": []}}},
            BUFFER: {"state_variables": {"resource_state": resource, "zone_1_part": {**resource, "domain": PARTS}}},
            M1: {"state_variables": {"resource_state": resource, "part_name": {**resource, "domain": PARTS},
                                     "selected_program": {**resource, "domain": ["circle", "square"]}}},
            CONVEYOR: {"state_variables": {"resource_state": resource, "part_location.P1": {
                **resource, "domain": [None, "loading_position_1", "loading_position_2"]}}},
        },
        "symbolic_resources": {
            # Every capture records a UR5e's gripper_state, which its gripper primitives require.
            UR: {"resource_state": "picked", "held_part": "P1", "gripper_state": "closed", "resource_location": "home",
                 "current_pose": pose},
            # Every grounded row carries held_part, also for resources that do not declare it.
            BUFFER: {"resource_state": "idle", "zone_1_part": None, "held_part": None},
            M1: {"resource_state": "loaded", "part_name": "P2", "selected_program": "square", "held_part": None},
            CONVEYOR: {"resource_state": "idle", "part_location.P1": "loading_position_1", "held_part": None},
        },
        "symbolic_parts": {"P1": {"part_location": UR}, "P2": {"part_location": "M1"}},
        "base_symbolic_parts": {}, "outline_diagnostic_bypass": True, "check_ra_feasibility": True,
    }


def _release():
    return {"outline_id": "R1", "event_name": "release_into_zone_1", "resource_jid": UR, "part_name": "P1",
            "expected_end_state": {"held_part": None, "part_location": "Buffer For Machined parts"},
            "neighbor_end_state": {BUFFER: {"zone_1_part": "P1"}}}


def _event(jid, end, part=None):
    return {"outline_id": "R1", "event_name": "proposed", "resource_jid": jid, "expected_end_state": end,
            **({"part_name": part} if part else {})}


def test_a_release_into_a_free_zone_derives_one_release_and_reaches_the_zone(cell, session):
    result = check_event(_release(), session, {}, cell)
    assert result["feasibility_status"] == "FEASIBLE"
    # detach_part needs a recorded open gripper; release_part needs only the held part, so it is the producer.
    assert result["primitive_set"] == ["release_part"]
    assert {(row["check"], row["status"]) for row in result["capability"]} == {("place", "FEASIBLE"),
                                                                              ("capacity", "FEASIBLE")}


def test_a_zone_holding_another_part_refuses_the_release_with_that_fact(cell, session):
    session["symbolic_resources"][BUFFER]["zone_1_part"] = "P2"
    result = check_event(_release(), session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reasons"] == ["Buffer For Machined parts zone_1_part holds P2 in the projected state"]


def test_a_value_no_declared_effect_makes_is_refused_and_an_undeclared_one_named(cell, session):
    result = check_event(_event(UR, {"gripper_state": "half", "job_state": "paused"}), session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    reasons = " | ".join(result["reasons"])
    assert "open_gripper: it sets open, not half" in reasons and "close_gripper: it sets closed, not half" in reasons
    assert "job_state is not a declared variable of ur5e-3's current model" in reasons


def test_a_failed_precondition_is_refused_without_looking_for_a_primitive_that_fixes_it(cell, session):
    result = check_event(_event(UR, {"held_part": "P2"}, part="P2"), session, {}, cell)
    held = next(row for row in result["conditions"] if row["condition"] == "held_part")
    assert held["status"] == "INFEASIBLE"
    assert "grasp_part needs held_part = null; projected held_part is P1" in held["reason"]


def test_a_failed_gripper_refuses_its_release_but_not_an_arm_move(cell, session):
    session["symbolic_resources"][UR]["gripper_state"] = "failed"
    release = check_event(_release(), session, {}, cell)
    assert release["feasibility_status"] == "INFEASIBLE"
    assert "release_part needs gripper_state ≠ failed; projected gripper_state is failed" in " | ".join(release["reasons"])
    x, y, z = ARM["reference_pose"][:3]
    moved = check_event(_event(UR, {"pose": {"x": x, "y": y, "z": z + 0.05}}), session, {}, cell)
    assert moved["feasibility_status"] == "FEASIBLE"


@pytest.mark.parametrize("pose, status", [
    ({"x": 3.0, "y": 0.5, "z": 1.2}, "INFEASIBLE"),
    (dict(zip(("x", "y", "z"), ARM["reference_pose"][:3], strict=True)), "FEASIBLE"),
])
def test_a_stated_pose_is_judged_by_the_resource_s_own_chain(cell, session, pose, status):
    result = check_event(_event(UR, {"pose": pose}), session, {}, cell)
    assert result["feasibility_status"] == status
    if status == "FEASIBLE":
        assert result["primitive_set"] == ["move_cartesian"]
    else:
        assert any("beyond ur5e-3's kinematic chain" in reason for reason in result["reasons"])


@pytest.mark.parametrize("state, program, status, reason", [
    ("loaded", "circle", "FEASIBLE", None),
    ("loaded", "triangle", "INFEASIBLE", "program triangle is outside its declared values ['circle', 'square']"),
    ("running", "circle", "INFEASIBLE", "select_program needs resource_state ≠ running; projected resource_state is running"),
])
def test_a_machine_program_comes_from_its_declared_options_while_not_running(cell, session, state, program, status,
                                                                            reason):
    session["symbolic_resources"][M1]["resource_state"] = state
    result = check_event(_event(M1, {"selected_program": program}), session, {}, cell)
    assert result["feasibility_status"] == status
    if reason:
        assert any(reason in text for text in result["reasons"])


def test_state_names_unchanged_values_and_bookkeeping_need_no_derivation(cell, session):
    end = {"resource_state": "recovered", "task_ctx.part_name": "P1", "resource_location": "home"}
    result = check_event(_event(UR, end), session, {}, cell)
    assert result["feasibility_status"] == "FEASIBLE"
    assert [row["derivation"] for row in result["conditions"]] == ["free", "bookkeeping", "unchanged"]


@pytest.mark.parametrize("start, target, status", [
    ("loading_position_1", "loading_position_2", "FEASIBLE"),
    ("loading_position_2", "loading_position_1", "INFEASIBLE"),
])
def test_a_transport_carries_a_part_downstream_only(cell, session, start, target, status):
    session["symbolic_resources"][CONVEYOR]["part_location.P1"] = start
    session["symbolic_parts"]["P1"]["part_location"] = "Conveyor"
    result = check_event(_event(CONVEYOR, {"part_location.P1": target}, part="P1"), session, {}, cell)
    assert result["feasibility_status"] == status
    assert [row["check"] for row in result["capability"]] == ["transport"]


def test_the_hook_applies_the_first_feasible_candidate_and_returns_facts_when_none_is(cell, session, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_diagnostic,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _outline_feedback_for_prompt,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import (
        handle_validated_candidates,
    )

    applied = []

    async def handle_event(*, session, prepared, response, planner):
        applied.append(response["next_transition"]["event_name"])
        return "need_next_task", {}

    monkeypatch.setattr(multi_turn_diagnostic, "handle_event", handle_event)
    planner = SimpleNamespace(product_agent=SimpleNamespace(recorded_cell=cell))
    refused = _event(UR, {"gripper_state": "half"})
    refused["event_name"] = "half_open"
    response = {"thought": "t", "candidate_events": [refused, _release()], "selected_candidate_index": 0}
    decision, audit = asyncio.run(handle_validated_candidates(
        session=session, prepared={"offline_recovery": True}, response=deepcopy(response), planner=planner))
    assert (decision, applied) == ("need_next_task", ["release_into_zone_1"])
    assert (audit["selected_candidate_index"], audit["llm_selected_candidate_index"]) == (1, 0)
    assert [row["feasibility_status"] for row in audit["ra_feasibility"]] == ["INFEASIBLE", "FEASIBLE"]
    assert session["ra_feasibility_checks"][-1]["applied_candidate_index"] == 1

    response = {"thought": "t", "candidate_events": [refused], "selected_candidate_index": 0}
    decision, audit = asyncio.run(handle_validated_candidates(
        session=session, prepared={"offline_recovery": True}, response=response, planner=planner))
    findings = session["outline_validation_findings"]
    assert decision == "need_revision" and len(applied) == 1
    assert [row["constraint_code"] for row in findings] == ["ra_feasibility"]
    assert session["ra_feasibility_checks"][-1]["applied_candidate_index"] is None
    # The next prompt shows the refusal's facts as the check wrote them.
    assert _outline_feedback_for_prompt(findings[0], set())["reason"] == findings[0]["reason"]


@pytest.mark.parametrize("machine, status, primitive", [
    # Resuming a job another machine paused is refused by resume_job's own guard on the part's job record.
    ("M2", "INFEASIBLE", None),
    # Starting the job afresh on this machine is a new job, which start_job declares.
    ("M1", "FEASIBLE", "start_job"),
])
def test_a_machine_resumes_only_the_job_it_paused(cell, session, machine, status, primitive):
    paused = {"process": "trim", "result": "square", "status": "paused", "machine": "M2"}
    session["recovery_des_models"][M1]["state_variables"]["current_process"] = {"scope": "part", "domain": []}
    session["symbolic_parts"]["P2"]["current_process"] = paused
    result = check_event(_event(M1, {"current_process": {**paused, "status": "running", "machine": machine}},
                                part="P2"), session, {}, cell)
    assert result["feasibility_status"] == status
    if primitive:
        assert result["primitive_set"] == [primitive]
    else:
        assert "current_process matches {status: paused, process: trim, result: square, machine: M1}" in \
            " ".join(result["reasons"])


def test_a_location_condition_may_be_null_and_a_resource_may_be_named_as_a_part_location(cell, session):
    # Null says the resource is at no named place (its pose says where); a name is custody like a JID.
    result = check_event(_event(UR, {"resource_location": None, "part_location": "ur5e-3"}, part="P1"),
                         session, {}, cell)
    assert result["feasibility_status"] == "FEASIBLE"
    assert [row["derivation"] for row in result["conditions"]] == ["place", "place"]


def test_a_resource_without_motion_cannot_move_a_part(cell, session):
    session["recovery_des_models"][M1]["state_variables"]["part_location"] = {"scope": "part", "domain": []}
    result = check_event(_event(M1, {"part_name": None, "part_location": "Buffer For Machined parts"}, part="P2"),
                         session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    place = next(row for row in result["capability"] if row["check"] == "place")
    assert place["reason"] == "M1 declares no motion that moves parts"


def test_a_part_variable_another_resource_declares_is_a_variable_of_the_part(cell, session):
    session["recovery_des_models"][BUFFER]["state_variables"]["product_location"] = {"scope": "part", "domain": []}
    result = check_event(_event(UR, {"held_part": None, "product_location": "Buffer For Machined parts"}, part="P1"),
                         session, {}, cell)
    product = next(row for row in result["conditions"] if row["condition"] == "product_location")
    assert (product["derivation"], product["status"]) == ("place", "FEASIBLE")


def test_a_record_value_no_primitive_of_the_resource_makes_is_refused_as_no_place(cell, session):
    # A robot restating a part's job record: a record is never a place, and no robot primitive makes it.
    session["recovery_des_models"][M1]["state_variables"]["current_process"] = {"scope": "part", "domain": []}
    paused = {"process": "trim", "result": "square", "status": "paused", "machine": "M1"}
    result = check_event(_event(UR, {"current_process": paused}, part="P1"), session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert [row["derivation"] for row in result["conditions"]] == ["no_producer"]


def test_a_refusal_names_only_the_primitives_that_make_the_stated_value(cell, session):
    # M1 is loaded, not running: complete_job makes this record but needs a running job; the job commands do not.
    session["recovery_des_models"][M1]["state_variables"]["current_process"] = {"scope": "part", "domain": []}
    session["symbolic_parts"]["P2"]["current_process"] = {"process": "trim", "result": "square", "status": "paused",
                                                         "machine": "M1"}
    result = check_event(_event(M1, {"current_process": None}, part="P2"), session, {}, cell)
    assert result["reasons"] == ["complete_job needs resource_state = running; projected resource_state is loaded"]


def _job(session, state, selected, process=None):
    session["recovery_des_models"][M1]["state_variables"]["current_process"] = {"scope": "part", "domain": []}
    session["symbolic_resources"][M1].update(resource_state=state, selected_program=selected)
    session["symbolic_parts"]["P2"]["current_process"] = process


def _running(result):
    return _event(M1, {"resource_state": "running", "current_process": {
        "process": "trim", "result": result, "status": "running", "machine": "M1"}}, part="P2")


@pytest.mark.parametrize("selected, result, status", [("square", "square", "FEASIBLE"), ("circle", "circle", "FEASIBLE"),
                                                      ("circle", "square", "INFEASIBLE")])
def test_a_job_starts_only_with_a_result_its_selected_program_makes(cell, session, selected, result, status):
    _job(session, "loaded", selected)
    checked = check_event(_running(result), session, {}, cell)
    assert checked["feasibility_status"] == status
    if status == "FEASIBLE":
        assert checked["primitive_set"] == ["start_job"]
    else:
        assert checked["reasons"] == ["start_job needs selected_program = square; projected selected_program is circle"]


@pytest.mark.parametrize("selected, result, status", [("square", "square", "FEASIBLE"), ("circle", "circle", "FEASIBLE"),
                                                      ("circle", "square", "INFEASIBLE")])
def test_a_paused_job_resumes_only_with_the_program_that_makes_its_result(cell, session, selected, result, status):
    _job(session, "paused", selected, {"process": "trim", "result": result, "status": "paused", "machine": "M1"})
    checked = check_event(_running(result), session, {}, cell)
    assert checked["feasibility_status"] == status
    if status == "FEASIBLE":
        assert checked["primitive_set"] == ["resume_job"]
    else:
        reasons = " | ".join(checked["reasons"])
        assert "resume_job needs selected_program = square; projected selected_program is circle" in reasons
        # The variants for a circle job do not concern this square job.
        assert "result: circle" not in reasons


def test_a_job_start_is_unresolved_while_the_selected_program_is_unrecorded(cell, session):
    # A machine that has run no job records no selected program, so the start cannot be checked.
    _job(session, "loaded", None)
    del session["symbolic_resources"][M1]["selected_program"]
    checked = check_event(_running("square"), session, {}, cell)
    assert checked["feasibility_status"] == "NEEDS_CONTEXT"
    assert checked["reasons"] == ["start_job needs selected_program = square, which the projected state does not record"]


def test_a_machine_selects_only_programs_its_installed_tool_and_workholding_accept(cell, session):
    scene = deepcopy(cell.scene)
    machine = next(row for row in scene["machines"] if row["resource_id"] == "M1")
    machine["program_options"]["slot"] = {**deepcopy(machine["program_options"]["square"]), "tool": "slot_cutter"}
    checked = check_event(_event(M1, {"selected_program": "slot"}), session, {},
                          replace(cell, declarations=current_primitive_declarations(scene)))
    assert checked["feasibility_status"] == "INFEASIBLE"
    assert "program slot is outside its declared values ['circle', 'square']" in " | ".join(checked["reasons"])


# The fixture arm's gripper point at its recorded pose, the recorded TCP offset along its downward tool axis.
GRIPPER = (Rotation.from_quat(ARM["reference_pose"][3:]).apply([0.0, 0.0, 0.218]) + ARM["reference_pose"][:3]).tolist()
# A 4 x 4 x 50 mm part whose frame sits at its bottom.
P1_BOUNDS = [[-0.002, 0.002], [-0.002, 0.002], [0.0, 0.05]]
# Where the bench under the gripper supports P1 as the fixture grasp holds it, 0.03 m below the gripper point.
TOP = GRIPPER[2] - 0.03


def _box_body(name, center, size, static=True):
    bounds = [[center[axis] - size[axis] / 2, center[axis] + size[axis] / 2] for axis in range(3)]
    collision = {"id": f"{name}::link::box", "pose": [*center, 0.0, 0.0, 0.0, 1.0],
                 "geometry": {"type": "box", "size": size}, "bounds": bounds, "contact_min_depth": 0.0}
    return {name: {"static": static, "links": {"link": {"collisions": [collision]}}}}


def _bench(top, size=(0.4, 0.4), thickness=0.03, name="bench", x=GRIPPER[0]):
    # A recorded table top under the gripper point, its top face at ``top``.
    return _box_body(name, [x, GRIPPER[1], top - thickness / 2], [size[0], size[1], thickness])


def _grasp(held_at=0.03, part="P1"):
    # The grasp recorded at the failure: the part upright, its bottom ``held_at`` below the gripper point.
    upright = Rotation.from_quat(ARM["reference_pose"][3:]).inv().as_quat().tolist()
    return {"ur5e-3": {**ARM_RECORD, "attachment": {"part_name": part, "observed_attachment_complete": True,
                                                    "grasp_transform": [0.0, 0.0, 0.218 + held_at, *upright]}}}


ARM_RECORD = {"joint_names": ARM["configuration"]["joint_names"], "joint_positions": ARM["joint_positions"],
              "current_pose": ARM["reference_pose"], "tcp_offset_z": -0.218, "tcp_link": "ur5e_3_rg2_gripper_tcp"}


def _pose_place_cell(cell, bodies, regions=None, resources=None):
    # P1's own recorded body: its centre of mass halfway up and no contact depth, as the captures record parts.
    part = {"P1": {"static": False, "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                   "links": {"link": {"center_of_mass": [0.0, 0.0, 0.025],
                                      "collisions": [{"id": "P1::link::collision", "contact_min_depth": 0.0}]}}}}
    return replace(cell, bodies={**part, **bodies}, regions={**cell.regions, **(regions or {})},
                   resources={**cell.resources, **(_grasp() if resources is None else resources)},
                   part_geometry={"P1": {"local_bounds": P1_BOUNDS}})


def _release_here(**end):
    return _event(UR, {"held_part": None, "part_location": None,
                       "pose": dict(zip("xyz", ARM["reference_pose"][:3], strict=True)), **end}, part="P1")


def _place(result):
    return next(row for row in result["capability"] if row["check"] == "place")


def test_a_part_released_over_a_free_surface_rests_there(cell, session):
    result = check_event(_release_here(), session, {}, _pose_place_cell(cell, _bench(TOP)))
    assert result["feasibility_status"] == "FEASIBLE"
    place = _place(result)
    assert place["support"] == "bench::link::box"
    # As the recorded grasp holds it, it rests with its bottom on the bench right under the gripper.
    assert [place["rest_pose"][axis] for axis in "xyz"] == pytest.approx([*GRIPPER[:2], TOP], abs=1e-3)


def test_a_part_resting_on_two_abutting_surfaces_of_one_height_is_supported(cell, session):
    # The seam between the two benches runs under the part's centre. A machine's top of the same height
    # elsewhere in the cell does not hold the part.
    benches = {**_bench(TOP, name="left", x=GRIPPER[0] - 0.2), **_bench(TOP, name="right", x=GRIPPER[0] + 0.2),
               "M1": _bench(TOP, x=GRIPPER[0] + 2.0)["bench"]}
    assert check_event(_release_here(), session, {}, _pose_place_cell(cell, benches))["feasibility_status"] == "FEASIBLE"


@pytest.mark.parametrize("bodies, regions, reason", [
    ("none", None, "no recorded surface lies under P1 released at pose"),
    ("far", None, "would drop 270.0 mm onto bench::link::box"),
    ("inside", None, "would sit inside bench::link::box"),
    ("slab", None, "would sit inside bench::link::box"),
    ("narrow", None, "would rest past the edge of bench::link::box"),
    ("block", None, "would overlap block::link::box where it rests"),
    ("bench", {"Exit.surface": _region([GRIPPER[0] - 0.05, GRIPPER[0] + 0.05], [GRIPPER[1] - 0.05, GRIPPER[1] + 0.05],
                                       [TOP, TOP + 0.05])},
     "would rest in Exit.surface, which the event does not state as its location"),
    ("machine", None, "would rest in M1, which the event does not state as its location"),
])
def test_a_part_released_where_it_cannot_rest_is_refused_with_the_fact(cell, session, bodies, regions, reason):
    # A wall beside the part: it overlaps the part's footprint but not its centre, so it is no support.
    wall = _box_body("block", [GRIPPER[0] + 0.011, GRIPPER[1], (TOP + GRIPPER[2] + 0.05) / 2],
                     [0.02, 0.02, GRIPPER[2] + 0.05 - TOP])
    # The slab is thinner than the part and lies across its middle, so the part passes through it.
    bodies = {"none": {}, "far": _bench(GRIPPER[2] - 0.30), "inside": _bench(TOP + 0.01),
              "slab": _bench(TOP + 0.02, thickness=0.005), "narrow": _bench(TOP, (0.003, 0.4)),
              "block": {**_bench(TOP), **wall}, "bench": _bench(TOP), "machine": {"M1": _bench(TOP)["bench"]}}[bodies]
    result = check_event(_release_here(), session, {}, _pose_place_cell(cell, bodies, regions))
    assert result["feasibility_status"] == "INFEASIBLE"
    assert reason in _place(result)["reason"]


@pytest.mark.parametrize("held_at, status", [(0.03, "FEASIBLE"), (0.0295, "INFEASIBLE"), (0.0305, "INFEASIBLE"),
                                             (0.025, "INFEASIBLE"), (0.0, "INFEASIBLE"), (0.05, "INFEASIBLE")])
def test_a_recorded_grasp_decides_where_the_released_part_s_bottom_is(cell, session, held_at, status):
    # The bench top is 0.03 m below the gripper point. With the recorded contact depths of zero, as the captures
    # record them, only a part whose bottom meets the top rests there: 0.5 mm above it would drop (settling is
    # not modelled) and 0.5 mm into it sits inside.
    result = check_event(_release_here(), session, {}, _pose_place_cell(cell, _bench(TOP), resources=_grasp(held_at)))
    assert result["feasibility_status"] == status


def test_a_part_that_would_land_tilted_leaves_the_release_unresolved(cell, session):
    # The tool turns 10 degrees about x, and the part with it; the bench top meets the part's lowest corner.
    tool = Rotation.from_euler("x", 10, degrees=True) * Rotation.from_quat(ARM["reference_pose"][3:])
    origin = tool.apply([0.0, 0.0, 0.248]) + ARM["reference_pose"][:3]
    corners = Rotation.from_euler("x", 10, degrees=True).apply(
        [[x, y, z] for x in P1_BOUNDS[0] for y in P1_BOUNDS[1] for z in P1_BOUNDS[2]])
    end = {"pose": {**dict(zip("xyz", ARM["reference_pose"][:3], strict=True)),
                    **dict(zip(("qx", "qy", "qz", "qw"), tool.as_quat().tolist(), strict=True))}}
    result = check_event(_release_here(**end), session, {},
                         _pose_place_cell(cell, _bench(origin[2] + corners[:, 2].min())))
    assert (_place(result)["status"], result["feasibility_status"]) == ("NEEDS_CONTEXT", "NEEDS_CONTEXT")
    assert "would tip over, its recorded centre of mass outside where it touches down, on bench::link::box" in \
        _place(result)["reason"]


@pytest.mark.parametrize("resources, accepted", [
    ({}, []),
    (_grasp(part="P2"), []),
    # A release and a re-grasp, the second stated without the part's name.
    (None, [{"part_name": "P1", "expected_start_state": {"held_part": "P1"}, "expected_end_state": {"held_part": None}},
            {"expected_start_state": {"held_part": None}, "expected_end_state": {"held_part": "P1"}}]),
])
def test_a_release_is_unresolved_while_no_grasp_of_the_part_is_recorded(cell, session, resources, accepted):
    # How the part hangs below the gripper is recorded only for a part held since the failure.
    session["accepted_outline_prefix"] = accepted
    result = check_event(_release_here(), session, {}, _pose_place_cell(cell, _bench(TOP), resources=resources))
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert _place(result)["reason"] == "the capture records no grasp of P1 by ur5e-3 that still holds"


def test_restating_the_held_part_keeps_the_recorded_grasp(cell, session):
    session["accepted_outline_prefix"] = [{"part_name": "P1", "expected_start_state": {"held_part": "P1"},
                                           "expected_end_state": {"held_part": "P1", "pose": {"x": 0.0}}}]
    result = check_event(_release_here(), session, {}, _pose_place_cell(cell, _bench(TOP)))
    assert result["feasibility_status"] == "FEASIBLE"


def test_a_release_pose_without_a_height_is_named_as_such(cell, session):
    end = {"held_part": None, "part_location": None, "pose": {"x": GRIPPER[0], "y": GRIPPER[1]}}
    result = check_event(_event(UR, end, part="P1"), session, {}, _pose_place_cell(cell, _bench(TOP)))
    assert _place(result)["reason"].endswith("does not state x, y and z")


def test_taking_a_part_and_setting_it_down_at_a_pose_in_one_event_is_checked_as_both(cell, session):
    # The robot holds nothing; the part sits in the Buffer, and one event moves it to an unnamed spot.
    session["symbolic_resources"][UR].update(held_part=None, gripper_state="open")
    session["symbolic_parts"]["P1"] = {"part_location": "Buffer For Machined parts"}
    result = check_event(_release_here(), session, {}, _pose_place_cell(cell, _bench(TOP), resources={}))
    assert [row["check"] for row in result["capability"]] == ["pick", "place"]
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert _place(result)["reason"] == "the capture records no grasp of P1 by ur5e-3 that still holds"


def test_a_part_resting_where_another_part_is_recorded_leaves_the_release_unresolved(cell, session):
    session["symbolic_parts"]["P2"] = {"part_location": None, "observed_pose": {
        "x": GRIPPER[0] + 0.002, "y": GRIPPER[1], "z": TOP, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}}
    pose_cell = _pose_place_cell(cell, _bench(TOP))
    pose_cell = replace(pose_cell, part_geometry={**pose_cell.part_geometry, "P2": {"local_bounds": P1_BOUNDS}})
    result = check_event(_release_here(), session, {}, pose_cell)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert "would rest within the recorded bounds of P2, whose shapes are not checked" in _place(result)["reason"]


def test_a_release_naming_the_place_the_part_still_records_is_checked_as_a_place(cell, session):
    # The pick stated only the held part, so the part's location still names where it was taken from.
    session["symbolic_parts"]["P1"]["part_location"] = "Buffer For Machined parts"
    result = check_event(_event(UR, {"held_part": None, "part_location": "Buffer For Machined parts"}, part="P1"),
                         session, {}, cell)
    place = _place(result)
    assert (place["status"], place["place"]) == ("FEASIBLE", "Buffer For Machined parts")


def _picks_p1(session, checks, location=None):
    session["symbolic_resources"][UR].update(held_part=None, gripper_state="open")
    session["symbolic_resources"][CONVEYOR]["part_location.P1"] = None
    session["symbolic_parts"]["P1"] = {"part_location": location, "part_holder_resource_jid": None}
    session["ra_feasibility_checks"] = checks
    return _event(UR, {"held_part": "P1", "part_location": "ur5e-3"}, part="P1")


def _applied(*capability):
    return {"applied_candidate_index": 0, "results": [{"candidate_index": 0, "capability": list(capability)}]}


# A release that stated no location leaves the part's location naming the robot that released it.
@pytest.mark.parametrize("location", [None, UR])
def test_a_part_left_at_a_pose_place_is_picked_from_where_it_rests(cell, session, location):
    rest = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), [*GRIPPER[:2], TOP, 0.0, 0.0, 0.0, 1.0], strict=True))
    placed = {"check": "place", "part": "P1", "status": "FEASIBLE", "rest_pose": rest}
    result = check_event(_picks_p1(session, [_applied(placed)], location), session, {},
                         _pose_place_cell(cell, _bench(TOP)))
    assert _place_or_pick(result) == ("pick", "FEASIBLE")
    # A later applied event that took the part again leaves no rest pose behind.
    picked = {"check": "pick", "part": "P1", "status": "FEASIBLE"}
    result = check_event(_picks_p1(session, [_applied(placed), _applied(picked)], location), session, {},
                         _pose_place_cell(cell, _bench(TOP)))
    assert _place_or_pick(result) == ("pick", "NEEDS_CONTEXT")


def test_a_part_whose_recorded_location_names_an_empty_gripper_is_picked_where_the_failure_recorded_it(cell, session):
    # At the failure the part's location already names the robot, which holds nothing; nothing has moved it since.
    session["symbolic_resources"][UR].update(held_part=None, gripper_state="open")
    session["symbolic_parts"]["P1"] = {"part_location": UR}
    session["base_symbolic_parts"] = {"P1": {"part_location": UR}}
    recorded = replace(cell, parts={"P1": {"pose": [*GRIPPER[:2], GRIPPER[2] - 0.05, 0.0, 0.0, 0.0, 1.0]}})
    result = check_event(_event(UR, {"held_part": "P1", "part_location": "ur5e-3"}, part="P1"), session, {}, recorded)
    assert _place_or_pick(result) == ("pick", "FEASIBLE")


def test_a_part_at_a_place_whose_row_carries_an_empty_held_part_is_picked_from_that_place(cell, session):
    # Every resource row may carry held_part; only a resource that declares it grips parts. The place records
    # the part as present, with no slot.
    session["recovery_des_models"][BUFFER]["state_variables"]["inventory.P1"] = {"scope": "resource"}
    session["symbolic_resources"][UR].update(held_part=None, gripper_state="open")
    session["symbolic_resources"][BUFFER].update({"held_part": None, "inventory.P1": True})
    session["symbolic_parts"]["P1"] = {"part_location": "Buffer For Machined parts"}
    result = check_event(_event(UR, {"held_part": "P1", "part_location": "ur5e-3"}, part="P1"), session, {}, cell)
    assert _place_or_pick(result) == ("pick", "FEASIBLE")


def test_a_part_put_into_a_slot_without_a_stated_location_is_picked_from_that_slot(cell, session):
    # The release stated only the Buffer's slot, so the part's location still names the robot.
    session["symbolic_resources"][UR].update(held_part=None, gripper_state="open")
    session["symbolic_resources"][BUFFER]["zone_1_part"] = "P1"
    session["symbolic_resources"][CONVEYOR]["part_location.P1"] = None
    session["symbolic_parts"]["P1"] = {"part_location": UR, "part_holder_resource_jid": None}
    result = check_event(_event(UR, {"held_part": "P1", "part_location": "ur5e-3"}, part="P1"), session, {}, cell)
    pick = next(row for row in result["capability"] if row["check"] == "pick")
    assert (pick["status"], pick["place"], pick["slot"]) == ("FEASIBLE", "Buffer For Machined parts", "zone_1_part")


def _place_or_pick(result):
    row = next(row for row in result["capability"] if row["check"] in ("pick", "place"))
    return row["check"], row["status"]
