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
UR, BUFFER, M1, CONVEYOR, KMR = (f"recovery-resource-{index}@localhost" for index in (3, 10, 5, 9, 8))
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
            # Every capture records a UR5e's gripper_state, which its gripper primitives require, and every
            # resource's availability, which all its primitives require.
            UR: {"resource_state": "picked", "held_part": "P1", "gripper_state": "closed", "resource_location": "home",
                 "current_pose": pose, "availability": "available"},
            # Every grounded row carries held_part, also for resources that do not declare it.
            BUFFER: {"resource_state": "idle", "zone_1_part": None, "held_part": None, "availability": "available"},
            M1: {"resource_state": "loaded", "part_name": "P2", "selected_program": "square", "held_part": None,
                 "availability": "available"},
            CONVEYOR: {"resource_state": "idle", "part_location.P1": "loading_position_1", "held_part": None,
                       "availability": "available"},
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


def test_a_release_that_states_held_part_but_not_part_location_is_refused(cell, session):
    # The robot's model declares both, and the release changes both; unstated, P1 would stay at the robot.
    event = _release()
    del event["expected_end_state"]["part_location"]
    result = check_event(event, session, {}, cell)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["reasons"] == [f"the event changes {cell.rids[UR]}'s held_part from P1 but does not state its "
                                 "part_location"]
    assert check_event(_release(), session, {}, cell)["feasibility_status"] == "FEASIBLE"


@pytest.mark.parametrize("name, values", [
    # A part written as if it were a resource, and a place written by a name that is no resource.
    ("P1", {"part_location": "Buffer For Machined parts"}),
    ("Buffer zone 1", {"zone_1_part": "P1"}),
])
def test_a_neighbor_that_is_no_resource_refuses_the_event_with_that_fact(cell, session, name, values):
    event = _release()
    event["neighbor_end_state"][name] = values
    result = check_event(event, session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reasons"] == [f"neighbor_end_state names {name}, which is not a resource, so its values there "
                                 "are not read"]
    # Without the entry the same release passes.
    del event["neighbor_end_state"][name]
    assert check_event(event, session, {}, cell)["feasibility_status"] == "FEASIBLE"


def test_a_dock_that_reaches_an_area_is_the_witness_before_a_closer_grid_pose(cell, session):
    from cais_spade_llm.recovery_framework.outline_draft import _navigation_floor
    from cais_spade_llm.recovery_framework.ra_feasibility import _Check

    kmr_arm = json.loads((Path(__file__).parent / "fixtures/recorded_arm_chains.json").read_text())["arms"]["KMR"]
    x, y, _, _, _, yaw = cell.scene["KMR"]["initial_pose"]
    kmr_cell = replace(cell, floor=_navigation_floor(), rids={**cell.rids, KMR: "KMR"},
                       models={**cell.models, "KMR": {"configuration": kmr_arm["configuration"]}},
                       resources={**cell.resources, "KMR": {"base_pose": [x, y, 0.0, *Rotation.from_euler("z", yaw).as_quat()]}})
    session["recovery_des_models"][KMR] = {"state_variables": {"resource_state": {"scope": "resource"},
                                                               "held_part": {"scope": "resource", "domain": PARTS}}}
    session["symbolic_resources"][KMR] = {"resource_state": "idle", "held_part": "P1", "availability": "available"}
    m2 = next(row for row in cell.scene["machines"] if row["resource_id"] == "M2")
    tx, ty, tz = m2["KMR_access"]["target_pose"][:3]
    area = {"frame": "world", "bounds": [[tx - 0.02, tx + 0.02], [ty - 0.02, ty + 0.02], [tz, tz + 0.04]], "capacity": 1}
    reached = _Check(_event(KMR, {"held_part": None}, part="P1"), session, {}, kmr_cell).reach_area(area, "M2 workholding")
    # Nominal production loads M2 from this dock; a finer grid has free poses nearer M2 that reach it as well.
    assert reached["status"] == "FEASIBLE"
    assert reached["witness"]["base_pose"] == [*m2["KMR_docking_pose"][:2], m2["KMR_docking_pose"][5]]


def test_a_place_that_does_not_state_the_held_part_is_refused(cell, session):
    # Unstated, the release would go unchecked, here on a failed gripper, while P1 leaves ur5e-3's gripper.
    session["symbolic_resources"][UR]["gripper_state"] = "failed"
    event = _release()
    del event["expected_end_state"]["held_part"]
    result = check_event(event, session, {}, cell)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert "the event moves P1 from ur5e-3's gripper but does not state its held_part" in result["reasons"]


def test_a_zone_holding_another_part_refuses_the_release_with_that_fact(cell, session):
    session["symbolic_resources"][BUFFER]["zone_1_part"] = "P2"
    result = check_event(_release(), session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reasons"] == ["Buffer For Machined parts zone_1_part holds P2 in the projected state"]


def test_a_value_no_declared_effect_makes_is_refused_and_an_undeclared_one_named(cell, session):
    result = check_event(_event(UR, {"gripper_state": "half", "job_state": "paused"}), session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    reasons = " | ".join(result["reasons"])
    assert "release_part: it sets open, not half" in reasons and "grasp_part: it sets closed, not half" in reasons
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
        # A refused producer says which part of C_rho refused it: the primitive that makes the stated pose fails
        # on the continuous pose constraint, the others on their effects.
        refused = [producer for row in result["conditions"] for producer in row.get("producers", [])
                   if producer["status"] != "FEASIBLE"]
        assert {producer["refused_by"] for producer in refused if producer.get("makes_value")} == {"pose_constraint"}
        assert {producer["refused_by"] for producer in refused if not producer.get("makes_value")} <= {"effect"}


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


def test_a_new_state_name_and_a_value_that_already_holds_need_nothing(cell, session):
    # No ur5e-3 primitive declares "recovered", so it names the new state; home is where ur5e-3 already is.
    result = check_event(_event(UR, {"resource_state": "recovered", "resource_location": "home"}), session, {}, cell)
    assert result["feasibility_status"] == "FEASIBLE"
    assert [(row["status"], "producers" in row, "found_by" in row) for row in result["conditions"]] == [
        ("FEASIBLE", False, False), ("FEASIBLE", False, False)]
    assert result["conditions"][1]["before"] == result["conditions"][1]["after"]


def test_a_private_field_nothing_sets_is_refused(cell, session):
    result = check_event(_event(UR, {"task_ctx.part_name": "P1"}), session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reasons"] == ["no ur5e-3 primitive sets ur5e-3's task_ctx.part_name to P1"]


def test_a_device_that_reports_itself_unavailable_runs_none_of_its_primitives(cell, session):
    # Every primitive needs its resource available; the device's own report says M1 is not.
    _job(session, "loaded", "square")
    session["symbolic_resources"][M1]["availability"] = "unavailable"
    checked = check_event(_running("square"), session, {}, cell)
    assert checked["feasibility_status"] == "INFEASIBLE"
    assert "availability = available; projected availability is unavailable" in " | ".join(checked["reasons"])
    # A name none of its primitives declares needs no primitive, but the event is still M1's own, so it needs
    # what all of M1's primitives need.
    label = check_event(_event(M1, {"resource_state": "idle"}, part="P2"), session, {}, cell)
    assert [(row["status"], "producers" in row) for row in label["conditions"]] == [("FEASIBLE", False)]
    assert label["feasibility_status"] == "INFEASIBLE"
    assert label["reasons"] == ["M1's primitives need availability = available; projected availability is unavailable"]


def test_a_device_that_reports_a_fault_but_stays_available_can_change_its_own_label(cell, session):
    session["symbolic_resources"][UR]["resource_state"] = "failed"
    result = check_event(_event(UR, {"resource_state": "idle"}), session, {}, cell)
    assert result["feasibility_status"] == "FEASIBLE"
    assert result["capability"] == []


def test_a_derived_primitive_states_the_unavailable_device_once(cell, session):
    session["symbolic_resources"][M1]["availability"] = "unavailable"
    result = check_event(_event(M1, {"selected_program": "circle"}), session, {}, cell)
    assert result["capability"] == []
    assert result["reasons"] == ["select_program needs availability = available; projected availability is unavailable"]


def test_an_unavailable_transport_carries_no_part(cell, session):
    # The belt's move is judged by its capability, which stands for its primitives and so needs it available.
    session["symbolic_resources"][CONVEYOR]["availability"] = "unavailable"
    session["symbolic_parts"]["P1"]["part_location"] = "Conveyor"
    result = check_event(_event(CONVEYOR, {"part_location.P1": "loading_position_2"}, part="P1"), session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["capability"][-1] == {
        "check": "primitives", "status": "INFEASIBLE", "refused_by": "preconditions",
        "reason": "Conveyor's primitives need availability = available; projected availability is unavailable"}


def test_a_belt_order_alone_is_no_slot_the_part_moves_to(cell, session):
    # part_order names no slot, so stating it alone is no transport the belt's reach decides.
    session["recovery_des_models"][CONVEYOR]["state_variables"]["part_order.P1"] = {
        "scope": "resource", "type": ["integer", "null"], "minimum": 0, "domain": [None], "domain_complete": False}
    session["symbolic_resources"][CONVEYOR]["part_order.P1"] = None
    session["symbolic_parts"]["P1"]["part_location"] = "Conveyor"
    result = check_event(_event(CONVEYOR, {"part_order.P1": 1}, part="P1"), session, {}, cell)
    assert "transport" not in [row["check"] for row in result["capability"]]


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


def test_a_resource_may_be_named_by_its_jid_or_its_name(cell, session):
    # P1's location records ur5e-3's JID; its name states the same value.
    result = check_event(_event(UR, {"part_location": "ur5e-3"}, part="P1"), session, {}, cell)
    assert result["feasibility_status"] == "FEASIBLE"
    assert result["conditions"][0]["before"] == UR


@pytest.mark.parametrize("pose, status", [
    # The stated pose is in no declared place's recorded area, so the robot is at no named place.
    (dict(zip(("x", "y", "z"), ARM["reference_pose"][:3], strict=True)), "FEASIBLE"),
    # Inside the buffer's zone 1, the robot is at the buffer, so null is refused.
    ({"x": -0.68, "y": 0.5, "z": 1.04}, "INFEASIBLE"),
])
def test_a_null_location_is_where_the_stated_pose_puts_the_resource(cell, session, pose, status):
    # The check locates a stated pose as the projection does: by the declared places' recorded areas.
    prepared = {"diagnostic_regions": {"Buffer For Machined parts.zone_1": cell.regions["Buffer For Machined parts.zone_1"]},
                "recovery_resources": {UR: {"static_capabilities": {"reachability": ["home", "Buffer For Machined parts"]}}}}
    result = check_event(_event(UR, {"resource_location": None, "pose": pose}), session, prepared, cell)
    location = next(row for row in result["conditions"] if row["condition"] == "resource_location")
    assert (location["status"], location["found_by"]) == (status, "pose")
    if status == "INFEASIBLE":
        assert location["reason"] == "the stated pose puts ur5e-3 at Buffer For Machined parts, not null"


def test_a_null_location_without_a_stated_pose_is_set_by_nothing(cell, session):
    result = check_event(_event(UR, {"resource_location": None}), session, {}, cell)
    assert result["reasons"] == ["no ur5e-3 primitive sets ur5e-3's resource_location to null"]


def test_a_resource_without_motion_cannot_move_a_part(cell, session):
    session["recovery_des_models"][M1]["state_variables"]["part_location"] = {"scope": "part", "domain": []}
    result = check_event(_event(M1, {"part_name": None, "part_location": "Buffer For Machined parts"}, part="P2"),
                         session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    place = next(row for row in result["capability"] if row["check"] == "place")
    assert place["reason"] == "M1 declares no motion that moves parts"


def test_a_part_variable_another_resource_declares_is_a_variable_of_the_part(cell, session):
    # Nothing sets P1's product_location: no ur5e-3 primitive, and no check of where the release puts P1.
    session["recovery_des_models"][BUFFER]["state_variables"]["product_location"] = {"scope": "part", "domain": []}
    result = check_event(_event(UR, {"held_part": None, "product_location": "Buffer For Machined parts"}, part="P1"),
                         session, {}, cell)
    product = next(row for row in result["conditions"] if row["condition"] == "product_location")
    assert (product["status"], product["reason"]) == (
        "INFEASIBLE", "no ur5e-3 primitive sets P1's product_location to Buffer For Machined parts")


def test_a_record_value_no_primitive_of_the_resource_makes_is_refused(cell, session):
    # A robot restating a part's job record: no robot primitive makes it, and no check finds it.
    session["recovery_des_models"][M1]["state_variables"]["current_process"] = {"scope": "part", "domain": []}
    paused = {"process": "trim", "result": "square", "status": "paused", "machine": "M1"}
    result = check_event(_event(UR, {"current_process": paused}, part="P1"), session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reasons"] == ["no ur5e-3 primitive sets P1's current_process to "
                                 "{process: trim, result: square, status: paused, machine: M1}"]


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
        assert checked["reasons"] == ["start_job (result square) needs selected_program = square; "
                                      "projected selected_program is circle"]


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
    # The circle alternative cannot make the stated square job, so its unrecorded program is not stated.
    assert checked["reasons"] == ["start_job (result square) needs selected_program = square, "
                                  "which the projected state does not record"]


def test_a_machine_reaches_a_label_its_job_commands_produce_only_through_them(cell, session):
    # M1's job commands produce running, so a paused M1 runs again only by a start or a resume. Each start_job
    # alternative is named by the result it makes.
    _job(session, "paused", "square")
    running = check_event(_event(M1, {"resource_state": "running"}, part="P2"), session, {}, cell)
    assert running["feasibility_status"] == "INFEASIBLE"
    assert running["reasons"] == ["start_job (result circle) needs resource_state = loaded; projected resource_state "
                                  "is paused; selected_program = circle; projected selected_program is square; "
                                  "start_job (result square) needs resource_state = loaded; projected resource_state "
                                  "is paused"]
    # No job command declares loaded, so it names M1's new state and needs no primitive.
    loaded = check_event(_event(M1, {"resource_state": "loaded"}, part="P2"), session, {}, cell)
    assert [(row["status"], "producers" in row) for row in loaded["conditions"]] == [("FEASIBLE", False)]


def test_a_label_an_uninterpreted_effect_may_set_stays_derived(cell, session):
    # An effect form the checker cannot read is not known to leave the label alone, so the label needs it.
    declarations = deepcopy(cell.declarations)
    for contract in declarations["M1"]["pause_job"]:
        contract["effects"]["resource_state"] = {"set_from_step": "label"}
    _job(session, "running", "square", {"process": "trim", "result": "square", "status": "running", "machine": "M1"})
    checked = check_event(_event(M1, {"resource_state": "paused"}, part="P2"), session, {},
                          replace(cell, declarations=declarations))
    assert checked["feasibility_status"] == "NEEDS_CONTEXT"
    assert ["producers" in row for row in checked["conditions"]] == [True]


def test_a_refusal_leaves_out_an_alternative_another_stated_condition_rules_out(cell, session):
    # The event states a square job, which start_job's circle alternative cannot make, so the refusal of the
    # running label leaves out the circle alternative's facts as well.
    _job(session, "paused", "square")
    checked = check_event(_running("square"), session, {}, cell)
    assert checked["feasibility_status"] == "INFEASIBLE"
    assert checked["reasons"] == ["start_job (result square) needs resource_state = loaded; "
                                  "projected resource_state is paused"]
    job = next(row for row in checked["conditions"] if row["condition"] == "current_process")
    starts = {producer["alternative"]: producer["refused_by"] for producer in job["producers"]
              if producer["primitive"] == "start_job"}
    assert sorted(starts.values()) == ["effect", "preconditions"]


def test_a_job_command_that_names_no_part_is_unresolved_on_the_event_not_on_v(cell, session):
    # pause_job binds the job's part from the event; M1's part is recorded, the event's is not stated.
    _job(session, "running", "square", {"process": "trim", "result": "square", "status": "running", "machine": "M1"})
    checked = check_event(_event(M1, {"resource_state": "paused"}), session, {}, cell)
    assert checked["feasibility_status"] == "NEEDS_CONTEXT"
    assert checked["reasons"] == ["pause_job needs part_name = the event's part_name, and the event names none; "
                                  "current_process matches {machine: M1} on the event's part, and the event names "
                                  "no part"]


def test_a_state_name_no_primitive_declares_names_the_new_state_for_robots_and_neighbours(cell, session):
    # No ur5e-3 primitive declares waiting, and no M1 primitive declares loaded.
    robot = check_event(_event(UR, {"resource_state": "waiting"}), session, {}, cell)
    assert [(row["status"], "producers" in row) for row in robot["conditions"]] == [("FEASIBLE", False)]
    session["symbolic_resources"][M1].update(part_name=None, resource_state="idle")
    loaded = check_event({**_event(UR, {"held_part": None, "part_location": "M1"}, part="P1"),
                          "neighbor_end_state": {M1: {"resource_state": "loaded", "part_name": "P1"}}}, session, {}, cell)
    state = next(row for row in loaded["conditions"] if row["condition"] == "resource_state")
    assert (state["status"], "producers" in state) == ("FEASIBLE", False)


def test_a_robot_state_name_its_primitives_declare_as_current_state_comes_only_from_them(cell, session):
    # grasp_part declares picked as current_state, ur5e-3's resource_state; it needs an empty gripper.
    session["symbolic_resources"][UR]["resource_state"] = "idle"
    result = check_event(_event(UR, {"resource_state": "picked"}), session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert [producer["primitive"] for producer in result["conditions"][0]["producers"]
            if producer.get("makes_value")] == ["grasp_part"]
    assert result["reasons"] == ["grasp_part needs held_part = null; projected held_part is P1"]


def test_a_neighbour_value_one_of_its_own_primitives_declares_needs_the_neighbour_to_act(cell, session):
    result = check_event({**_release(), "neighbor_end_state": {BUFFER: {"zone_1_part": "P1"},
                                                                M1: {"resource_state": "running"}}},
                         session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reasons"] == ["M1's start_job or resume_job sets resource_state to running, "
                                 "and M1 does not act in this event"]


def test_an_event_of_a_resource_that_declares_no_primitives_is_refused(cell, session):
    exit_jid = "recovery-resource-12@localhost"
    session["recovery_des_models"][exit_jid] = {"state_variables": {"resource_state": {"scope": "resource"}}}
    session["symbolic_resources"][exit_jid] = {"resource_state": "occupied", "held_part": None}
    result = check_event(_event(exit_jid, {"resource_state": "empty"}), session, {},
                         replace(cell, rids={**cell.rids, exit_jid: "Exit"}))
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reasons"] == ["Exit declares no primitives"]


def test_a_part_location_other_than_where_the_place_check_puts_the_part_is_refused(cell, session):
    event = _release()
    event["expected_end_state"]["part_location"] = "M1"
    result = check_event(event, session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reasons"] == ["the place check finds P1's part_location Buffer For Machined parts, not M1"]
    zone = next(row for row in check_event(_release(), session, {}, cell)["conditions"]
                if row["condition"] == "zone_1_part")
    assert (zone["status"], zone["found_by"]) == ("FEASIBLE", "place")


def test_a_pick_empties_the_slot_the_part_was_in_and_leaves_it_with_the_robot(cell, session):
    session["symbolic_resources"][UR].update(held_part=None, resource_state="idle", gripper_state="open")
    session["symbolic_resources"][BUFFER]["zone_1_part"] = "P2"
    session["symbolic_parts"]["P2"]["part_location"] = "Buffer For Machined parts"
    result = check_event({**_event(UR, {"held_part": "P2", "part_location": "ur5e-3"}, part="P2"),
                          "neighbor_end_state": {BUFFER: {"zone_1_part": None}}}, session, {}, cell)
    assert result["feasibility_status"] == "FEASIBLE"
    assert {row["condition"]: row.get("found_by") for row in result["conditions"]} == {
        "held_part": None, "part_location": "pick", "zone_1_part": "pick"}


def test_a_pick_empties_every_value_that_records_the_part_where_it_rests(cell, session):
    # P1 rests on the belt, recorded by its slot and its order number 0; the place's name is no resource name.
    session["recovery_des_models"][CONVEYOR]["state_variables"]["part_order.P1"] = {
        "scope": "resource", "type": ["integer", "null"], "minimum": 0}
    session["symbolic_resources"][CONVEYOR]["part_order.P1"] = 0
    session["symbolic_resources"][UR].update(held_part=None, resource_state="idle", gripper_state="open")
    session["symbolic_parts"]["P1"]["part_location"] = "Conveyor loading position 1"
    result = check_event({**_event(UR, {"held_part": "P1", "part_location": "ur5e-3"}, part="P1"),
                          "neighbor_end_state": {CONVEYOR: {"part_location.P1": None, "part_order.P1": None}}},
                         session, {}, cell)
    pick = next(row for row in result["capability"] if row["check"] == "pick")
    # Each value takes the pick check's status: the stated place has no recorded geometry to reach.
    assert {row["condition"]: (row["status"], row.get("found_by")) for row in result["conditions"]
            if row["condition"] != "held_part"} == {"part_location": (pick["status"], "pick"),
                                                     "part_location.P1": (pick["status"], "pick"),
                                                     "part_order.P1": (pick["status"], "pick")}


def test_a_value_a_check_finds_overrides_a_restated_old_value(cell, session):
    # The pick takes P2 into ur5e-3's gripper, so restating P2's old location contradicts it.
    session["symbolic_resources"][UR].update(held_part=None, resource_state="idle", gripper_state="open")
    session["symbolic_resources"][BUFFER]["zone_1_part"] = "P2"
    session["symbolic_parts"]["P2"]["part_location"] = "Buffer For Machined parts"
    result = check_event(_event(UR, {"held_part": "P2", "part_location": "Buffer For Machined parts"}, part="P2"),
                         session, {}, cell)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reasons"] == ["the pick check finds P2's part_location ur5e-3, not Buffer For Machined parts"]


def test_a_part_put_where_a_flag_records_it_is_found_by_the_place_check(cell, session):
    session["recovery_des_models"][CONVEYOR]["state_variables"]["inventory.P1"] = {
        "scope": "resource", "domain": [False, True]}
    session["symbolic_resources"][CONVEYOR]["inventory.P1"] = False
    result = check_event({**_event(UR, {"held_part": None, "part_location": "Conveyor"}, part="P1"),
                          "neighbor_end_state": {CONVEYOR: {"inventory.P1": True}}}, session, {}, cell)
    flag = next(row for row in result["conditions"] if row["condition"] == "inventory.P1")
    assert flag["found_by"] == "place"


def test_a_part_put_at_a_resource_needs_the_variable_that_records_it(cell, session):
    # Unstated, nothing records P1 at the buffer, and zone 1's capacity goes unchecked.
    result = check_event(_event(UR, {"held_part": None, "part_location": "Buffer For Machined parts"}, part="P1"),
                         session, {}, cell)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert ("the event puts P1 at Buffer For Machined parts but states none of Buffer For Machined parts's "
            "variables that can record it") in result["reasons"]


def test_a_transport_to_a_slot_with_no_recorded_area_is_unresolved(cell, session):
    session["symbolic_parts"]["P1"]["part_location"] = "Conveyor"
    result = check_event(_event(CONVEYOR, {"part_location.P1": "bogus_slot"}, part="P1"), session, {}, cell)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["reasons"] == ["bogus_slot names no recorded area of Conveyor"]


def test_a_belt_event_that_clears_its_own_slot_is_unresolved_rather_than_an_error(cell, session):
    session["symbolic_parts"]["P1"]["part_location"] = "Conveyor"
    result = check_event({**_event(CONVEYOR, {"part_location.P1": None}, part="P1"),
                          "neighbor_end_state": {BUFFER: {"zone_1_part": "P1"}}}, session, {}, cell)
    assert result["feasibility_status"] != "FEASIBLE"
    assert "null names no recorded area of Conveyor" in result["reasons"]


def test_a_part_location_stated_as_a_mapping_is_unresolved_rather_than_an_error(cell, session):
    # The belt's per-part slot written nested, {part: slot}, also as the robot's own part_location.
    stated = {"P1": "loading_position_1"}
    result = check_event({**_event(UR, {"held_part": None, "resource_location": "Conveyor", "part_location": stated},
                                   part="P1"),
                          "neighbor_end_state": {CONVEYOR: {"part_location": stated}}}, session, {}, cell)
    assert result["feasibility_status"] != "FEASIBLE"
    place = next(row for row in result["capability"] if row["check"] == "place")
    assert (place["place"], place["status"]) == (stated, "NEEDS_CONTEXT")
    assert "{P1: loading_position_1} has no recorded geometry" in result["reasons"]


def test_a_variable_nothing_sets_is_refused(cell, session):
    session["recovery_des_models"][CONVEYOR]["state_variables"]["loading_reserved_by"] = {
        "scope": "resource", "domain": [None, "ur5e-3"]}
    session["symbolic_resources"][CONVEYOR]["loading_reserved_by"] = None
    result = check_event(_event(CONVEYOR, {"loading_reserved_by": "ur5e-3"}), session, {}, cell)
    assert result["reasons"] == ["no Conveyor primitive sets Conveyor's loading_reserved_by to ur5e-3"]


def test_a_refusal_leaves_out_a_value_only_another_alternative_of_the_primitive_takes(cell, session):
    # No primitive makes a paused job on M1 from nothing; start_job's circle alternative refusing result square
    # says nothing, because its square alternative takes square.
    _job(session, "loaded", "square")
    stated = {"process": "trim", "result": "square", "status": "paused", "machine": "M1"}
    checked = check_event(_event(M1, {"current_process": stated}, part="P2"), session, {}, cell)
    reasons = " | ".join(checked["reasons"])
    assert checked["feasibility_status"] == "INFEASIBLE"
    assert "start_job (result square): it sets running, not paused" in reasons
    assert "outside its declared values ['circle']" not in reasons


def test_a_refusal_keeps_every_alternative_when_none_takes_all_the_stated_values(cell, session):
    # Alternatives that differ in process and result: neither takes both values of a trimmed hole, so both say why.
    scene = deepcopy(cell.scene)
    machine = next(row for row in scene["machines"] if row["resource_id"] == "M1")
    machine["program_options"]["circle"]["program"]["effects"] = [{"process": "drill", "result": "hole"}]
    machine["process_capabilities"]["drill"] = {"supported_results": ["hole"]}
    _job(session, "loaded", "square")
    stated = {"process": "trim", "result": "hole", "status": "running", "machine": "M1"}
    checked = check_event(_event(M1, {"current_process": stated}, part="P2"), session, {},
                          replace(cell, declarations=current_primitive_declarations(scene)))
    reasons = " | ".join(checked["reasons"])
    assert checked["feasibility_status"] == "INFEASIBLE"
    assert "start_job (process drill, result hole): process trim is outside its declared values ['drill']" in reasons
    assert "start_job (process trim, result square): result hole is outside its declared values ['square']" in reasons


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
    assert [row["check"] for row in result["capability"]] == ["pick", "place", "handling"]
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert _place(result)["reason"] == "the capture records no grasp of P1 by ur5e-3 that still holds"
    # The part passes through the gripper: it is grasped where it is taken and released where it is put.
    handling = result["capability"][-1]
    assert (handling["status"], handling["primitives"]) == ("FEASIBLE", ["grasp_part", "release_part"])


def test_a_failed_gripper_cannot_take_a_part_and_set_it_down_in_one_event(cell, session):
    # held_part is null before and after, so no grasp is derived from it; the event still needs one.
    session["symbolic_resources"][UR].update(held_part=None, gripper_state="failed")
    session["symbolic_parts"]["P1"] = {"part_location": "Buffer For Machined parts"}
    result = check_event(_release_here(), session, {}, _pose_place_cell(cell, _bench(TOP), resources={}))
    handling = result["capability"][-1]
    assert (handling["check"], handling["status"]) == ("handling", "INFEASIBLE")
    assert "grasp_part needs gripper_state ≠ failed; projected gripper_state is failed" in handling["reason"]
    assert result["feasibility_status"] == "INFEASIBLE"


def _second_gripper(cell, session):
    """Add the KMR as a second gripping resource that holds P1."""
    session["recovery_des_models"][KMR] = {"state_variables": {
        "resource_state": {"scope": "resource"}, "held_part": {"scope": "resource", "domain": PARTS}}}
    session["symbolic_resources"][KMR] = {"resource_state": "idle", "held_part": "P1", "availability": "available"}
    session["symbolic_parts"]["P1"] = {"part_location": KMR}
    return replace(cell, rids={**cell.rids, KMR: "KMR"})


def test_a_part_in_another_gripper_is_named_as_held_by_it_not_as_a_place(cell, session):
    handoff_cell = _second_gripper(cell, session)
    session["symbolic_resources"][UR].update(held_part=None, gripper_state="open")
    result = check_event(_event(UR, {"held_part": "P1"}, part="P1"), session, {}, handoff_cell)
    pick = next(row for row in result["capability"] if row["check"] == "pick")
    assert (pick["status"], pick["reason"]) == ("NEEDS_CONTEXT",
                                                "P1 is held by KMR's gripper, which is not a recorded place")


def test_a_part_put_into_another_gripper_names_that_gripper_as_no_place(cell, session):
    handoff_cell = _second_gripper(cell, session)
    session["symbolic_resources"][UR].update(held_part=None, gripper_state="open")
    event = {**_event(KMR, {"held_part": None}, part="P1"), "neighbor_end_state": {UR: {"held_part": "P1"}}}
    result = check_event(event, session, {}, handoff_cell)
    place = next(row for row in result["capability"] if row["check"] == "place")
    assert (place["status"], place["reason"]) == ("NEEDS_CONTEXT",
                                                  "ur5e-3's gripper is not a recorded place to put P1")


@pytest.mark.parametrize("slot", ["held_part", "approached_part"])
def test_a_gripper_whose_body_has_geometry_is_still_no_recorded_place(cell, session, slot):
    # The receiving resource's recorded body (a mobile platform) is not where its gripper takes a part.
    handoff_cell = _second_gripper(cell, session)
    session["recovery_des_models"][KMR]["state_variables"]["approached_part"] = {"scope": "resource", "domain": PARTS}
    session["symbolic_resources"][KMR].update(held_part=None, approached_part=None)
    session["symbolic_parts"]["P1"] = {"part_location": UR}
    platform = {"id": "KMR::base::platform", "bounds": [[-0.4, 0.4], [-0.3, 0.3], [0.3, 0.7]]}
    handoff_cell = replace(handoff_cell, bodies={"KMR": {"links": {"base": {"collisions": [platform]}}}})
    event = {**_event(UR, {"held_part": None}, part="P1"), "neighbor_end_state": {KMR: {slot: "P1"}}}
    place = next(row for row in check_event(event, session, {}, handoff_cell)["capability"] if row["check"] == "place")
    assert (place["status"], place["reason"]) == ("NEEDS_CONTEXT", "KMR's gripper is not a recorded place to put P1")


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


def test_a_place_that_ends_at_a_named_posture_takes_it_from_the_posture_primitive(cell, session):
    # Put into zone 1, then home: the place check does not decide a named posture, move_to_named_pose does.
    event = _release()
    event["expected_end_state"]["resource_location"] = "home"
    session["symbolic_resources"][UR]["resource_location"] = "Buffer For Machined parts"
    result = check_event(event, session, {}, cell)
    location = next(row for row in result["conditions"] if row["condition"] == "resource_location")
    assert "found_by" not in location and [producer["primitive"] for producer in location["producers"]
                                           if producer.get("makes_value")] == ["move_to_named_pose"]
