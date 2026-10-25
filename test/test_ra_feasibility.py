"""RA feasibility of a proposed recovery event: backward derivation (Alg. 1) and the forward check (Eq. 12)."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        rids={UR: "ur5e-3", BUFFER: "Buffer For Machined parts", M1: "M1", CONVEYOR: "Conveyor"})


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
            UR: {"resource_state": "picked", "held_part": "P1", "resource_location": "home", "current_pose": pose},
            BUFFER: {"resource_state": "idle", "zone_1_part": None},
            M1: {"resource_state": "loaded", "part_name": "P2", "selected_program": "square"},
            CONVEYOR: {"resource_state": "idle", "part_location.P1": "loading_position_1"},
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
        assert "current_process matches {status: paused, machine: M1}" in " ".join(result["reasons"])


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
