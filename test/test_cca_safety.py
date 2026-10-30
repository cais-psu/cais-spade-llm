"""CCA safety validation of each recovery outline event (Eq. 13), applied after RA feasibility (Eq. 9)."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.predefined_safety import (
    compile_predefined_safety,
    parse_predefined_safety,
)
from cais_spade_llm.recovery_framework import ROOT, cca_safety

SAFETY = ROOT / "cais_spade_llm/specification/safety"
SELECTED = [f"cais_spade_llm/specification/safety/safety_{name}.txt" for name in (
    "assembly_board-v1_mutex", "M1_mutex", "M2_mutex", "assembly_board-v1_trim_before_entry",
    "assembly_board-v1_gear_small_before_KET4_Square_4mm",
    "assembly_board-v1_KET4_Square_4mm_before_RGOCG4-50_Round_4mm",
    "KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing")]
RECORDS = ROOT / "cais_spade_llm/monitor/environment_runs/recovery"
UR, KMR = "recovery-resource-3@localhost", "recovery-resource-8@localhost"


def test_the_selected_files_merge_into_the_plant_document():
    plant = parse_predefined_safety((SAFETY / "safety_assembly_board-v1_plant.txt").read_text())
    assert cca_safety._document(SELECTED, ROOT) == plant


@pytest.fixture(scope="module")
def area_rule():
    """The board area rule grounded for one pair of resources, as the CCA's monitor for it."""
    plant = parse_predefined_safety((SAFETY / "safety_assembly_board-v1_plant.txt").read_text())
    document = {**plant, "catalog": {**plant["catalog"], "specifications": [
        spec for spec in plant["catalog"]["specifications"] if spec["id"] == "SAFE_shared_area_mutex"]},
        "requirement_scopes": [scope for scope in plant["requirement_scopes"]
                               if scope["specification"] == "SAFE_shared_area_mutex"]}
    rules, compiled = compile_predefined_safety(document)
    rule = {**rules[0], "rule_id": "SAFE_shared_area_mutex:pair", "ap_kinds": {
        "ap001": ("resource_region", {"resource": "ur5e-3", "region": "assembly_board-v1"}),
        "ap002": ("resource_region", {"resource": "ur5e-4", "region": "assembly_board-v1"})}}
    checker = BaseSafetyChecker({rule["rule_id"]: compiled[rule["id"]]}, [rule])
    return {"rules": [rule], "checker": checker,
            "requirements": {"SAFE_shared_area_mutex": "At most one resource may occupy the board area."}}


@pytest.mark.parametrize("third, fourth, broken", [(False, False, False), (True, False, False), (True, True, True)])
def test_a_second_resource_in_the_area_breaks_the_rule_and_the_refusal_names_both(area_rule, third, fourth, broken):
    rule_id = area_rule["rules"][0]["rule_id"]
    states = {rule_id: {area_rule["checker"].dfas[rule_id]["initial"]}}
    base = {"occupancy": {"ur5e-3": {"assembly_board-v1": (third,)}, "ur5e-4": {"assembly_board-v1": (fourth,)}},
            "part_entry": {}, "completions": {}}
    reached, violation = cca_safety._advance(area_rule, states, base, universal=True)
    assert (violation is not None) == broken
    if broken:
        text = cca_safety._violation_text(area_rule, "after the event", violation)
        assert text == ("SAFE_shared_area_mutex (At most one resource may occupy the board area.) is broken after "
                        "the event: ur5e-3 is in assembly_board-v1 while ur5e-4 is in assembly_board-v1")
    else:
        assert reached[rule_id]


def test_an_unknown_value_is_judged_by_every_value_it_may_take(area_rule):
    rule_id = area_rule["rules"][0]["rule_id"]
    states = {rule_id: {area_rule["checker"].dfas[rule_id]["initial"]}}
    base = {"occupancy": {"ur5e-3": {"assembly_board-v1": (True,)}, "ur5e-4": {"assembly_board-v1": (False, True)}},
            "part_entry": {}, "completions": {}}
    # Checking an event, any value that breaks the rule refuses it; replaying history, that value is ruled out.
    assert cca_safety._advance(area_rule, states, base, universal=True)[1] is not None
    reached, violation = cca_safety._advance(area_rule, states, base, universal=False)
    assert violation is None and reached[rule_id]


@pytest.fixture
def placing():
    robot = {"named": {"home": {"positions": [0.0] * 6}}, "failure_pose": {"joints": [1.0] * 6, "base": None}}
    mobile = {"named": {}, "failure_pose": {"joints": [0.5] * 7, "base": [0.0, 0.0, 0.0]}}
    return {"robots": {"ur5e-3": robot, "KMR": mobile}, "docks": {"M1": [-4.7, 2.2, 1.57]},
            "arms": {"parked": [0.1] * 7, "carrying": [0.2] * 7}}


@pytest.mark.parametrize("rid, jid, end, witness, held, expected", [
    # A named posture puts the robot there.
    ("ur5e-3", UR, {"resource_location": "home"}, None, None, {"joints": [0.0] * 6, "base": None}),
    # Without a named posture, the RA's reach configuration does.
    ("ur5e-3", UR, {"resource_location": "Buffer For Machined parts"}, [2.0] * 6, None,
     {"joints": [2.0] * 6, "base": None}),
    # An event that states no new position leaves the robot where it was.
    ("ur5e-3", UR, {"resource_state": "idle"}, None, None, {"joints": [1.0] * 6, "base": None}),
    # A mobile base at a dock carries its arm parked, or in its carrying posture while it holds a part.
    ("KMR", KMR, {"resource_location": "M1"}, None, None, {"joints": [0.1] * 7, "base": [-4.7, 2.2, 1.57]}),
    ("KMR", KMR, {"resource_location": "M1"}, None, "KET4_Square_4mm", {"joints": [0.2] * 7, "base": [-4.7, 2.2, 1.57]}),
    # A stated base pose wins over the dock its location names, as the RA reaches from it.
    ("KMR", KMR, {"resource_location": "M1", "pose": {"x": -4.95, "y": 1.9, "yaw": 3.14}}, None, None,
     {"joints": [0.1] * 7, "base": [-4.95, 1.9, 3.14]}),
    # A new place with no reach configuration and no declared pose cannot be placed.
    ("ur5e-3", UR, {"resource_location": "Conveyor"}, None, None, None),
])
def test_the_acting_robot_is_placed_by_named_posture_dock_or_reach_configuration(placing, rid, jid, end, witness, held,
                                                                                  expected):
    session = {"symbolic_resources": {jid: {"resource_location": "Buffer For Machined parts" if rid == "ur5e-3" else "Storage"}}}
    after = {"symbolic_resources": {jid: {"held_part": held}}}
    ra = {"capability": [{"check": "move", "status": "FEASIBLE", "witness": {"joint_positions": witness}}] if witness else []}
    _, pose = cca_safety._acting_poses(rid, jid, {"expected_end_state": end}, session, after, placing, {}, ra)
    assert pose == expected


def _hook(monkeypatch, verdicts):
    """Run the candidate hook with stand-in RA and CCA checks keyed by event name."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_diagnostic,
    )
    from cais_spade_llm.recovery_framework import ra_feasibility

    applied = []

    async def handle_event(*, session, prepared, response, planner):
        applied.append(response["next_transition"]["event_name"])
        return "need_next_task", {}

    def ra_check(event, session, prepared, cell):
        return {"feasibility_status": verdicts[event["event_name"]][0], "reasons": ["the RA fact"], "capability": []}

    def cca_check(event, session, prepared, capture, feasibility):
        status = verdicts[event["event_name"]][1]
        result = {"safety_status": status, "reasons": [] if status == "SAFE" else ["the CCA fact"], "steps": []}
        return {**result, "state_after": {"monitors": {}, "robots": {"ur5e-3": event["event_name"]}}} if status == "SAFE" else result

    monkeypatch.setattr(multi_turn_diagnostic, "handle_event", handle_event)
    monkeypatch.setattr(ra_feasibility, "check_event", ra_check)
    monkeypatch.setattr(cca_safety, "check_event", cca_check)
    return applied


def _run(events, session, choice=0):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import (
        handle_validated_candidates,
    )

    planner = SimpleNamespace(product_agent=SimpleNamespace(recorded_cell=object(), cca_safety_capture={}))
    response = {"thought": "t", "candidate_events": [{"event_name": name, "resource_jid": UR} for name in events],
                "selected_candidate_index": choice}
    return asyncio.run(handle_validated_candidates(session=session, prepared={"offline_recovery": True},
                                                   response=deepcopy(response), planner=planner))


def test_the_hook_applies_the_first_candidate_both_checks_pass(monkeypatch):
    applied = _hook(monkeypatch, {"unsafe": ("FEASIBLE", "UNSAFE"), "safe": ("FEASIBLE", "SAFE")})
    session = {"outline_diagnostic_bypass": True, "check_ra_feasibility": True, "check_cca_safety": True,
               "turn_index": 2}
    decision, audit = _run(["unsafe", "safe"], session)
    assert (decision, applied) == ("need_next_task", ["safe"])
    assert (audit["selected_candidate_index"], audit["llm_selected_candidate_index"]) == (1, 0)
    # The RA's record keeps its shape: the candidate actually applied and every RA result.
    assert session["ra_feasibility_checks"][-1]["applied_candidate_index"] == 1
    assert [row["candidate_index"] for row in session["ra_feasibility_checks"][-1]["results"]] == [0, 1]
    assert [row["safety_status"] for row in session["cca_safety_checks"][-1]["results"]] == ["UNSAFE", "SAFE"]
    assert all("state_after" not in row for row in session["cca_safety_checks"][-1]["results"])
    assert session["cca_safety_state"]["robots"] == {"ur5e-3": "safe"}
    assert [row["validator_role"] for row in audit["validation_stages"]] == [
        "ResourceAgent", "ResourceAgent", "CentralControllerAgent", "CentralControllerAgent"]


def test_the_hook_returns_ra_and_cca_facts_when_no_candidate_passes(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _outline_feedback_for_prompt,
    )

    applied = _hook(monkeypatch, {"infeasible": ("INFEASIBLE", "SAFE"), "unsafe": ("FEASIBLE", "UNSAFE")})
    session = {"outline_diagnostic_bypass": True, "check_ra_feasibility": True, "check_cca_safety": True}
    decision, audit = _run(["infeasible", "unsafe"], session)
    findings = session["outline_validation_findings"]
    assert decision == "need_revision" and applied == []
    assert [row["constraint_code"] for row in findings] == ["ra_feasibility", "cca_safety"]
    assert session["ra_feasibility_checks"][-1]["applied_candidate_index"] is None
    # The CCA only checks what the RA finds feasible.
    assert [row["candidate_index"] for row in session["cca_safety_checks"][-1]["results"]] == [1]
    # The next prompt shows the CCA's facts as the check wrote them.
    assert _outline_feedback_for_prompt(findings[1], set())["reason"] == findings[1]["reason"] == "the CCA fact"


def test_the_cca_safety_check_needs_the_ra_check(monkeypatch):
    _hook(monkeypatch, {"safe": ("FEASIBLE", "SAFE")})
    session = {"outline_diagnostic_bypass": True, "check_cca_safety": True}
    with pytest.raises(ValueError, match="RA feasibility check"):
        _run(["safe"], session)


@pytest.mark.parametrize("capture, assumptions, moved", [
    ("capture_b7dea70c1db74e1e9e98849380c09655", None, {
        "SAFE_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing", "SAFE_gear_small_before_KET4_Square_4mm",
        "SAFE_trim_before_assembly_board-v1_entry"}),
    ("capture_ad64bf1628564c3fa7f928b7bdebd5d9", [{"resource": "ur5e-3", "clear_of": ["assembly_board-v1"],
                                                   "reason": "it would have waited clear with the CCA on"}],
     {"SAFE_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing", "SAFE_trim_before_assembly_board-v1_entry"}),
])
def test_monitors_start_where_the_recorded_history_leaves_them(capture, assumptions, moved):
    path = RECORDS / capture / "observations.json"
    if not path.exists():
        pytest.skip("the recorded capture is not on this machine")
    inputs = cca_safety.capture_inputs(json.loads(path.read_text()), SELECTED, assumptions)
    checker = inputs["checker"]
    assert len(inputs["rules"]) == 97
    assert {rule_id.split(":")[0] for rule_id, states in inputs["failure_states"].items()
            if states != [checker.dfas[rule_id]["initial"]]} == moved


def test_a_failure_state_that_already_breaks_a_rule_stops_the_check():
    path = RECORDS / "capture_ad64bf1628564c3fa7f928b7bdebd5d9/observations.json"
    if not path.exists():
        pytest.skip("the recorded capture is not on this machine")
    with pytest.raises(ValueError, match="ur5e-3 in assembly_board-v1; ur5e-4 in assembly_board-v1"):
        cca_safety.capture_inputs(json.loads(path.read_text()), SELECTED)


def _job(result):
    return {"event_name": "complete_job", "controllable": False, "start_event_id": None,
            "guards": {"resource_state": {"equals": "running"}, "part_name": {"equals_from_param": "part_name"}},
            "parameter_bindings": {"resource_id": {"equals": "M2"}, "part_name": {"equals": "RGOCG4-50_Round_4mm"},
                                   "process": {"equals": "trim"}, "result": {"equals": result}},
            "product_effects": {"processCompleted": [{"process": "trim", "result": {"set_from_param": "result"}}]}}


@pytest.mark.parametrize("state, faulted, expected", [
    ("running", set(), [("RGOCG4-50_Round_4mm", "trim", "result", "circle")]),
    # A paused machine completes nothing, whatever the part's own record still says.
    ("paused", set(), []),
    # A faulted machine completes nothing.
    ("running", {"recovery-resource-6@localhost"}, []),
])
def test_a_job_may_complete_on_its_own_only_while_its_machine_runs_it(monkeypatch, state, faulted, expected):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_diagnostic

    monkeypatch.setattr(multi_turn_diagnostic, "faulted_resources", lambda session, prepared: faulted)
    machine = "recovery-resource-6@localhost"
    after = {"recovery_des_models": {machine: {"events": [_job("square"), _job("circle")]}},
             "symbolic_resources": {machine: {"resource_state": state, "part_name": "RGOCG4-50_Round_4mm"}},
             "symbolic_parts": {"RGOCG4-50_Round_4mm": {"processCompleted": [], "current_process": {
                 "process": "trim", "result": "circle", "status": "running", "machine": "M2"}}}}
    keys = [key for _, key in cca_safety._running_completions(after, {}, {"rids": {machine: "M2"}})]
    assert keys == expected


@pytest.mark.parametrize("in_motion, moved, expected", [
    # At rest, restating its location leaves the robot where it is.
    ([], {}, {"joints": [1.0] * 6, "base": None}),
    # Still moving at the failure, the location it records does not place it.
    (["ur5e-3"], {}, None),
    # Once the recovery has placed it, restating its location leaves it there.
    (["ur5e-3"], {"ur5e-3": {"joints": [3.0] * 6, "base": None}}, {"joints": [3.0] * 6, "base": None}),
])
def test_a_restated_location_places_a_robot_only_where_it_is_known_to_rest(placing, in_motion, moved, expected):
    session = {"symbolic_resources": {UR: {"resource_location": "3D Printing Station"}}}
    after = {"symbolic_resources": {UR: {}}}
    _, pose = cca_safety._acting_poses("ur5e-3", UR, {"expected_end_state": {"resource_location": "3D Printing Station"}},
                                       session, after, {**placing, "in_motion": in_motion}, moved, {"capability": []})
    assert pose == expected


def test_a_restated_pose_leaves_the_robot_where_it_is(placing):
    session = {"symbolic_resources": {UR: {"resource_location": "Buffer For Machined parts"}}}
    pose = {"x": -0.68, "y": 0.5, "z": 1.04}
    ra = {"capability": [], "conditions": [{"entity": "ur5e-3", "condition": "pose", "before": pose, "after": pose}]}
    _, pose = cca_safety._acting_poses("ur5e-3", UR, {"expected_end_state": {"pose": pose}},
                                       session, {"symbolic_resources": {UR: {}}}, {**placing, "in_motion": []}, {}, ra)
    assert pose == {"joints": [1.0] * 6, "base": None}


def test_an_acting_robot_the_event_leaves_in_place_keeps_its_declared_assumption():
    region = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    failure = {"joints": [1.0] * 6, "base": None}
    capture = {"assumptions": [{"resource": "ur5e-3", "clear_of": ["board"]}], "rids": {}, "region_resources": ["ur5e-3"],
               "robots": {"ur5e-3": {"failure_pose": failure}}, "recorded_boxes": {"ur5e-3": [region]},
               "regions": {"board": region}, "used_regions": ["board"], "entry_regions": [], "part_boxes": {}}
    base = cca_safety._base(capture, {}, "ur5e-3", deepcopy(failure), {}, {}, {}, {})
    assert base["occupancy"]["ur5e-3"]["board"] == (False,)


def test_a_planned_task_the_recovery_claimed_counts_as_completed():
    insert = {"event_name": "place_insert_done", "parameter_bindings": {"resource_id": {"equals": "ur5e-4"}},
              "product_effects": {"processCompleted": [{"process": "assembly", "target": {"set_from_param": "target"}}]}}
    node = {"id": "n1", "function_name": "place_insert",
            "params": {"resource_id": "ur5e-4", "part_name": "gear_small", "target": "Gear_Plate/Gear_Shaft_1"}}
    state = {"symbolic_parts": {"gear_small": {"processCompleted": [{"process": "print_part"}]}},
             "diagnostic_replaced_task_claims": {"n1": {}}}
    done = cca_safety._completions(state, {"saved_pending_nodes": [node]}, {"models": {"ur5e-4": {"events": [insert]}}})
    assert set(done) == {("gear_small", "print_part", None, None),
                         ("gear_small", "assembly", "target", "Gear_Plate/Gear_Shaft_1")}
