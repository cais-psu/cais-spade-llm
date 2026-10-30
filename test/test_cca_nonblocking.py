"""CCA nonblocking validation of a declared recovery (Eq. 6) in the composition of the runtime models."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cais_spade_llm.agents.central_controller.local_composition import Budget
from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor
from cais_spade_llm.product.environment import EnvironmentProductContext
from cais_spade_llm.recovery_framework import (
    PRODUCT_PATH,
    ROOT,
    SCENE_PATH,
    cca_nonblocking,
    read_json,
)
from cais_spade_llm.recovery_framework.environment_composition import EnvironmentPlant
from cais_spade_llm.resources.environment_models import owners

TRIMMED = {"process": "trim", "result": "square"}


@pytest.fixture(scope="module")
def inputs():
    meta = next(iter(read_json(PRODUCT_PATH).values()))
    return {"scene": read_json(SCENE_PATH), "product_order": read_json(ROOT / meta["product_order_file"]),
            "geometry": read_json(ROOT / meta["product_geometry_file"])["gazebo"]}


@pytest.fixture
def cell(inputs):
    """The order's initial values, recorded the way an outline session and its capture record them."""
    context = EnvironmentProductContext(**deepcopy(inputs))
    jids = {rid: f"{rid}@localhost" for rid in context.resources}
    values = context.snapshot()
    # Captures observe each robot at home; the models' initial values leave its place unset.
    for row in inputs["scene"]["robots"]:
        values[row["resource_id"]]["resource_location"] = "home"
    parts = {part: {"part_location": state["location"], "current_process": state.get("current_process"),
                    "processCompleted": list(state.get("processCompleted") or [])}
             for part, state in context.part_tracker.items()}
    session = {"symbolic_resources": {jids[rid]: deepcopy(row) for rid, row in values.items()},
               "base_symbolic_resources": {jids[rid]: deepcopy(row) for rid, row in values.items()},
               "symbolic_parts": deepcopy(parts), "base_symbolic_parts": deepcopy(parts),
               "outline_diagnostic_bypass": True, "check_nonblocking": True}
    prepared = {"offline_recovery": True, "observation_jids": jids, "saved_pending_nodes": [],
                "code_grounding": {"resources": deepcopy(values), "fault_event": {"signals": []}}}
    capture = {"inputs": deepcopy(inputs), "permitted_resources": sorted(context.resources), "jids": jids,
               "unavailable_resources": []}
    return SimpleNamespace(context=context, session=session, prepared=prepared, capture=capture, jids=jids)


def _set(cell, rid, **values):
    cell.session["symbolic_resources"][cell.jids[rid]].update(values)


def _place(cell, part, location, *done):
    cell.session["symbolic_parts"][part].update(part_location=location, processCompleted=list(done))


def _check(cell):
    return cca_nonblocking.check_completion(cell.session, cell.prepared, cell.capture)


def _target(cell, part):
    return cell.context.requirements[part][-1]["processesToComplete"][0]["target"]


def test_a_part_left_on_its_route_is_accepted_after_a_search_to_the_end(cell):
    part = "KET4_Square_4mm"
    _set(cell, "Storage", **{f"inventory.{part}": False})
    _set(cell, "Buffer For Machined parts", zone_4_part=part)
    _place(cell, part, "Buffer For Machined parts", TRIMMED)
    check = _check(cell)
    assert (check["nonblocking_status"], check["reasons"], check["blocked"]) == ("NONBLOCKING", [], [])
    assert check["evidence"]["parts"] == [part] and check["evidence"]["explored_states"] > 1


def test_only_a_robot_acting_for_a_part_or_away_from_rest_keeps_its_way_home(cell):
    part = "KET4_Square_4mm"
    _set(cell, "Storage", **{f"inventory.{part}": False})
    _set(cell, "Buffer For Machined parts", zone_4_part=part)
    _place(cell, part, "Buffer For Machined parts", TRIMMED)
    _set(cell, "ur5e-1", resource_location="Conveyor")
    context, invalid, _ = cca_nonblocking._end_context(cell.session, cell.prepared, cell.capture)
    plant = EnvironmentPlant(context.calculation_snapshot(), OnlineSafetyMonitor({}, []), cell.jids,
                             {part: context.requirements[part][-1]})
    rows = plant._part_templates(part, Budget())
    # move_home_done is part of move_home: it must not make every robot look busy with the part.
    homing = {row["task"]["resource_id"] for row in rows if row["task"]["event_name"].startswith("move_home")}
    assert invalid == [] and homing == {"ur5e-1", "ur5e-3"}


def test_a_robot_left_idle_away_from_home_keeps_its_way_home(cell):
    part = "KET4_Square_4mm"
    _set(cell, "Storage", **{f"inventory.{part}": False})
    _set(cell, "Buffer For Machined parts", zone_4_part=part)
    _place(cell, part, "Buffer For Machined parts", TRIMMED)
    # The recovery used ur5e-1 and left it at the belt; a reservation may bring it into the composition.
    _set(cell, "ur5e-1", resource_location="Conveyor")
    assert _check(cell)["nonblocking_status"] == "NONBLOCKING"


def test_a_part_no_normal_event_can_carry_on_is_refused_by_name_and_place(cell):
    part = "RGOCG4-50_Round_4mm"
    _set(cell, "Storage", **{f"inventory.{part}": False})
    # Events state where a part is on the belt, never the belt's order, which the check derives.
    _set(cell, "Conveyor", **{f"part_location.{part}": "output_nest"})
    _place(cell, part, "Conveyor")
    check = _check(cell)
    assert check["nonblocking_status"] == "BLOCKING" and check["blocked"] == [part]
    assert check["reasons"] == [f"{part} cannot reach its remaining requirement (trim → circle, then assembly "
                                f"at {_target(cell, part)}) from Conveyor output_nest"]


@pytest.mark.parametrize("exit_state, blocked, reason", [
    ("occupied", "KET4_Square_4mm", "KET4_Square_4mm cannot reach its remaining requirement (trim → square, then "
                                    "assembly at {target}) from Exit"),
    ("empty", "Exit", "Exit is in a state normal production does not have (empty; part_name KET4_Square_4mm)"),
])
def test_a_part_left_at_the_exit_is_refused_as_that_part_not_as_the_product(cell, exit_state, blocked, reason):
    part = "KET4_Square_4mm"
    _set(cell, "Storage", **{f"inventory.{part}": False})
    _set(cell, "Exit", resource_state=exit_state, part_name=part)
    _place(cell, part, "Exit")
    check = _check(cell)
    # The Exit holds the product only when its part_name is the product (user decision, 2026-10-30); another part
    # there is that part, which no normal event takes from the Exit.
    assert check["nonblocking_status"] == "BLOCKING" and check["blocked"] == [blocked]
    assert check["reasons"] == [reason.format(target=_target(cell, part))]


def test_a_part_recorded_away_from_the_resource_that_holds_it_names_the_values_that_say_so(cell):
    part = "KET4_Square_4mm"
    _set(cell, "Storage", **{f"inventory.{part}": False})
    _set(cell, "ur5e-3", resource_state="positioned", held_part=part)
    _place(cell, part, "Buffer For Machined parts")
    check = _check(cell)
    assert check["nonblocking_status"] == "BLOCKING"
    assert (f"{part} is recorded at Buffer For Machined parts, but ur5e-3's values (held_part {part}) put it at "
            "ur5e-3") in check["reasons"]


def test_a_resource_left_in_a_state_its_model_does_not_have_is_refused_without_a_search(cell):
    _set(cell, "KMR", resource_state="positioned", resource_location=None)
    check = _check(cell)
    assert check == {"nonblocking_status": "BLOCKING", "blocked": ["KMR"], "reasons": [
        "KMR is in a state normal production does not have (positioned; held_part none; resource_location none)"],
        "evidence": {"invalid_state": ["Resource state disagrees with held_part"]}}


def test_a_planned_task_the_recovery_stood_in_for_completes_its_part(cell):
    part, product = "gear_small", cell.context.product_name
    _set(cell, "3D Printing Station", **{f"output.{part}": False})
    # A robot's recovery event leaves the part on the product but records no assembly.
    _place(cell, part, product, {"process": "print_part"})
    cell.prepared["saved_pending_nodes"] = [{"id": "insert_gear", "function_name": "place_insert", "params": {
        "resource_id": "ur5e-4", "part_name": part, "destination_location": product,
        "target": _target(cell, part)}}]
    assert _check(cell)["reasons"] == [
        f"{part} cannot reach its remaining requirement (assembly at {_target(cell, part)}) from {product}"]
    cell.session["diagnostic_replaced_task_claims"] = {"insert_gear": {"outline_id": "R3"}}
    assert _check(cell) == {"nonblocking_status": "NONBLOCKING", "reasons": [], "blocked": [],
                            "evidence": {"parts": [], "explored_states": 0}}


def test_owners_names_every_violation_by_resource_when_asked(cell):
    context = cell.context
    values = context.snapshot()
    values["KMR"]["resource_state"] = "positioned"
    values["M1"]["resource_state"] = "paused"
    with pytest.raises(ValueError, match="disagrees with"):
        owners(context.models, values, context.part_tracker, context.product_name)
    errors = []
    owners(context.models, values, context.part_tracker, context.product_name, errors=errors)
    assert sorted(errors) == [("KMR", "Resource state disagrees with held_part"),
                              ("M1", "Machine state disagrees with workholding")]


def test_the_hook_refuses_a_blocking_completion_and_the_prompt_shows_the_refusal_as_written(cell):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _outline_feedback_for_prompt,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import (
        handle_validated_candidates,
    )

    _set(cell, "KMR", resource_state="positioned", resource_location=None)
    declare = {"thought": "The failure is overcome.", "decision": "recovery_complete"}
    planner = SimpleNamespace(product_agent=SimpleNamespace(cca_capture=cell.capture))
    decision, audit = asyncio.run(handle_validated_candidates(
        session=cell.session, prepared=cell.prepared, response=deepcopy(declare), planner=planner))
    finding = cell.session["outline_validation_findings"][0]
    assert (decision, cell.session["status"]) == ("need_revision", "paused_after_outline_turn")
    assert finding["constraint_code"] == "cca_nonblocking"
    assert finding["reason"] == audit["cca_nonblocking"]["reasons"][0]
    assert cell.session["cca_nonblocking_checks"][-1]["nonblocking_status"] == "BLOCKING"
    # The next prompt shows the refusal as the CCA wrote it: what is blocked, never what it needs.
    assert _outline_feedback_for_prompt(finding, set())["reason"] == finding["reason"]
    with pytest.raises(ValueError, match="capture's runtime inputs"):
        asyncio.run(handle_validated_candidates(session=cell.session, prepared=cell.prepared,
                                                response=deepcopy(declare),
                                                planner=SimpleNamespace(product_agent=SimpleNamespace())))


def test_a_machine_left_in_its_fault_state_is_one_normal_production_has_with_or_without_its_part(cell):
    # Its part taken out, the broken M1 stays interrupted: a declared fault state, not an empty machine at work.
    _set(cell, "M1", resource_state="interrupted", part_name=None)
    assert _check(cell)["nonblocking_status"] == "NONBLOCKING"
    errors = []
    owners(cell.context.models, {**cell.context.snapshot(), "M1": {**cell.context.snapshot()["M1"],
                                 "resource_state": "interrupted", "part_name": None}},
           cell.context.part_tracker, cell.context.product_name, errors=errors)
    assert errors == []


@pytest.mark.parametrize("gripper, status", [("open", "NONBLOCKING"), ("failed", "BLOCKING")])
def test_normal_production_cannot_use_a_grasp_the_device_reports_it_cannot_make(cell, gripper, status):
    part = "KET4_Square_4mm"
    _set(cell, "Storage", **{f"inventory.{part}": False})
    _set(cell, "Buffer For Machined parts", zone_4_part=part)
    _place(cell, part, "Buffer For Machined parts", TRIMMED)
    # The device's own report, as grounding records it from the failure on: a condition the nominal models
    # do not have, so normal production leaves it as it is.
    for rows in (cell.session["symbolic_resources"], cell.session["base_symbolic_resources"]):
        rows[cell.jids["ur5e-3"]]["gripper_state"] = gripper
    check = _check(cell)
    assert check["nonblocking_status"] == status
    if status == "BLOCKING":
        assert check["blocked"] == [part]
        assert check["evidence"]["unrunnable_events"]["ur5e-3"]["pick_grasp"] == (
            "grasp_part needs gripper_state ≠ failed; gripper_state is failed")
        # Its arm still moves: only the events whose programs grasp or release are out.
        assert "move_home" not in check["evidence"]["unrunnable_events"]["ur5e-3"]
    else:
        assert "unrunnable_events" not in check["evidence"]


@pytest.mark.parametrize("unavailable, status", [(["M1"], "NONBLOCKING"), ([], "BLOCKING")])
def test_a_device_that_reports_itself_unavailable_owes_normal_production_no_state(cell, unavailable, status):
    # Out of production until repaired by hand, it need not be in a state normal production has.
    _set(cell, "M1", resource_state="paused", part_name=None)
    cell.capture["unavailable_resources"] = unavailable
    check = _check(cell)
    assert check["nonblocking_status"] == status
    if status == "BLOCKING":
        assert check["evidence"] == {"invalid_state": ["Machine state disagrees with workholding"]}


def test_a_faulted_robot_owes_normal_production_no_return_to_rest(cell):
    part = "KET4_Square_4mm"
    _set(cell, "Storage", **{f"inventory.{part}": False})
    _set(cell, "Buffer For Machined parts", zone_4_part=part)
    _place(cell, part, "Buffer For Machined parts", TRIMMED)
    # ur5e-1 serves M1, so the KMR's trips to M1 reserve it; reported failed, it never returns to rest.
    _set(cell, "ur5e-1", resource_state="failed")
    cell.prepared["code_grounding"]["fault_event"]["signals"] = [
        {"signal": "device_fault_report", "resource_id": "ur5e-1", "values": {"resource_state": "failed"}}]
    assert _check(cell)["nonblocking_status"] == "NONBLOCKING"


def test_a_robot_left_positioned_with_its_part_keeps_the_place_it_approached(cell):
    part = "KET4_Square_4mm"
    _set(cell, "Storage", **{f"inventory.{part}": False})
    _set(cell, "ur5e-3", resource_state="positioned", held_part=part, resource_location=cell.context.product_name)
    _place(cell, part, cell.jids["ur5e-3"], TRIMMED)
    assert _check(cell)["nonblocking_status"] == "NONBLOCKING"


def test_a_value_no_event_stated_is_reported_alone(cell):
    part, m2 = "RGOCG4-50_Round_4mm", cell.jids["M2"]
    # Grounded, M2 held the part; the events moved it to the belt and left M2's workholding unknown.
    cell.prepared["code_grounding"]["resources"]["M2"].update(resource_state="loaded", part_name=part)
    _set(cell, "Storage", **{f"inventory.{part}": False})
    _set(cell, "Conveyor", **{f"part_location.{part}": "output_nest"})
    _place(cell, part, "Conveyor")
    del cell.session["symbolic_resources"][m2]["part_name"]
    cell.session["diagnostic_unknown_owner_fields"] = {m2: ["part_name"]}
    assert _check(cell)["reasons"] == ["M2 is in a state normal production does not have (part_name not stated)"]


def test_a_belt_order_an_event_stated_is_kept_as_stated(cell):
    part = "RGOCG4-50_Round_4mm"
    _set(cell, "Storage", **{f"inventory.{part}": False})
    _set(cell, "Conveyor", **{f"part_location.{part}": "output_nest", f"part_order.{part}": 1})
    _place(cell, part, "Conveyor")
    assert _check(cell)["reasons"] == [
        f"Conveyor is in a state normal production does not have "
        f"(idle; part_location.{part} output_nest; part_order.{part} 1)"]
