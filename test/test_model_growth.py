"""Model growth after recovery: an accepted recovery's events added to the nominal runtime models.

Supplementary demonstration only (PAPER_FRAMEWORK.md, "Model growth after recovery"). The tests that read the
accepted ur5e-1 breakdown records skip when the gitignored records are absent.
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cais_spade_llm.product.environment import EnvironmentProductContext
from cais_spade_llm.recovery_framework import PRODUCT_PATH, ROOT, SCENE_PATH, read_json
from cais_spade_llm.recovery_framework.model_growth import (
    composed_step_parameters,
    exclude_reported_faults,
    grow_models,
    resource_names,
    states_before_events,
)
from cais_spade_llm.resources import environment_models
from cais_spade_llm.resources.environment_models import candidates, project_activity
from cais_spade_llm.resources.function_contracts import validated_function_contract
from cais_spade_llm.resources.nominal_des import FINISH_SUFFIX

RECORDS = ROOT / "cais_spade_llm/monitor/environment_runs/recovery"
COMPOSITION = RECORDS / "draft_compositions/composition_3075cbc499dc4693a0271b242f57e9a8/composition.json"
CAPTURE = RECORDS / "capture_4f9097aa46bf450ca0717840262806a4/observations.json"
KMR_EVENTS = [
    "Retrieve the completed part from M1 with KMR",
    "Deliver to conveyor loading position 1 using a south-facing base heading",
    "Return KMR to Storage and verify idle readiness",
]
LATER_PART = "KET8_Square_8mm"


def _context() -> EnvironmentProductContext:
    meta = next(iter(read_json(PRODUCT_PATH).values()))
    return EnvironmentProductContext(
        scene=read_json(SCENE_PATH), product_order=read_json(ROOT / meta["product_order_file"]),
        geometry=read_json(ROOT / meta["product_geometry_file"])["gazebo"],
    )


@pytest.fixture(scope="module")
def records() -> tuple[dict, dict]:
    if not (COMPOSITION.exists() and CAPTURE.exists()):
        pytest.skip("The accepted ur5e-1 breakdown records are not retained here")
    return json.loads(COMPOSITION.read_text()), json.loads(CAPTURE.read_text())


@pytest.fixture()
def grown(records) -> tuple[EnvironmentProductContext, dict]:
    composition, observations = records
    context = _context()
    names = resource_names(observations)
    states = states_before_events(observations, composition["accepted_outline_prefix"], names)
    report = grow_models(context.models, composition, names, states, set(context.models) - {"ur5e-1"})
    return context, report


def _start(models: dict, name: str) -> dict:
    return next(e for e in models["KMR"]["events"] if e["event_name"] == name and "finish_event_id" in e)


def _task(models: dict, name: str, part: str | None) -> dict:
    event = _start(models, name)
    parameters = {key: rule.get("equals", part) for key, rule in event["parameter_bindings"].items()}
    return {"resource_id": "KMR", "event_id": event["event_id"], "event_name": name, "parameters": parameters}


def _finished_in_m1(context: EnvironmentProductContext, part: str) -> tuple[dict, dict]:
    valuation, products = context.snapshot(), deepcopy(context.part_tracker)
    valuation["M1"].update(resource_state="completed", part_name=part)
    valuation["Storage"][f"inventory.{part}"] = False
    products[part].update(location="M1", processCompleted=[{"process": "trim", "result": "square"}])
    return valuation, products


def _run(context: EnvironmentProductContext, valuation: dict, products: dict, part: str) -> tuple[dict, dict]:
    for name in KMR_EVENTS:
        valuation, products = project_activity(context.models, valuation, products, _task(context.models, name, part),
                                               context.product_name, context.requirements)
    return valuation, products


def test_belt_order_follows_the_conveyor_side_for_exactly_the_nominal_loads():
    models = _context().models
    by_name, by_declaration = {}, {}
    for model in models.values():
        for event in model["events"]:
            if event["parameter_bindings"]["resource_id"]["equals"] != model["resource_id"]:
                continue
            bindings = event["parameter_bindings"]
            if (event["event_name"] == "place_release" + FINISH_SUFFIX
                    and (bindings.get("destination_location") or {}).get("equals") == "Conveyor"):
                by_name[event["event_id"]] = bindings["loading_position"]["equals"]
            position = environment_models._conveyor_loading_position(models, event)
            if position is not None:
                by_declaration[event["event_id"]] = position
    assert by_name and by_declaration == by_name


def test_growth_adds_the_kmr_events_under_their_accepted_names(grown):
    context, report = grown
    assert [row["event_name"] for row in report["added"]] == KMR_EVENTS
    assert {row["resource_id"] for row in report["added"]} == {"KMR"}
    assert report["left_out"] == [{
        "outline_id": "RECOVERY_SEQ1", "event_name": "Retract the empty arm to its named home pose",
        "resource_id": "ur5e-1", "part_name": None, "reason": "ur5e-1 is not in production"}]
    retrieve, deliver, back = report["added"]
    m1_start = next(e for e in context.models["M1"]["events"] if e["event_id"] == retrieve["copied_from"]["M1"][0])
    belt_start = next(e for e in context.models["Conveyor"]["events"]
                      if e["event_id"] == deliver["copied_from"]["Conveyor"][0])
    assert (m1_start["event_name"], m1_start["parameter_bindings"]["resource_id"]["equals"]) == ("pick_grasp", "ur5e-1")
    assert m1_start["guards"]["resource_state"] == {"equals": "completed"}
    assert belt_start["parameter_bindings"]["loading_position"] == {"equals": "loading_position_1"}
    assert deliver["not_copied"] == {"Conveyor": {"part_order.{part_name}": 0}}
    assert deliver["stated_outside_the_model"] == ["pose"]
    assert back["participants"] == ["KMR"]
    for name in KMR_EVENTS:
        assert name in context.models["KMR"]["local_event_alphabet"]
        assert name in context.models["KMR"]["state_variables"]["resource_state"]["domain"]
        contract = validated_function_contract(context.models, _start(context.models, name)["event_id"])
        assert contract["event_name"] == name and contract["steps"]
    assert {"Conveyor", "Buffer For Machined parts"} <= set(context.models["KMR"]["neighbors"])


def test_kmr_side_keeps_the_stated_values_with_the_part_open(grown):
    context, _ = grown
    retrieve = _start(context.models, KMR_EVENTS[0])
    assert retrieve["guards"] == {"held_part": {"equals": None}, "resource_location": {"equals": "Storage"},
                                  "resource_state": {"equals": "idle"}}
    finish = next(e for e in context.models["KMR"]["events"] if e["event_id"] == retrieve["finish_event_id"])
    assert finish["updates"] == {"resource_location": {"set": "M1"}, "resource_state": {"set": "carrying"},
                                 "held_part": {"set_from_param": "part_name"}}
    assert retrieve["parameter_bindings"]["origin_resource_location"] == {"equals": "M1"}


def test_a_later_part_goes_through_m1_and_takes_its_belt_order(grown):
    context, _ = grown
    valuation, products = _run(context, *_finished_in_m1(context, LATER_PART), LATER_PART)
    assert valuation["KMR"] == {"resource_state": "idle", "held_part": None, "approached_part": None,
                                "resource_location": "Storage"}
    assert (valuation["M1"]["resource_state"], valuation["M1"]["part_name"]) == ("idle", None)
    assert valuation["Conveyor"][f"part_location.{LATER_PART}"] == "loading_position_1"
    assert valuation["Conveyor"][f"part_order.{LATER_PART}"] == 0
    assert products[LATER_PART]["location"] == "Conveyor"

    valuation, products = _finished_in_m1(context, LATER_PART)
    ahead = "RGOCG8-50_8mm"
    valuation["Conveyor"].update({f"part_location.{ahead}": "loading_position_2", f"part_order.{ahead}": 0})
    valuation["Storage"][f"inventory.{ahead}"] = False
    products[ahead].update(location="Conveyor", processCompleted=[{"process": "trim", "result": "circle"}])
    valuation, _ = _run(context, valuation, products, LATER_PART)
    assert valuation["Conveyor"][f"part_order.{LATER_PART}"] == 1


def test_m1_and_the_conveyor_keep_their_own_conditions(grown):
    context, _ = grown
    valuation, products = _finished_in_m1(context, LATER_PART)
    valuation["M1"]["resource_state"] = "running"
    with pytest.raises(ValueError, match="M1.resource_state"):
        project_activity(context.models, valuation, products, _task(context.models, KMR_EVENTS[0], LATER_PART),
                         context.product_name, context.requirements)

    valuation, products = _finished_in_m1(context, LATER_PART)
    valuation, products = project_activity(context.models, valuation, products,
                                           _task(context.models, KMR_EVENTS[0], LATER_PART),
                                           context.product_name, context.requirements)
    occupied, moving = deepcopy(valuation), deepcopy(valuation)
    other = "RGOCG4-50_Round_4mm"
    occupied["Conveyor"].update({f"part_location.{other}": "loading_position_1", f"part_order.{other}": 0})
    occupied["Storage"][f"inventory.{other}"] = False
    occupied_products = deepcopy(products)
    occupied_products[other].update(location="Conveyor")
    with pytest.raises(ValueError, match="occupied"):
        project_activity(context.models, occupied, occupied_products, _task(context.models, KMR_EVENTS[1], LATER_PART),
                         context.product_name, context.requirements)
    moving["Conveyor"]["resource_state"] = "moving"
    with pytest.raises(ValueError, match="Conveyor.resource_state"):
        project_activity(context.models, moving, products, _task(context.models, KMR_EVENTS[1], LATER_PART),
                         context.product_name, context.requirements)


def test_the_scheduler_is_offered_the_added_events(grown):
    context, _ = grown
    valuation, _ = _finished_in_m1(context, LATER_PART)
    offered = {task["event_name"] for task in candidates(
        context.models["KMR"], valuation, LATER_PART, context.requirements[LATER_PART][-1],
        context.requirements[LATER_PART])}
    assert set(KMR_EVENTS) <= offered


def test_growth_names_no_resource_a_synthetic_event_at_m2_copies_ur5e_2s_entry():
    context = _context()
    part = "RGOCG8-50_8mm"
    jids = {rid: f"synthetic-{index}@localhost" for index, rid in enumerate(context.models)}
    values = context.snapshot()
    values["M2"].update(resource_state="completed", part_name=part)
    observations = {"jids": jids, "failure": {"resource_values_before": values}}
    event = {
        "outline_id": "SYNTHETIC_1", "event_name": "Take the finished round peg out of M2",
        "resource_jid": jids["KMR"], "part_name": part,
        "expected_start_state": {"held_part": None, "part_location": "M2", "resource_location": "Storage",
                                 "resource_state": "idle"},
        "expected_end_state": {"resource_location": "M2", "resource_state": "carrying", "held_part": part,
                               "part_location": jids["KMR"]},
        "neighbor_end_state": {jids["M2"]: {"part_name": None, "resource_state": "idle"}},
    }
    refused = deepcopy(event) | {"outline_id": "SYNTHETIC_2", "event_name": "Take it while M2 pauses",
                                 "neighbor_end_state": {jids["M2"]: {"resource_state": "paused"}}}
    program = {"resource_jid": jids["KMR"], "part_name": part,
               "primitive_steps": [{"primitive": "move_to_named_pose", "params": {"pose_name": "transport"}}],
               "composition_validation": {"trace": [{"primitive": "move_to_named_pose",
                                                     "resolved_params": {"pose_name": "transport"}}]}}
    composition = {"accepted_outline_prefix": [event, refused],
                   "accepted_primitive_program": [program | {"outline_id": "SYNTHETIC_1"},
                                                  program | {"outline_id": "SYNTHETIC_2"}]}
    names = resource_names(observations)
    report = grow_models(context.models, composition, names, states_before_events(observations, [event, refused],
                                                                                  names), set(context.models))
    added, = report["added"]
    copied = next(e for e in context.models["M2"]["events"] if e["event_id"] == added["copied_from"]["M2"][0])
    assert (copied["event_name"], copied["parameter_bindings"]["resource_id"]["equals"]) == ("pick_grasp", "ur5e-2")
    left, = report["left_out"]
    assert left["outline_id"] == "SYNTHETIC_2" and "no entry of M2's own model" in left["reason"]


def test_composed_steps_take_this_runs_part_calculations_and_grasp(records):
    composition, _ = records
    retrieve, deliver = composition["accepted_primitive_program"][1:3]
    grasp = next(index for index, step in enumerate(retrieve["primitive_steps"]) if step["primitive"] == "grasp_part")
    assert composed_step_parameters(retrieve, grasp, LATER_PART, {}, None)["part_name"] == LATER_PART

    live_targets = {"approach": [1.0] * 7, "transfer_waypoints": [[2.0] * 7], "target": [3.0] * 7,
                    "retreat": [4.0] * 7, "transport": [5.0] * 7, "withdraw_waypoints": [[6.0] * 7]}
    path = next(row["event_fact_path"] for row in deliver["composition_validation"]["trace"]
                if row.get("event_fact_path"))
    evidence = {"grasp_transform": [0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 1.0]}
    trace = deliver["composition_validation"]["trace"]
    calculate = next(index for index, row in enumerate(trace) if row.get("event_fact_path"))
    params = composed_step_parameters(deliver, calculate, LATER_PART, {}, evidence)
    assert params["transform"] == evidence["grasp_transform"]
    assert params["part_pose"] == trace[calculate]["resolved_params"]["part_pose"]
    approach = composed_step_parameters(deliver, calculate + 1, LATER_PART, {path: live_targets}, evidence)
    assert approach == {"target": live_targets["approach"], "waypoints": live_targets["transfer_waypoints"]}
    drive = composed_step_parameters(deliver, 1, LATER_PART, {}, evidence)
    assert drive["target_pose"] == trace[1]["resolved_params"]["target_pose"]
    assert drive["transform"] == evidence["grasp_transform"]
    with pytest.raises(ValueError, match="before its calculation step ran"):
        composed_step_parameters(deliver, calculate + 1, LATER_PART, {}, evidence)


def test_a_resource_with_a_fault_report_is_taken_out_of_production():
    cleared = {"pick_approach": object()}
    context = SimpleNamespace(
        admission_lock=threading.RLock(),
        fault_reports={"ur5e-1": {"reported_state": "failed"}}, unavailable_resources=set(),
        resources={"ur5e-1": SimpleNamespace(executors=cleared), "KMR": SimpleNamespace(executors={"x": 1})})
    assert exclude_reported_faults(SimpleNamespace(context=context)) == ["ur5e-1"]
    assert context.unavailable_resources == {"ur5e-1"} and cleared == {}
    assert context.resources["KMR"].executors == {"x": 1}
