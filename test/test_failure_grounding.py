from __future__ import annotations

"""Generic failure grounding reads raw facts only and fills recorded gaps."""

import asyncio
import json
from copy import deepcopy

import pytest

from cais_spade_llm.recovery_framework import failure_grounding as grounding
from cais_spade_llm.recovery_framework.recovery_observations import RECORDS, read_observations

SETTINGS = {"goal_status_not_completed": [5, 6], "job_states_not_completed": ["interrupted", "stopped"],
            "fault_report_values": ["failed"], "region_margin_m": 0.0,
            "location_parameters": ["destination_location", "origin_resource_location"]}


def capture(**changes) -> dict:
    """A healthy two-robot cell: arm_a holds peg, gear rests in the cell region."""
    record = {
        "scenario": "Supplied scenario name",
        "jids": {"arm_a": "jid-a", "arm_b": "jid-b", "cell": "jid-cell"},
        "failure": {
            "resource_values_before": {"arm_a": {"resource_state": "picked", "held_part": "peg"},
                                       "arm_b": {"resource_state": "idle", "held_part": None}},
            "part_tracker_before": {"peg": {"location": "arm_a", "current_process": None},
                                    "gear": {"location": "cell", "current_process": None}},
        },
        "transitions": [{"acknowledgement": {"task_id": "t1", "revision": 1},
                         "after": {"arm_a": {"resource_state": "picked", "held_part": "peg"}}}],
        "pending_nodes": [
            {"id": "t2", "function_name": "place_approach", "sequence_index": 0,
             "params": {"resource_id": "arm_a", "destination_location": "cell", "part_name": "peg"}},
            {"id": "t3", "function_name": "pick_grasp", "sequence_index": 1,
             "params": {"resource_id": "arm_b", "origin_resource_location": "cell", "part_name": "gear"}}],
        "resource_states": {"arm_a": {"resource_state": "picked"}, "arm_b": {"resource_state": "idle"}},
        "physical_evidence": {
            "resources": {"arm_a": {"attachment": {"model_name": "peg"}, "controller_hold_observation": {"fault_reason": ""}},
                          "arm_b": {"attachment": {"model_name": None}, "controller_hold_observation": {"fault_reason": ""}}},
            "parts": {"peg": {"name": "peg", "pose": [1.0, 0.0, 1.2, 0, 0, 0, 1]},
                      "gear": {"name": "gear", "pose": [0.0, 0.0, 1.1, 0, 0, 0, 1]}},
            "geometry": {"regions": {"cell": {"bounds": [[-0.2, 0.2], [-0.2, 0.2], [1.0, 1.4]]}},
                         "parts": {"gear": {"local_bounds": [[-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01]]}}},
            "physics_evidence": {"attachments": []}},
    }
    for path, value in changes.items():
        target = record
        keys = path.split("/")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return record


def rules(record: dict) -> list[str]:
    return [row["rule"] for row in grounding.ground_failure(record, settings=SETTINGS)["fault_event"]["signals"]]


def test_healthy_cell_has_no_signal():
    event = grounding.ground_failure(capture(), settings=SETTINGS)["fault_event"]
    assert event["detected"] is False and event["signals"] == [] and event["failed_task_id"] == ""


def test_cancelled_goal_and_missing_part_ground_the_interrupted_task():
    record = capture()
    record["failure"].update(controller_goal_id=[1], task_id="t2", resource_id="arm_a", goal_cancelled=True,
                             terminal_status=5, progress=0.5)
    record["physical_evidence"]["resources"]["arm_a"]["attachment"]["model_name"] = None
    record["physical_evidence"]["parts"]["peg"]["pose"] = [3.0, 3.0, 1.0, 0, 0, 0, 1]
    grounded = grounding.ground_failure(record, settings=SETTINGS)
    event = grounded["fault_event"]
    assert [row["rule"] for row in event["signals"]] == ["A1", "C1"]
    assert (event["failed_task_id"], event["failed_function_name"], event["failed_resource_jid"]) == ("t2", "place_approach", "jid-a")
    assert grounded["parts"]["peg"] == {"location": None, "current_process": None, "observed_pose": {"x": 3.0, "y": 3.0, "z": 1.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}}
    assert grounded["resources"]["arm_a"]["des_consistent"] is False
    assert "failure_type" not in event


def test_interrupted_job_is_a_task_outcome_and_running_job_is_not():
    record = capture(processing_observations={"cell": {"jobs": {
        "j1": {"task_id": "t9", "job_state": "interrupted", "process_completed": False, "part_name": "gear"},
        "j2": {"task_id": "t8", "job_state": "running", "process_completed": False, "part_name": "peg"}}}})
    event = grounding.ground_failure(record, settings=SETTINGS)["fault_event"]
    assert [(row["rule"], row["task_id"]) for row in event["signals"]] == [("A2", "t9")]
    assert event["faulted_resources"] == ["cell"] and event["failed_task_id"] == "t9"


@pytest.mark.parametrize("reported, state", [(None, "interrupted"), ("failed", "failed")])
def test_a_faulted_machine_is_grounded_as_its_own_job_record_reports_it(reported, state):
    # The machine's job of the part it holds ended interrupted: its observed fault state, unless the device
    # reports a fault state itself. The part's unfinished job on that machine carries the same state.
    record = capture(**{
        "failure/resource_values_before/cell": {"resource_state": "running", "part_name": "gear"},
        "failure/part_tracker_before/gear": {"location": "cell", "current_process": {
            "process": "trim", "result": "square", "status": "running", "machine": "cell"}},
        "processing_observations": {"cell": {"active_job": None, "selected_program": "square", "jobs": {
            "j1": {"task_id": "t9", "job_state": "interrupted", "process_completed": False, "part_name": "gear"}}}}})
    if reported:
        record["resource_states"]["cell"] = {"resource_state": reported}
    grounded = grounding.ground_failure(record, settings=SETTINGS)
    assert grounded["resources"]["cell"]["resource_state"] == state
    assert grounded["parts"]["gear"]["current_process"]["status"] == "interrupted"
    assert {"resource_id": "cell", "field": "resource_state", "value": state} in grounded["belief_updates"]


def test_a_faulted_machine_whose_record_shows_the_job_completed_grounds_the_completion():
    # The device reports a fault after the job completed but before the DES saw it: the part is grounded as
    # complete_job leaves it, its current process cleared and its result recorded.
    record = capture(**{
        "failure/resource_values_before/cell": {"resource_state": "running", "part_name": "gear"},
        "failure/part_tracker_before/gear": {"location": "cell", "processCompleted": [], "current_process": {
            "process": "trim", "result": "square", "status": "running", "machine": "cell"}},
        "processing_observations": {"cell": {"active_job": None, "selected_program": "square", "jobs": {
            "j1": {"task_id": "t9", "job_state": "completed", "process_completed": True, "part_name": "gear"}}}}})
    record["resource_states"]["cell"] = {"resource_state": "failed"}
    grounded = grounding.ground_failure(record, settings=SETTINGS)
    gear = grounded["parts"]["gear"]
    assert (gear["current_process"], gear["processCompleted"]) == (None, [{"process": "trim", "result": "square"}])
    assert {"part_name": "gear", "current_process": None,
            "processCompleted": [{"process": "trim", "result": "square"}]} in grounded["belief_updates"]


@pytest.mark.parametrize("change,rule", [
    ({"unavailable_resources": ["arm_b"]}, "B2"),
    ({"resource_states/arm_b": {"resource_state": "failed"}}, "B2"),
    ({"physical_evidence/resources/arm_b/controller_hold_observation": {"fault_reason": "joint limit"}}, "B1"),
])
def test_device_fault_reports_ground_the_first_task_naming_the_resource(change, rule):
    event = grounding.ground_failure(capture(**change), settings=SETTINGS)["fault_event"]
    assert [row["rule"] for row in event["signals"]] == [rule]
    assert (event["failed_task_id"], event["blocked_tasks"], event["failed_resource_jid"]) == ("t3", ["t3"], "jid-b")


def capture_v2(**changes) -> dict:
    """The same cell in schema 2: pending work comes from the planner's own report."""
    record = capture(**changes)
    record.update(schema_version=2, run_id="run", acknowledgements={})
    record["failure"].update(continuations={}, pending_tasks=[
        {"task_id": node["id"], "resource_id": node["params"]["resource_id"], "event_id": node["function_name"],
         "event_name": node["function_name"], "parameters": node["params"]} for node in record.pop("pending_nodes")])
    return record


def test_schema_two_reads_device_fault_reports_instead_of_beliefs():
    labelled = capture_v2(**{"resource_states/arm_b": {"resource_state": "failed"}})
    assert grounding.ground_failure(labelled, settings=SETTINGS)["fault_event"]["signals"] == []
    grounded = grounding.ground_failure(capture_v2(fault_reports={"arm_b": {"reported_state": "failed"}}),
                                        settings=SETTINGS)
    event = grounded["fault_event"]
    assert [(row["rule"], row["evidence"]) for row in event["signals"]] == [("B2", "/fault_reports/*/reported_state")]
    assert (event["failed_task_id"], event["failed_resource_jid"]) == ("t3", "jid-b")
    assert grounded["belief_updates"] == [{"resource_id": "arm_b", "field": "resource_state", "value": "failed"}]
    unavailable = grounding.ground_failure(capture_v2(fault_reports={"arm_b": {"available": False}}), settings=SETTINGS)
    assert [row["values"] for row in unavailable["fault_event"]["signals"]] == [{"available": False}]
    assert unavailable["belief_updates"] == []


def test_a_condition_the_device_reports_is_passed_through_in_its_own_words():
    report = {"arm_b": {"reported_state": "failed", "source": "gripper object detection", "gripper": "not working",
                        "reported_at_unix": 1.0}}
    grounded = grounding.ground_failure(capture_v2(fault_reports=report), settings=SETTINGS)
    signal, = grounded["fault_event"]["signals"]
    assert signal["values"] == {"resource_state": "failed", "gripper": "not working",
                                "reported_by": "gripper object detection"}
    assert grounding.prompt_fault_event(grounded)["detected_signals"][0]["gripper"] == "not working"
    # A reported condition is a fact for the prompt; it does not change what the grounding identifies.
    assert grounding.fault_identity(grounded) == grounding.fault_identity(
        grounding.ground_failure(capture_v2(fault_reports={"arm_b": {"reported_state": "failed"}}), settings=SETTINGS))


def test_a_reported_condition_in_configured_words_is_the_model_condition():
    settings = {**SETTINGS, "fault_report_conditions": {"gripper": {"not working": {"gripper_state": "failed"}}}}
    report = {"arm_b": {"reported_state": "failed", "gripper": "not working"}}
    grounded = grounding.ground_failure(capture_v2(fault_reports=report), settings=settings)
    assert grounded["resources"]["arm_b"]["gripper_state"] == "failed"
    # The live valuation declares no gripper field, so the condition is not a belief update.
    assert [row["field"] for row in grounded["belief_updates"] if "field" in row] == ["resource_state"]
    unlisted = {"arm_b": {"reported_state": "failed", "gripper": "slipping"}}
    assert "gripper_state" not in grounding.ground_failure(
        capture_v2(fault_reports=unlisted), settings=settings)["resources"]["arm_b"]
    # The configured words reach the projected state through the prepared request.
    broken = {"arm_b": {"reported_state": "failed", "gripper": "not working (hardware fault, needs repair)"}}
    prepared = {"llm_input": {"observed_runtime_state": {"resources": [
        {"resource_jid": "jid-b", "current_state": "idle", "gripper_state": "open"}]}}}
    grounding.apply_grounding(prepared, capture_v2(fault_reports=broken))
    assert prepared["llm_input"]["observed_runtime_state"]["resources"][0]["gripper_state"] == "failed"

def test_the_reporting_device_is_passed_through_without_a_derived_label():
    report = {"arm_b": {"reported_state": "failed", "source": "gripper object detection"}}
    grounded = grounding.ground_failure(capture_v2(fault_reports=report), settings=SETTINGS)
    signal, = grounded["fault_event"]["signals"]
    assert signal["values"] == {"resource_state": "failed", "reported_by": "gripper object detection"}
    assert grounding.prompt_fault_event(grounded)["detected_signals"] == [
        {"signal": "device_fault_report", "resource_id": "arm_b", "resource_state": "failed",
         "reported_by": "gripper object detection"}]
    assert grounding.fault_identity(grounded) == grounding.fault_identity(
        grounding.ground_failure(capture_v2(fault_reports={"arm_b": {"reported_state": "failed"}}), settings=SETTINGS))


def test_grounded_beliefs_update_the_live_context_and_the_recorded_copies():
    import threading
    from types import SimpleNamespace

    record = capture_v2(fault_reports={"arm_a": {"reported_state": "failed"}},
                        part_tracker={"peg": {"location": "arm_a", "current_process": None}},
                        resource_states={"arm_a": {"resource_state": "picked", "held_part": "peg"}})
    record["failure"].update(controller_goal_id=[1], task_id="t2", resource_id="arm_a", goal_cancelled=True,
                             terminal_status=5, progress=0.5)
    record["physical_evidence"]["resources"]["arm_a"]["attachment"]["model_name"] = None
    record["physical_evidence"]["parts"]["peg"]["pose"] = [3.0, 3.0, 1.0, 0, 0, 0, 1]
    context = SimpleNamespace(
        admission_lock=threading.Lock(), revision=4,
        resources={"arm_a": SimpleNamespace(valuation={"resource_state": "picked", "held_part": "peg"}, revision=1)},
        part_tracker={"peg": {"location": "arm_a", "current_process": None}})
    grounded = grounding.ground_failure(record, settings=SETTINGS)
    assert [row["rule"] for row in grounded["fault_event"]["signals"]] == ["A1", "B2", "C1"]
    assert grounding.update_beliefs(context, record, grounded) == grounded["belief_updates"]
    assert context.resources["arm_a"].valuation == record["resource_states"]["arm_a"] == {
        "resource_state": "failed", "held_part": None}
    assert context.part_tracker["peg"] == record["part_tracker"]["peg"] == {
        "location": None, "current_process": None, "observed_pose": {"x": 3.0, "y": 3.0, "z": 1.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}}
    assert (context.revision, context.resources["arm_a"].revision) == (5, 3)
    assert grounding.update_beliefs(context, record, grounding.ground_failure(capture_v2(), settings=SETTINGS)) == []
    assert context.revision == 5


def test_a_missing_robot_observation_is_not_an_empty_gripper():
    record = capture()
    del record["physical_evidence"]["resources"]["arm_a"]
    assert rules(record) == []
    record["physical_evidence"]["resources"]["arm_a"] = {"attachment": {"model_name": None}}
    assert rules(record) == ["C1"]


def test_loose_part_outside_its_expected_region_is_a_discrepancy():
    record = capture()
    record["physical_evidence"]["parts"]["gear"]["pose"] = [0.5, 0.5, 1.1, 0, 0, 0, 1]
    assert rules(record) == ["C2"]
    record["physical_evidence"]["physics_evidence"]["attachments"] = [{"model1": "cell_fixture", "model2": "gear"}]
    assert rules(record) == []


def test_settings_are_honored():
    record = capture(**{"resource_states/arm_b": {"resource_state": "failed"}})
    assert grounding.ground_failure(record, settings={**SETTINGS, "fault_report_values": []})["fault_event"]["signals"] == []
    record = capture()
    record["physical_evidence"]["parts"]["gear"]["pose"] = [0.25, 0.0, 1.1, 0, 0, 0, 1]
    assert rules(record) == ["C2"]
    assert grounding.ground_failure(record, settings={**SETTINGS, "region_margin_m": 0.05})["fault_event"]["signals"] == []
    assert set(SETTINGS) <= set(grounding.load_grounding_settings())


def test_transition_log_disagreement_is_reported():
    record = capture()
    record["transitions"][0]["after"]["arm_a"] = {"resource_state": "failed", "held_part": None}
    conflicts = grounding.ground_failure(record, settings=SETTINGS)["expected_state_conflicts"]
    assert {(row["field"], row["reported"], row["transition_log"]) for row in conflicts} == {
        ("resource_state", "picked", "failed"), ("held_part", "peg", None)}


@pytest.mark.parametrize("case", ["arm", "mobile base", "not mid-activity"])
def test_a_resource_faulted_before_its_activity_finish_set_its_location_is_at_its_measured_pose(case, monkeypatch):
    from cais_spade_llm.resources import environment_models

    record = capture_v2(fault_reports={"arm_a": {"reported_state": "failed"}})
    running = "at_pick" if case == "not mid-activity" else "approaching_place"
    record["failure"]["resource_values_before"]["arm_a"].update(resource_state=running, resource_location="home")
    record["inputs"] = {"scene": {}}
    # The configured activity model: only the finish that leaves approaching_place sets the location.
    finish = {"event_name": "place_approach_done", "start_event_id": "place_approach",
              "guards": {"resource_state": {"equals": "approaching_place"}},
              "updates": {"resource_state": {"set": "positioned"}, "resource_location": {"set": "board"}}}
    monkeypatch.setattr(environment_models, "build_environment_models", lambda scene: {"arm_a": {"events": [finish]}})
    measured = record["physical_evidence"]["resources"]["arm_a"]
    measured["current_pose"] = [0.6, 0.1, 1.3, 0.0, 1.0, 0.0, 0.0]
    if case == "mobile base":
        measured["base_pose"] = [2.0, 1.0, 0.0, 0.0, 0.0, 0.7071068, 0.7071068]
    grounded = grounding.ground_failure(record, settings=SETTINGS)
    arm_a = grounded["resources"]["arm_a"]
    if case == "not mid-activity":
        assert arm_a["resource_location"] == "home" and "current_pose" not in arm_a
        return
    assert arm_a["resource_location"] is None
    if case == "mobile base":
        assert arm_a["current_pose"] == pytest.approx({"x": 2.0, "y": 1.0, "yaw": 1.5707963}, abs=1e-6)
    else:
        assert arm_a["current_pose"] == {"x": 0.6, "y": 0.1, "z": 1.3, "qx": 0.0, "qy": 1.0, "qz": 0.0, "qw": 0.0}
    assert {"resource_id": "arm_a", "field": "resource_location", "value": None} in grounded["belief_updates"]
    # The prompt and the planning session start from the measured pose, not the place it set out from.
    prepared = {"llm_input": {"observed_runtime_state": {"resources": [
        {"resource_jid": "jid-a", "current_state": running, "held_part": "peg",
         "resource_location": "home", "current_location": "home"}]}}}
    grounding.apply_grounding(prepared, record)
    row, = prepared["llm_input"]["observed_runtime_state"]["resources"]
    assert (row["resource_location"], row["current_location"]) == (None, None)
    assert row["current_pose"] == arm_a["current_pose"]


def test_work_the_stop_cancelled_leaves_its_resource_where_it_stopped():
    record = capture_v2(fault_reports={"arm_a": {"reported_state": "failed"}})
    # arm_b had started t3 when the stop cancelled it; its own worker reports the cancellation.
    record["failure"]["resource_values_before"]["arm_b"] = {"resource_state": "travelling", "held_part": None,
                                                           "resource_location": "cell"}
    before = {"resource_state": "idle", "held_part": None, "resource_location": "cell"}
    record["transitions"].append({"kind": "start", "acknowledgement": {"task_id": "t3", "revision": 2},
                                  "before": {"arm_b": before}, "after": {"arm_b": {**before, "resource_state": "travelling"}}})
    record["physical_evidence"]["resources"]["arm_b"]["base_pose"] = [2.0, 1.0, 0, 0, 0, 0.7071068, 0.7071068]
    worker = {"cancelled": True, "status": "failed", "operations": [{"operation": "move", "success": False, "status": "canceled"}],
              "primitive_results": [{"task_id": "t3", "effects": {"base_pose": "observed"}}]}
    record["checkpoint_motion_cancellation"] = {"status": "observed_stopped", "resources": {"arm_b": {"worker_result": worker}}}
    grounded = grounding.ground_failure(record, settings=SETTINGS)
    event = grounded["fault_event"]
    assert [row["rule"] for row in event["signals"]] == ["B2", "A3"]
    # The cancellation is an outcome of the stop, not a fault: arm_a's task stays the failed one.
    assert (event["faulted_resources"], event["failed_task_id"]) == (["arm_a"], "t2")
    assert event["signals"][1]["values"] == {"function_name": "pick_grasp", "origin_resource_location": "cell",
                                             "status": "canceled"}
    arm_b = grounded["resources"]["arm_b"]
    assert (arm_b["resource_state"], arm_b["resource_location"], arm_b["held_part"]) == ("idle", None, None)
    assert arm_b["current_pose"] == pytest.approx({"x": 2.0, "y": 1.0, "yaw": 1.5707963}, abs=1e-6)
    assert [(row["field"], row["value"]) for row in grounded["belief_updates"] if row.get("resource_id") == "arm_b"] == [
        ("resource_state", "idle"), ("resource_location", None)]
    # The prompt and the planning session start from the observed row.
    prepared = {"llm_input": {"observed_runtime_state": {"resources": [
        {"resource_jid": "jid-b", "current_state": "travelling", "resource_state": "travelling", "held_part": None,
         "resource_location": "cell", "current_location": "cell"}]}}}
    grounding.apply_grounding(prepared, record)
    row, = prepared["llm_input"]["observed_runtime_state"]["resources"]
    assert (row["current_state"], row["resource_state"], row["resource_location"], row["current_location"]) == (
        "idle", "idle", None, None)
    assert row["current_pose"] == arm_b["current_pose"]
    # An earlier completed result left in the worker, or a stop never observed, cancelled nothing.
    for status, result in (("observed_stopped", {**worker, "cancelled": False}), ("cancellation_requested", worker)):
        record["checkpoint_motion_cancellation"] = {"status": status, "resources": {"arm_b": {"worker_result": result}}}
        assert rules(record) == ["B2"]


def test_rules_never_see_recorded_labels_or_the_scenario_name():
    record = capture(part_tracker={"gear": {"state": "misplaced"}}, setup={"failure_scenario": "x"},
                     prepared_recovery_request={"llm_input": {"fault_event": {"failure_type": "Supplied scenario name"}}})
    view = json.dumps(grounding.raw_failure_view(record, SETTINGS))
    assert "misplaced" not in view and "Supplied scenario name" not in view and "failure_scenario" not in view


OFFICIAL = {
    # Restored ur5e-3 route and URs without collision checks (2026-10-24).
    "capture_ad64bf1628564c3fa7f928b7bdebd5d9": (["A1", "B2", "C1"], ["ur5e-4"], "environment_8", "place_approach"),
    # The stop cancelled the KMR's trip environment_22 (A3); M1 stays the only faulted resource.
    "capture_37e090a54d0242f4948ae80c95d3cd76": (["A2", "B2", "A3"], ["M1"], "environment_25", "complete_job"),
    "capture_3ad495d2bd7845269fa70f1a09963bc9": (["B2"], ["Conveyor"], "cf4b66ddc51348a2987d5207878e5eea_pending_0", "place_approach"),
    "capture_b7dea70c1db74e1e9e98849380c09655": (["B2"], ["ur5e-1"], "67689d8a6f1a4b9c87a47a16f000e313_pending_1", "pick_grasp"),
}


@pytest.mark.parametrize("identifier", sorted(OFFICIAL))
def test_official_captures_ground_from_raw_facts(identifier):
    path = RECORDS / identifier / "observations.json"
    if not path.is_file():
        pytest.skip("retained capture is not present")
    observations = read_observations(path)
    view = json.dumps(grounding.raw_failure_view(observations, SETTINGS))
    assert "misplaced" not in view and observations["scenario"] not in view
    event = grounding.ground_failure(deepcopy(observations), settings=SETTINGS)["fault_event"]
    expected_rules, resources, task, function = OFFICIAL[identifier]
    assert [row["rule"] for row in event["signals"]] == expected_rules
    assert (event["faulted_resources"], event["failed_task_id"], event["failed_function_name"]) == (resources, task, function)


def test_grounding_replaces_stored_failure_summaries_in_a_prepared_request():
    record = capture(**{"resource_states/arm_b": {"resource_state": "failed"}})
    prepared = {"llm_input": {"fault_event": {"failure_type": "Supplied scenario name", "checkpoint": {"name": "label"},
                                              "blocked_at_task_id": ""},
                              "part_facts": [{"part_name": "gear", "current_state": "stale", "current_location": "x"}],
                              "observed_runtime_state": {"resources": [{"resource_jid": "jid-b", "current_state": "idle",
                                                                        "held_part": None}]}},
                "context_summary": {"fault_event": {"failure_type": "Supplied scenario name"}},
                "recovery_debug": {"context_summary": {"fault_event": {"failure_type": "Supplied scenario name"}}},
                "failure_anchor": {"failure_type": "Supplied scenario name", "checkpoint": {"name": "label"}}}
    grounded = grounding.apply_grounding(prepared, record)
    event = prepared["llm_input"]["fault_event"]
    assert grounded == prepared["code_grounding"]
    assert (event["focused_resource_jid"], event["blocked_at_task_id"], event["blocked_at_function"]) == ("jid-b", "t3", "pick_grasp")
    assert event["detected_signals"] == [{"signal": "device_fault_report", "resource_id": "arm_b", "resource_state": "failed"}]
    assert "Supplied scenario name" not in json.dumps(prepared)
    assert "checkpoint" not in prepared["failure_anchor"] and prepared["failure_anchor"]["failed_task_id"] == "t3"
    # A part fact states location and process record; no handling label survives grounding.
    assert "current_state" not in prepared["llm_input"]["part_facts"][0]
    assert prepared["llm_input"]["observed_runtime_state"]["resources"][0]["current_state"] == "failed"
    assert prepared["context_summary"]["fault_event"] == prepared["recovery_debug"]["context_summary"]["fault_event"] == event
    assert grounded["execution_monitoring"] == grounding.load_grounding_settings()


def test_a_stored_grounding_is_reused_only_while_its_raw_facts_still_give_it(monkeypatch):
    record = capture(**{"resource_states/arm_b": {"resource_state": "failed"}})
    prepared = {}
    grounded = grounding.apply_grounding(prepared, record)
    stored = json.loads(json.dumps(prepared))
    assert grounding.apply_grounding(stored, record) == grounded
    edited = deepcopy(stored)
    edited["code_grounding"]["fault_event"]["failed_task_id"] = "edited"
    with pytest.raises(ValueError, match="re-derive it"):
        grounding.apply_grounding(edited, record)
    changed = {**grounding.load_grounding_settings(), "region_margin_m": 9.0}
    monkeypatch.setattr(grounding, "load_grounding_settings", lambda: changed)
    with pytest.raises(ValueError, match="re-derive it"):
        grounding.apply_grounding(stored, record)


def test_storing_the_grounding_rewrites_only_the_beliefs(tmp_path):
    from cais_spade_llm.recovery_framework import capture_files

    record = capture_v2(fault_reports={"arm_b": {"reported_state": "failed"}})
    record["prepared_recovery_request"] = {"llm_input": {"fault_event": {"failure_type": "Supplied scenario name"}}}
    path = tmp_path / "observations.json"
    capture_files.write_capture(path, record)
    raw = path.read_bytes()
    stored = grounding.store_grounding(path)
    assert path.read_bytes() == raw
    prepared = capture_files.read_capture(path)["prepared_recovery_request"]
    assert prepared["code_grounding"] == stored and stored["fault_event"]["failed_task_id"] == "t3"
    assert "Supplied scenario name" not in json.dumps(prepared)
    assert grounding.store_grounding(path) == stored


def test_a_grounded_request_starts_its_session_at_the_outline():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import (
        build_multi_turn_session_seed,
    )

    prepared = {"part_tracker": {"peg": {"state": "misplaced", "location": None,
                                         "observed_pose": {"x": 3.0, "y": 3.0, "z": 1.0}}},
                "llm_input": {"part_facts": [{"part_name": "peg", "current_location": None}]}}
    seed = build_multi_turn_session_seed(prepared)
    assert (seed["current_phase"], seed["observation_store"]) == ("grounding", {})
    seed = build_multi_turn_session_seed({**prepared, "code_grounding": {"fault_event": {}}})
    assert seed["current_phase"] == "outline" and seed["turns"] == []
    assert seed["observation_store"] == {"observed_pose_peg": {"part_name": "peg", "pose": {"x": 3.0, "y": 3.0, "z": 1.0},
                                                               "x": 3.0, "y": 3.0, "z": 1.0}}


def test_grounding_keeps_legacy_captures_and_requires_a_detectable_fault():
    legacy = capture()
    legacy["failure"] = {}
    prepared = {"llm_input": {"fault_event": {"blocked_at_task_id": "stored"}}}
    assert grounding.apply_grounding(prepared, legacy) is None
    assert prepared == {"llm_input": {"fault_event": {"blocked_at_task_id": "stored"}}}
    with pytest.raises(ValueError, match="NEEDS_CONTEXT: no failure is detectable"):
        grounding.apply_grounding({}, capture())


def test_scenario_name_is_not_forwarded_to_the_prompt():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _slim_fault_event,
    )

    compact = _slim_fault_event({"failure_type": "Supplied scenario name", "detected_signals": [{"signal": "x"}]})
    assert compact == {"detected_signals": [{"signal": "x"}]}


def test_plant_model_accepts_only_declared_fault_states():
    from cais_spade_llm.resources import environment_models

    path = RECORDS / "capture_ad64bf1628564c3fa7f928b7bdebd5d9" / "observations.json"
    if not path.is_file():
        pytest.skip("retained capture is not present")
    physical = read_observations(path)["physical_evidence"]
    arguments = (physical["environment_models"], physical["environment_valuation"], physical["products"], physical["product_name"])
    assert physical["environment_valuation"]["ur5e-4"]["resource_state"] in environment_models.declared_fault_states()
    environment_models.owners(*arguments)
    physical["environment_valuation"]["ur5e-4"]["resource_state"] = "undeclared_state"
    with pytest.raises(ValueError, match="Invalid resource value: ur5e-4.resource_state"):
        environment_models.owners(*arguments)


def test_the_first_turn_is_the_outline_prompt_with_the_grounded_fault(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from cais_spade_llm.recovery_framework import outline_draft
    from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files

    path = RECORDS / "capture_ad64bf1628564c3fa7f928b7bdebd5d9" / "observations.json"
    if not path.is_file():
        pytest.skip("retained capture is not present")
    observations = read_observations(path)
    provider = AsyncMock(side_effect=RuntimeError("stop after the first LLM prompt"))
    monkeypatch.setattr(outline_draft.SavedRecoveryProduct, "ask_llm_structured", provider)
    draft = outline_draft.OutlineDraft(observations, default_safety_files("Part slippage"), root=tmp_path)
    asyncio.run(draft.generate())
    prompt = provider.call_args_list[0].kwargs.get("prompt") or provider.call_args_list[0].args[0]
    assert "\nFailure\n" in prompt and provider.await_count == 1
    assert "- ur5e-4 did not complete place_approach of gear_small (" in prompt and "task environment_8" in prompt
    assert "Part slippage" not in prompt and "during_place_lowering" not in prompt
    assert draft.record["fault_grounding"]["failed_task_id"] == "environment_8"
    session = draft.prepared["multi_turn_session_state"]
    assert (session["turn_index"], session["current_phase"], session["turns"]) == (1, "outline", [])
    assert not list(draft.directory.glob("recovery_outline/turns/*grounding*"))
