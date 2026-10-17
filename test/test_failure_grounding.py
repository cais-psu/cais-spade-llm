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
            "part_tracker_before": {"peg": {"state": "in_gripper", "location": "arm_a"},
                                    "gear": {"state": "ready", "location": "cell"}},
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
    assert grounded["parts"]["peg"] == {"state": "misplaced", "location": None, "observed_pose": {"x": 3.0, "y": 3.0, "z": 1.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}}
    assert grounded["resources"]["arm_a"]["des_consistent"] is False
    assert "failure_type" not in event


def test_interrupted_job_is_a_task_outcome_and_running_job_is_not():
    record = capture(processing_observations={"cell": {"jobs": {
        "j1": {"task_id": "t9", "job_state": "interrupted", "process_completed": False, "part_name": "gear"},
        "j2": {"task_id": "t8", "job_state": "running", "process_completed": False, "part_name": "peg"}}}})
    event = grounding.ground_failure(record, settings=SETTINGS)["fault_event"]
    assert [(row["rule"], row["task_id"]) for row in event["signals"]] == [("A2", "t9")]
    assert event["faulted_resources"] == ["cell"] and event["failed_task_id"] == "t9"


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
                        part_tracker={"peg": {"state": "in_gripper", "location": "arm_a"}},
                        resource_states={"arm_a": {"resource_state": "picked", "held_part": "peg"}})
    record["failure"].update(controller_goal_id=[1], task_id="t2", resource_id="arm_a", goal_cancelled=True,
                             terminal_status=5, progress=0.5)
    record["physical_evidence"]["resources"]["arm_a"]["attachment"]["model_name"] = None
    record["physical_evidence"]["parts"]["peg"]["pose"] = [3.0, 3.0, 1.0, 0, 0, 0, 1]
    context = SimpleNamespace(
        admission_lock=threading.Lock(), revision=4,
        resources={"arm_a": SimpleNamespace(valuation={"resource_state": "picked", "held_part": "peg"}, revision=1)},
        part_tracker={"peg": {"state": "in_gripper", "location": "arm_a"}})
    grounded = grounding.ground_failure(record, settings=SETTINGS)
    assert [row["rule"] for row in grounded["fault_event"]["signals"]] == ["A1", "B2", "C1"]
    assert grounding.update_beliefs(context, record, grounded) == grounded["belief_updates"]
    assert context.resources["arm_a"].valuation == record["resource_states"]["arm_a"] == {
        "resource_state": "failed", "held_part": None}
    assert context.part_tracker["peg"] == record["part_tracker"]["peg"] == {
        "state": "misplaced", "location": None, "observed_pose": {"x": 3.0, "y": 3.0, "z": 1.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}}
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


def test_rules_never_see_recorded_labels_or_the_scenario_name():
    record = capture(part_tracker={"gear": {"state": "misplaced"}}, setup={"failure_scenario": "x"},
                     prepared_recovery_request={"llm_input": {"fault_event": {"failure_type": "Supplied scenario name"}}})
    view = json.dumps(grounding.raw_failure_view(record, SETTINGS))
    assert "misplaced" not in view and "Supplied scenario name" not in view and "failure_scenario" not in view


OFFICIAL = {
    "capture_f923020bf9ce496eb443826291f8f2c2": (["A1", "B2", "C1"], ["ur5e-4"], "environment_9", "place_approach"),
    "capture_8beafdb8a01e444894cfaf41d29921fd": (["A2", "B2"], ["M1"], "environment_21", "machine_part"),
    "capture_1b3f3d1d729a4701b94cff514c783130": (["B2"], ["Conveyor"], "d49d4e9e71354325b11328a1cee4a5ce_pending_2", "place_approach"),
    "capture_0a7fde8074b6426586b204204f2b3417": (["B2"], ["ur5e-1"], "01eeb91143e741c69242142690a5a4be_pending_0", "pick_grasp"),
    "capture_77666affc1b9468ca157cf366d706dc8": (["A1", "B2", "C1"], ["ur5e-4"], "environment_9", "place_approach"),
    "capture_4d1e4cc0fd9b4e07812ce7cbb01d2c87": (["A2", "B2"], ["M1"], "environment_21", "machine_part"),
    "capture_3aed616530294a4bbdf353000fc69596": (["B2"], ["Conveyor"], "5a764b64795247fe927cef6fc1ef4d72_pending_0", "place_approach"),
    "capture_d1ca51654ff7423193d1bd13b146964d": (["B2"], ["ur5e-1"], "cb6f780418494c8589b00b8ddc759ec7_pending_1", "pick_grasp"),
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
    assert prepared["llm_input"]["part_facts"][0]["current_state"] == "ready"
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

    path = RECORDS / "capture_f923020bf9ce496eb443826291f8f2c2" / "observations.json"
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

    path = RECORDS / "capture_f923020bf9ce496eb443826291f8f2c2" / "observations.json"
    if not path.is_file():
        pytest.skip("retained capture is not present")
    observations = read_observations(path)
    provider = AsyncMock(side_effect=RuntimeError("stop after the first LLM prompt"))
    monkeypatch.setattr(outline_draft.SavedRecoveryProduct, "ask_llm_structured", provider)
    draft = outline_draft.OutlineDraft(observations, default_safety_files("Part slippage"), root=tmp_path)
    asyncio.run(draft.generate())
    prompt = provider.call_args_list[0].kwargs.get("prompt") or provider.call_args_list[0].args[0]
    assert "Recovery Event Candidate Proposal" in prompt and provider.await_count == 1
    assert '"detected_signals"' in prompt and "Part slippage" not in prompt and "during_place_lowering" not in prompt
    assert draft.record["fault_grounding"]["failed_task_id"] == "environment_9"
    session = draft.prepared["multi_turn_session_state"]
    assert (session["turn_index"], session["current_phase"], session["turns"]) == (1, "outline", [])
    assert not list(draft.directory.glob("recovery_outline/turns/*grounding*"))
