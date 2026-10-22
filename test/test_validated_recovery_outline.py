from __future__ import annotations

"""Outline event form, response schemas, prompt feedback and completion conditions."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_diagnostic
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import multi_turn_phase_response_schema

JID = "ur5e-3@localhost"


@pytest.mark.parametrize("resource_list,part_list", [(False, False), (True, False), (False, True), (True, True)])
def test_candidate_comparison_accepts_recorded_and_projected_pose_formats(resource_list, part_list):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_outline_state import (
        _infer_outline_macro_signature,
    )

    values = [1., 2., 3., 0., 0., 0., 1.]
    pose = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), values, strict=True))
    resources = {JID: {"current_state": "idle", "current_location": "M2", "held_part": None, "current_pose": pose}}
    parts = {"part": {"current_state": "ready", "current_location": "M2", "observed_pose": pose}}
    task = {"resource_jid": JID, "part_name": "part", "expected_end_state": {"held_part": "part"}}
    expected = _infer_outline_macro_signature(task, resources_by_jid=resources, parts_by_name=parts)
    if resource_list:
        resources[JID]["current_pose"] = values
    if part_list:
        parts["part"]["observed_pose"] = values
    before = deepcopy((task, resources, parts))
    assert _infer_outline_macro_signature(task, resources_by_jid=resources, parts_by_name=parts) == expected
    assert (task, resources, parts) == before


@pytest.mark.parametrize("pose", [None, [], [1., 2., 3.], [1., 2., float("nan"), 0., 0., 0., 1.]])
def test_candidate_comparison_keeps_incomplete_pose_unknown(pose):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_outline_state import (
        _infer_outline_macro_signature,
    )

    result = _infer_outline_macro_signature({"resource_jid": JID, "expected_end_state": {}},
        resources_by_jid={JID: {"current_pose": pose}}, parts_by_name={})
    assert result["resource_before"]["current_pose"] is None


def _case():
    snapshot = {"resource_jid": JID, "resource_state": "failed", "current_state": "failed", "held_part": None}
    primitive = {"name": "reset_fault", "params": {}, "preconditions": {
        "resource_state": {"equals": "failed"}}, "effects": {"resource_state": {"set": "idle"}}}
    model = {"resource_type": "robot", "state_variables": {
        "resource_state": {"scope": "resource", "type": "string", "domain": ["failed", "idle"]},
        "held_part": {"scope": "resource", "domain": [None, "gear_small"]}},
        "events": [], "primitive_catalog": [primitive]}
    command = {"primitive": "reset_fault", "params": {}}
    descriptor = {"id": "recorded_test_owner", "version": 1,
                  "configuration": {"geometry_unchanged": [command]}}
    physical = {"snapshot": {"frame": "world", "resources": {
        rid: {"current_pose": [index * 3., 0., 0., 0., 0., 0., 1.], "held_part": None}
        for index, rid in enumerate(("ur5e-3", "ur5e-4"))}, "parts": {}},
        "geometry": {"frame": "world", "resources": {
            rid: {"component_bounds": [{"id": rid, "bounds": [[index * 3. - .1, index * 3. + .1], [-.1, .1], [-.1, .1]]}]}
            for index, rid in enumerate(("ur5e-3", "ur5e-4"))}, "parts": {},
            "regions": {"assembly_board-v1": {"bounds": [[.8, 1.2], [-.2, .2], [-.2, .2]]}}},
        "resource_models": {"ur5e-3": {"primitive_model": descriptor}},
        "jids": {"ur5e-3": JID, "ur5e-4": "ur5e-4@localhost"}, "step_evidence": []}
    path = Path(__file__).resolve().parents[1] / "cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt"
    document = json.loads(path.read_text())
    prepared = {"validation_contract_version": 2, "offline_recovery": True,
        "source_context_revision": 4, "observation_revision": 4,
        "recovery_resources": {JID: {"recovery_snapshot": deepcopy(snapshot), "recovery_des_model": model}},
        "physical_evidence": physical, "saved_pending_nodes": [],
        "outline_safety_specifications": [{"document": document, "source_path": str(path), "source_sha256": "recorded-rule"}],
        "llm_input": {"goal_conditions": [{"entity_kind": "resource", "entity": JID,
                                           "field": "resource_state", "expected": "idle"}]}}
    session = {"session_id": "single-event", "turn_index": 2, "validation_contract_version": 2,
        "symbolic_resources": {JID: deepcopy(snapshot)}, "symbolic_parts": {},
        "accepted_outline_prefix": [], "recovery_des_models": {JID: model},
        "outline_monitor_context": {}}
    event = {"outline_id": "RECOVERY1", "event_name": "unlisted_recovery_event", "resource_jid": JID,
             "expected_end_state": {"resource_state": "idle"}, "rationale": "Restore the interrupted resource state."}
    return prepared, session, event


def test_candidate_state_comparison_ignores_history_length_but_preserves_relevant_changes():
    _, session, _ = _case()
    session['outline_monitor_context'] = {'spec': {
        'states': {'SAFE': ['q0']}, 'rules_fingerprint': 'exact-rule', 'observations': [{'time': 0}]}}
    identity = multi_turn_diagnostic._comparison_state_fingerprint(session)
    same = deepcopy(session)
    same['outline_monitor_context']['spec']['observations'].append({'time': 1})
    same['outline_monitor_context']['spec']['evidence_history'] = [{'projection': 'another record'}]
    assert multi_turn_diagnostic._comparison_state_fingerprint(same) == identity
    for field, value in [('held_part', 'gear_small'), ('resource_state', 'picked')]:
        changed = deepcopy(same)
        changed['symbolic_resources'][JID][field] = value
        assert multi_turn_diagnostic._comparison_state_fingerprint(changed) != identity
    same['outline_monitor_context']['spec']['states']['SAFE'] = ['q1']
    assert multi_turn_diagnostic._comparison_state_fingerprint(same) != identity


def test_candidate_outline_task_accepts_well_formed_event():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import _derive_candidate_outline_task

    prepared, session, event = _case()
    _, findings = _derive_candidate_outline_task(candidate_task=event, session_state=session,
                                                prepared_recovery_request=prepared)
    assert findings == []


@pytest.mark.parametrize("resource_jid", [JID, "M2@localhost", "Conveyor@localhost"])
def test_recent_accepted_prefix_shows_only_changed_public_conditions(resource_jid):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _accepted_transition_prefix_for_prompt

    tasks = [{"outline_id": str(index), "event_name": "accepted_event", "resource_jid": resource_jid,
        "expected_start_state": {"resource_state": "idle", "held_part": "gear_small", "_private": "before"},
        "expected_end_state": {"resource_state": "idle", "held_part": None, "_private": "after"},
        "primitive_steps": ["private operation"], "rationale": "private explanation"} for index in range(9)]
    original = deepcopy(tasks)
    rows = _accepted_transition_prefix_for_prompt(tasks, hide_registered_functions=True,
        public_fields={resource_jid: {"resource_state", "held_part"}})
    assert len(rows) == len(tasks)
    assert all("expected_end_state" not in row for row in rows[:3])
    assert all(row["expected_end_state"] == {"held_part": None} for row in rows[3:])
    assert "private" not in json.dumps(rows)
    assert tasks == original
    assert all("expected_end_state" not in row for row in _accepted_transition_prefix_for_prompt(tasks))


def test_outline_response_schema_accepts_event_for_open_declared_field():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import _get_response_schema

    _, session, event = _case()
    session["outline_mode"] = "incremental_validated"
    session["recovery_des_models"][JID]["state_variables"]["held_part"].update(
        type=["string", "null"], domain_complete=False)
    event["expected_end_state"] = {"held_part": "gear_small"}
    schema = _get_response_schema("outline", session)["schema"]
    Draft202012Validator(schema).validate({"thought": "Recorded declaration fixture", "next_transition": event})


@pytest.mark.parametrize("outline_mode", ["incremental", "incremental_candidates_validated", "single_pass"])
def test_versioned_schema_keeps_legacy_suffix_only_for_legacy_attempts(outline_mode):
    declarations = _case()[0]["recovery_resources"][JID]["recovery_des_model"]["state_variables"]
    options = {"outline_context_retrieval": True, "declared_state_variables_by_resource_jid": {JID: declarations}}
    legacy = multi_turn_phase_response_schema("outline", **options)
    current = multi_turn_phase_response_schema("outline", validation_contract_version=2, outline_mode=outline_mode, **options)
    Draft202012Validator.check_schema(current["schema"])
    assert "transition_suffix" in legacy["schema"]["properties"]
    assert "transition_suffix" not in current["schema"]["properties"]
    field = "candidate_events" if outline_mode == "incremental_candidates_validated" else "next_transition"
    assert field in current["schema"]["properties"]
    assert "anyOf" not in current["schema"]
    assert current["strict"] is False


@pytest.mark.parametrize("outline_mode", ["incremental", "incremental_candidates_validated", "single_pass"])
def test_contract_2_outline_event_gives_only_its_form(outline_mode):
    declarations = _case()[0]["recovery_resources"][JID]["recovery_des_model"]["state_variables"]
    schema = multi_turn_phase_response_schema("outline", validation_contract_version=2, outline_mode=outline_mode,
                                              declared_state_variables_by_resource_jid={JID: declarations})
    assert list(schema["schema"]["$defs"]) == ["outline_event"]
    event = schema["schema"]["$defs"]["outline_event"]
    assert not {"anyOf", "allOf"} & event.keys()
    assert event["properties"]["expected_end_state"] == {"type": "object", "minProperties": 1}
    assert event["properties"]["part_name"] == {"type": ["string", "null"], "minLength": 1}
    # Holding a part without stating its location fits the form; the outline turn judges the names.
    Draft202012Validator(schema["schema"]["$defs"]["outline_event"]).validate({
        "outline_id": "local_fixture", "event_name": "pick_part", "resource_jid": JID, "part_name": "part_a",
        "expected_end_state": {"held_part": "part_a"}, "rationale": "Local schema check"})


def test_nominal_archive_binding_includes_retained_task_completions():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import nominal_continuation_context_binding

    prepared, session, _ = _case()
    before = nominal_continuation_context_binding(session, prepared)
    session["nominal_task_completion_records"] = [{"source_task_id": "recorded-nominal", "fingerprint": "verified-record"}]
    assert nominal_continuation_context_binding(session, prepared) != before


def test_validated_outline_cannot_enter_legacy_diagnostic_execution():
    from cais_spade_llm.recovery_framework.saved_recovery_execution import validate_saved_proposal

    with pytest.raises(ValueError, match="does not authorize execution"):
        asyncio.run(validate_saved_proposal(SimpleNamespace(), {"validation_contract_version": 2}, {}))


@pytest.mark.parametrize("reason", [
    "pose is outside gripper_reach: xy_radius=0.8000 m, max=0.7000 m",
    "pose is outside gripper_reach: z=1.7000 m, range=[0.8500, 1.6000] m",
])
def test_owner_capability_prompt_preserves_only_supported_numeric_reasons(reason):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _outline_feedback_for_prompt

    check = {"feasibility_status": "INFEASIBLE", "reason": reason,
             "checked_pose": {"x": 0., "y": .8, "z": 1.7, "private_model": "secret"},
             "gripper_reach": {"max_xy_radius_m": .7, "private_model": "secret"},
             "private_model": "secret"}
    finding = {"constraint_code": "physical_capability_rejected", "reason": reason,
               "evidence": {"capability_result": {"feasibility_status": "INFEASIBLE", "checks": [check]}}}
    original = deepcopy(finding)
    public = _outline_feedback_for_prompt(finding, set())
    explanation = "The calculated target is outside the recorded owner motion capability."
    assert public["reason"] == explanation
    assert public["evidence"]["capability_result"]["checks"][0]["reason"] == explanation
    assert "gripper_reach" not in json.dumps(public)
    assert "secret" not in json.dumps(public)
    assert finding == original
    finding["reason"] = "private proof suggests a particular event"
    assert "private proof" not in json.dumps(_outline_feedback_for_prompt(finding, set()))
    check["reason"] += "; private proof suggests a particular event"
    assert "capability_result" not in _outline_feedback_for_prompt(finding, set())["evidence"]


def test_owner_effect_feedback_keeps_explicit_null_and_filters_private_fields():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _outline_feedback_for_prompt

    finding = {"constraint_code": "owner_effect_rejected", "evidence": {"owner_effect_result": {
        "feasibility_status": "INFEASIBLE", "conflicts": [
            {"field": "held_part", "requested": None, "derived": "KET4_Square_4mm"},
            {"field": "held_part", "derived": None},
            {"field": "_task_ctx", "requested": "private", "derived": "private"},
            {"field": "held_part", "requested": {"private": "model"}, "derived": None},
        ], "private_model": "private"}}}
    original = deepcopy(finding)
    public = _outline_feedback_for_prompt(finding, {"held_part"})
    assert public["evidence"]["owner_effect_result"]["conflicts"] == [
        {"field": "held_part", "requested": None, "derived": "KET4_Square_4mm"}]
    assert "private" not in json.dumps(public)
    assert finding == original


@pytest.mark.parametrize("scope,field,row,expected", [
    ("part", "state", {"current_state": "misplaced", "part_state": "assembled"}, "assembled"),
    ("part", "location", {"current_location": None, "part_location": "assembly_board-v1"}, "assembly_board-v1"),
    ("part", "holder", {"current_holder_resource_jid": JID, "part_holder_resource_jid": None}, None),
    ("resource", "state", {"current_state": "failed", "resource_state": "idle"}, "idle"),
    ("resource", "location", {"current_location": "assembly_board-v1", "resource_location": "Buffer For Machined parts"}, "Buffer For Machined parts"),
])
def test_version2_completion_reads_supported_projected_fields_without_changing_capture(scope, field, row, expected):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import _continuation_condition_satisfied

    prepared, session, _ = _case()
    entity = "gear_small" if scope == "part" else JID
    session["symbolic_parts" if scope == "part" else "symbolic_resources"][entity] = deepcopy(row)
    condition = {"entity_kind": scope, "entity": entity, "field": field, "expected": expected}
    original = deepcopy((prepared, session))
    assert _continuation_condition_satisfied(condition, session_state=session, prepared_recovery_request=prepared)
    session["validation_contract_version"] = 1
    assert not _continuation_condition_satisfied(condition, session_state=session, prepared_recovery_request=prepared)
    session["validation_contract_version"] = 2
    assert (prepared, session) == original


@pytest.mark.parametrize("row,expected,accepted", [
    ({"part_location": None, "current_location": "assembly_board-v1"}, None, True),
    ({"part_location": None, "current_location": "assembly_board-v1"}, "assembly_board-v1", False),
    ({"part_location": None}, "", False),
    ({"part_location": ""}, None, False),
    ({"part_location": True}, 1, False),
    ({"part_location": 1}, 1.0, False),
    ({"part_location": 1}, "1", False),
    ({"part_location": " assembly_board-v1"}, "assembly_board-v1", False),
    ({"part_location": None}, {"not": "assembly_board-v1"}, True),
    ({"part_state": "assembled"}, None, False),
    ({"part_state": "assembled"}, {"not": "assembly_board-v1"}, False),
    ({"current_location": "assembly_board-v1"}, "assembly_board-v1", True),
])
def test_version2_completion_keeps_missing_null_and_exact_values_distinct(row, expected, accepted):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import _continuation_condition_satisfied

    prepared, session, _ = _case()
    session["symbolic_parts"]["gear_small"] = deepcopy(row)
    condition = {"entity_kind": "part", "entity": "gear_small", "field": "location", "expected": expected}
    assert _continuation_condition_satisfied(condition, session_state=session, prepared_recovery_request=prepared) is accepted


@pytest.mark.parametrize("field,expected,accepted", [
    ("current_state", "misplaced", True),
    ("current_state", "assembled", False),
    ("part_state", "assembled", True),
    ("current_location", None, True),
    ("part_location", "assembly_board-v1", True),
    ("part_holder_resource_jid", None, False),
])
def test_version2_completion_preserves_explicit_field_names(field, expected, accepted):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import _continuation_condition_satisfied

    prepared, session, _ = _case()
    session["symbolic_parts"]["gear_small"] = {
        "current_state": "misplaced", "part_state": "assembled", "current_location": None,
        "part_location": "assembly_board-v1", "current_holder_resource_jid": None,
    }
    condition = {"entity_kind": "part", "entity": "gear_small", "field": field, "expected": expected}
    assert _continuation_condition_satisfied(condition, session_state=session, prepared_recovery_request=prepared) is accepted
