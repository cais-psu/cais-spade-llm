from __future__ import annotations

"""Saved Gazebo inputs, equipment-free generation, and explicit execution handoff."""

import asyncio
import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.recovery_framework import offline_recovery
from cais_spade_llm.recovery_framework.diagnostics import write_record
from cais_spade_llm.recovery_framework.recovery_observations import (
    list_observations,
    observation_fingerprint,
    read_observations,
)


@pytest.fixture
def captured(tmp_path, monkeypatch):
    jid = "recovery-resource-4@localhost"
    catalog = [{"name": "move_to_named_pose", "params": {"pose_name": {"type": "string", "enum": ["home"]}},
                "required_params": ["pose_name"], "effects": {"current_state": {"set": "idle"}}}]
    model = {"state_variables": {"resource_state": {"scope": "resource", "domain": ["failed", "idle"]},
                                 "held_part": {"scope": "resource", "domain": [None]}},
             "events": [{"event_name": "move_home", "guards": {}, "updates": {"resource_state": {"set": "idle"}},
                         "primitive_support": [{"primitive": "move_to_named_pose", "params": {"pose_name": "home"}}]}],
             "primitive_catalog": catalog}
    model["descriptor_fingerprint"] = recovery_validation_fingerprint(model)
    snapshot = {"resource_type": "robot", "resource_jid": jid, "current_state": "failed", "held_part": None,
                "current_pose": {"x": 0.1, "y": 0.2, "z": 1.2}, "current_pose_captured_at": 123.0,
                "named_poses": ["home"], "controller_ready": True}
    entry = {"resource_jid": jid, "resource_type": "robot", "recovery_snapshot": snapshot,
             "recovery_des_model": model, "recovery_des_model_fingerprint": model["descriptor_fingerprint"],
             "execution_primitive_catalog": catalog, "primitive_catalog": catalog,
             "static_capabilities": {"resource_type": "robot"}}
    return {"run_id": "captured-run", "scenario": "Part slippage", "captured_at": "2026-10-08T00:00:00+00:00",
            "offline_ready": True, "missing": [], "jids": {"ur5e-4": jid},
            "part_tracker": {"gear_small": {"state": "misplaced", "location": None,
                                            "observed_pose": {"x": 0.0, "y": 0.3, "z": 1.025}}},
            "resource_states": {"ur5e-4": {"resource_state": "failed", "held_part": None}},
            "failure": {"injection_status": "completed", "checkpoint": "during_place_lowering"},
            "prepared_recovery_request": {"ra_jid": jid, "recovery_resources": {jid: entry},
                                          "recovery_session": {}, "recovery_debug": {}, "llm_input": {}}}


def test_readiness_reports_conflicting_saved_fields_before_generation(captured):
    from cais_spade_llm.recovery_framework.recovery_observations import offline_preparation_missing, observation_summary

    jid = captured['jids']['ur5e-4']
    captured['resource_evidence'] = {jid: {'snapshot': {'held_part': 'another-part'}}}
    before = deepcopy(captured)
    reasons = offline_preparation_missing(captured)
    assert any('Conflicting recorded resource state' in row and '.held_part' in row for row in reasons)
    assert observation_summary(captured)['offline_ready'] is False
    assert captured == before


@pytest.mark.parametrize(
    ("saved", "environment", "overrides", "expected", "sources"),
    [
        ({}, {}, {}, ("gpt-5.4", "medium"), ("default", "default")),
        ({}, {"OPENAI_MODEL": "environment-model", "OPENAI_REASONING_EFFORT": "low"}, {},
         ("environment-model", "low"), ("environment", "environment")),
        ({"llm_model": "saved-model", "llm_reasoning_effort": "high"},
         {"OPENAI_MODEL": "environment-model", "OPENAI_REASONING_EFFORT": "low"}, {},
         ("saved-model", "high"), ("saved", "saved")),
        ({"llm_model": "gpt-5.4", "llm_reasoning_effort": "high"},
         {"OPENAI_MODEL": "environment-model", "OPENAI_REASONING_EFFORT": "medium"},
         {"model": "gpt-5.6", "reasoning_effort": "low"}, ("gpt-5.6", "low"), ("override", "override")),
        ({"llm_model": "gpt-5.4", "llm_reasoning_effort": "low"}, {}, {"model": "gpt-5.6"},
         ("gpt-5.6", "low"), ("override", "saved")),
        ({"llm_model": "gpt-5.4", "llm_reasoning_effort": "high"}, {}, {"reasoning_effort": "low"},
         ("gpt-5.4", "low"), ("saved", "override")),
        ({"llm_model": "gpt-5.4", "llm_reasoning_effort": "minimal"}, {}, {},
         ("gpt-5.4", "none"), ("saved", "saved")),
        ({}, {"OPENAI_REASONING_EFFORT": "unsupported"}, {},
         ("gpt-5.4", "medium"), ("default", "default")),
        ({"llm_model": "gpt-5.4", "llm_reasoning_effort": "high"}, {}, {"model": "gpt-4o"},
         ("gpt-4o", None), ("override", "not_applicable")),
        ({"llm_model": "gpt-4o", "llm_reasoning_effort": "low"}, {}, {},
         ("gpt-4o", None), ("saved", "not_applicable")),
        ({}, {"OPENAI_MODEL": "gpt-4o", "OPENAI_REASONING_EFFORT": "medium"}, {},
         ("gpt-4o", None), ("environment", "not_applicable")),
        ({"llm_model": "gpt-5.4", "llm_reasoning_effort": "high"}, {},
         {"model": "gpt-5.6-sol", "reasoning_effort": "low"}, ("gpt-5.6-sol", "low"), ("override", "override")),
    ],
)
def test_offline_attempt_model_settings_precedence(captured, monkeypatch, saved, environment, overrides, expected, sources):
    for key in ("CAIS_SPADE_LLM_MODEL", "OPENAI_MODEL", "CASE3_RECOVERY_MODEL",
                "CAIS_SPADE_REASONING_EFFORT", "OPENAI_REASONING_EFFORT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    captured.update(saved)
    original = deepcopy(captured)
    product = offline_recovery.SavedRecoveryProduct(captured, [], **overrides)
    assert (product.model, product.reasoning_effort) == expected
    settings = product.llm_settings
    assert (settings["model"], settings["reasoning_effort"]) == expected
    assert (settings["model_source"], settings["reasoning_effort_source"]) == sources
    assert settings["model_override"] == overrides.get("model")
    assert settings["reasoning_effort_override"] == overrides.get("reasoning_effort")
    assert settings["saved_model"] == saved.get("llm_model")
    assert settings["saved_reasoning_effort"] == saved.get("llm_reasoning_effort")
    assert captured == original


@pytest.mark.parametrize("overrides", [
    {"model": ""}, {"model": " "}, {"model": " gpt-5.6"}, {"model": 5.6},
    {"reasoning_effort": ""}, {"reasoning_effort": "automatic"}, {"reasoning_effort": "LOW"},
    {"reasoning_effort": 0}, {"model": "gpt-5.4", "reasoning_effort": "minimal"},
    {"model": "gpt-4o", "reasoning_effort": "low"}, {"model": "gpt-4o", "reasoning_effort": "none"},
])
def test_offline_attempt_rejects_invalid_explicit_model_settings(captured, overrides):
    original = deepcopy(captured)
    with pytest.raises(ValueError, match="model|reasoning_effort"):
        offline_recovery.SavedRecoveryProduct(captured, [], **overrides)
    assert captured == original


@pytest.mark.parametrize("model,effort", [("gpt-5.6", "low"), ("gpt-5.6-sol", "low"), ("gpt-4o", None)])
def test_offline_model_override_reaches_captured_provider_request(captured, monkeypatch, tmp_path, model, effort):
    from cais_spade_llm.agents.shared_information import llm_agent
    from cais_spade_llm.agents.shared_information.llm_request_records import capture_requests

    captured.update(llm_model="gpt-5.4", llm_reasoning_effort="high")
    original = deepcopy(captured)
    reply = SimpleNamespace(model=model + "-returned", choices=[SimpleNamespace(
        message=SimpleNamespace(content='{"decision":"need_context"}', tool_calls=None))])
    provider = Mock(return_value=reply)
    monkeypatch.setattr(llm_agent, "_client", SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=provider))))
    product = offline_recovery.SavedRecoveryProduct(captured, [], model=model, reasoning_effort=effort)
    with capture_requests(directory=tmp_path / "capture", run_id=captured["run_id"], stage="outline", turn=1):
        response = asyncio.run(product.ask_llm_structured(
            "Read recorded facts", response_format={"name": "test", "strict": False,
                                                    "schema": {"type": "object"}}))
    assert response == {"decision": "need_context"}
    provider.assert_called_once()
    request = json.loads(next((tmp_path / "capture").glob("requests/*/*_request.json")).read_text())
    assert request["payload"] == provider.call_args.kwargs
    assert request["payload"]["model"] == model
    if model == "gpt-4o":
        assert "reasoning_effort" not in request["payload"]
        assert "reasoning_effort" not in product._last_structured_request
        assert product.llm_settings["reasoning_effort_source"] == "not_applicable"
    else:
        assert request["payload"]["reasoning_effort"] == effort
    raw_response = json.loads(next((tmp_path / "capture").glob("requests/*/*_response.json")).read_text())
    assert raw_response["raw_response"]["model"] == model + "-returned"
    assert captured == original


def test_standalone_saved_resource_keeps_defensive_capture_copy(captured):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    original = deepcopy(captured)
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    captured["part_tracker"]["gear_small"]["observed_pose"]["x"] = 99.
    assert resource._observations == original
    assert resource._observations is not captured


@pytest.mark.parametrize("defect", [None, "revision", "run", "pose", "explicit_null"])
def test_saved_resource_recovers_omitted_frame_and_tcp_from_same_capture(captured, defect):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    snapshot = deepcopy(entry["recovery_snapshot"])
    captured["context_revision"] = 4
    entry["resource_evidence"] = {"snapshot": deepcopy(snapshot), "resource_type": "robot"}
    physical = {"source": "GETRECOVERYSTATE", "frame": "world", "observation_state": deepcopy(snapshot),
        "controlled_link": "tool0", "tcp_link": "tcp", "tcp_offset_z": -.2}
    captured["physical_evidence"] = {"provenance": {"run_id": captured["run_id"], "context_revision": 4},
        "snapshot": {"frame": "world", "resources": {"ur5e-4": deepcopy(snapshot)}},
        "resources": {"ur5e-4": physical}, "resource_models": {"ur5e-4": {"configuration": {
            "description_sha256": "recorded-model", "joints": [{"type": "fixed", "parent": "tool0",
                "child": "tcp", "xyz": [0., 0., .2], "rpy": [0., 0., 0.]}]}}}}
    if defect == "revision":
        captured["physical_evidence"]["provenance"]["context_revision"] += 1
    elif defect == "run":
        captured["physical_evidence"]["provenance"]["run_id"] = "another-run"
    elif defect == "pose":
        physical["observation_state"]["current_pose"]["x"] += 1
    elif defect == "explicit_null":
        entry["resource_evidence"]["motion_config"] = {"tcp_transform": None}
    original = deepcopy(captured)
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    result = resource.get_recovery_resource_evidence()
    if defect in {None, "explicit_null"}:
        assert result["frame"] == "world"
        assert result["motion_config"]["tcp_link"] == "tcp"
        if defect is None:
            assert result["motion_config"]["tcp_transform"]["translation"] == [0., 0., .2]
            assert result["motion_config"]["tcp_transform"]["source"]["description_sha256"] == "recorded-model"
        else:
            assert result["motion_config"]["tcp_transform"] is None
    else:
        assert "tcp_transform" not in result.get("motion_config", {})
    if defect == "pose":
        assert any(row["field"] == "current_pose" for row in result["conflicts"])
    assert captured == original


def test_public_preparation_uses_recorded_physical_part_pose(captured):
    from cais_spade_llm.recovery_framework.recovery_observations import restore_recorded_public_facts

    part = 'KET4_Square_4mm'
    pose = [-4.13, 1.82, 1.06, 0., 0., 0., 1.]
    captured['physical_evidence'] = {'snapshot': {'frame': 'world'}, 'parts': {
        part: {'name': part, 'frame': 'world', 'pose': pose, 'source': 'GETRECOVERYSTATE',
               'simulation_time': 83.862, 'instance_id': 'capture'}}}
    prepared = {'part_tracker': {part: {'observed_pose': None, 'location': 'M1'}},
                'grounding_context': {'parts': {part: {'observed_pose': None}}},
                'llm_input': {'part_facts': [{'part_name': part, 'observed_pose': None}]}}
    original = deepcopy(captured)
    sources = restore_recorded_public_facts(prepared, captured)
    expected = dict(zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), pose))
    assert prepared['part_tracker'][part]['observed_pose'] == expected
    assert prepared['grounding_context']['parts'][part]['observed_pose'] == expected
    assert prepared['llm_input']['part_facts'][0]['observed_pose'] == expected
    assert prepared['part_tracker'][part]['location'] == 'M1'
    assert sources['/parts/' + part + '/observed_pose']['simulation_time'] == 83.862
    assert captured == original
    captured['physical_evidence']['snapshot']['parts'] = {part: {'model_name': 'recorded_model'}}
    captured['physical_evidence']['parts'][part]['name'] = 'recorded_model'
    assert restore_recorded_public_facts(prepared, captured)['/parts/' + part + '/observed_pose']
    prepared['part_tracker'][part]['observed_pose']['x'] = 100.
    with pytest.raises(ValueError, match='conflicting recorded observed_pose'):
        restore_recorded_public_facts(prepared, captured)


@pytest.mark.parametrize('offset_m,conflict', [(1.9e-08, False), (0.002, True)])
def test_recorded_part_poses_agree_within_configured_bounds(offset_m, conflict):
    from cais_spade_llm.recovery_framework.recovery_observations import restore_recorded_public_facts

    part = 'gear_small'
    pose = [0., 0.3, 1.025, 0., 0., 0.00065, 0.9999997887]
    tracked = dict(zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), pose))
    tracked.update(y=0.3 + offset_m, qx=-2.8e-05)
    observations = {'run_id': 'capture', 'physical_evidence': {'snapshot': {'frame': 'world'}, 'parts': {
        part: {'name': part, 'frame': 'world', 'pose': pose, 'source': 'GETRECOVERYSTATE'}}}}
    prepared = {'part_tracker': {part: {'observed_pose': tracked}}}
    if conflict:
        with pytest.raises(ValueError, match='conflicting recorded observed_pose'):
            restore_recorded_public_facts(prepared, observations)
        return
    restore_recorded_public_facts(prepared, observations)
    assert prepared['part_tracker'][part]['observed_pose']['y'] == 0.3


@pytest.mark.parametrize('field,value', [('frame', 'other'), ('name', 'other'), ('pose', [0., 0., 1.])])
def test_public_preparation_rejects_unbound_physical_part_pose(field, value):
    from cais_spade_llm.recovery_framework.recovery_observations import restore_recorded_public_facts

    record = {'name': 'part', 'frame': 'world', 'pose': [0., 0., 1., 0., 0., 0., 1.]}
    record[field] = value
    source = {'physical_evidence': {'snapshot': {'frame': 'world'}, 'parts': {'part': record}}}
    with pytest.raises(ValueError, match='invalid recorded part observation'):
        restore_recorded_public_facts({}, source)


def test_saved_observations_use_recorded_values_without_controller(captured):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_primitives import (
        extract_step_output,
    )

    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    pose_result = resource.get_current_pose()
    assert pose_result["pose"] == entry["recovery_snapshot"]["current_pose"]
    assert resource.get_recovery_snapshot()["current_pose_captured_at"] == 123.0
    output, error = extract_step_output(primitive="get_current_pose", params={}, step_result=pose_result, resource_type="robot")
    assert error is None and output["pose"] == pose_result["pose"]
    detected = resource.detect_parts("gear_small")
    output, error = extract_step_output(primitive="detect_parts", params={"part_name": "gear_small"},
                                       step_result={"success": True, "data": detected}, resource_type="robot")
    assert error is None and output["pose"] == captured["part_tracker"]["gear_small"]["observed_pose"]
    with pytest.raises(ValueError, match="NEEDS_CONTEXT"):
        resource.detect_parts("KET4_Square_4mm")
    assert not hasattr(resource, "move_to_named_pose")
    assert not hasattr(resource, "_controller")


@pytest.mark.parametrize("x,expected", [(0.1, "FEASIBLE"), (5., "INFEASIBLE"), (None, "NEEDS_CONTEXT")])
def test_saved_cartesian_capability_requires_explicit_parameters_and_recorded_bounds(captured, x, expected):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    entry["recovery_des_model"]["primitive_catalog"].append({"name": "move_cartesian", "frame": "world"})
    entry["static_capabilities"]["gripper_reach"] = {
        "frame": "world", "origin_pose": {"x": 0., "y": 0., "z": 1.},
        "max_xy_radius_m": .7, "z_min_m": .8, "z_max_m": 1.6, "tolerance_m": .01}
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    row = resource.check_recovery_primitive_feasibility(primitive="move_cartesian", params={"x": x, "y": .2, "z": 1.2},
        task={}, recovery_snapshot=resource.get_recovery_snapshot(), part_context={}, grounded_action={})
    assert row["feasibility_status"] == expected
    assert row["evidence"]["command_sent"] is False
    assert row["evidence"]["params"]["x"] is x


def test_saved_primitive_evaluator_does_not_invent_controller_or_projected_observations(captured):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    entry["recovery_des_model"]["primitive_catalog"].extend([
        {"name": "compute_pick_targets"}, {"name": "detect_parts"}, {"name": "move_relative"}])
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    common = {"task": {}, "recovery_snapshot": resource.get_recovery_snapshot(), "grounded_action": {}}
    computed = resource.check_recovery_primitive_feasibility(primitive="compute_pick_targets",
        params={"part_name": "gear_small"}, part_context={}, **common)
    observed = resource.check_recovery_primitive_feasibility(primitive="detect_parts",
        params={"part_name": "gear_small"}, part_context={"observed_pose_current": False}, **common)
    relative = resource.check_recovery_primitive_feasibility(primitive="move_relative", params={"dz": .1}, part_context={}, **common)
    assert all(row["feasibility_status"] == "NEEDS_CONTEXT" for row in (computed, observed, relative))


def test_saved_place_capability_uses_recorded_physical_scene(captured, monkeypatch):
    from cais_spade_llm.resources.robot import recovery_capability

    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    entry["recovery_des_model"]["primitive_catalog"].append({"name": "compute_place_targets"})
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    scene = {"snapshot": {"resources": {"ur5e-4": {"joint_positions": [1., 2.]}}},
             "source": {"capture": "immutable-observations", "revision": 7}}
    original = deepcopy((scene, captured))
    calls = []

    def calculate(**kwargs):
        calls.append(deepcopy(kwargs))
        return {"status": "calculated", "context_requests": [], "source": scene["source"],
                "insert_pose": {"x": .1, "y": .2, "z": 1.2}}

    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", calculate)
    result = resource.check_recovery_primitive_feasibility(primitive="compute_place_targets",
        params={"part_name": "gear_small"}, task={"part_name": "gear_small"},
        recovery_snapshot=resource.get_recovery_snapshot(), part_context={}, grounded_action={},
        resource_evidence=resource.get_recovery_resource_evidence(), physical_evidence=scene)
    assert result["feasibility_status"] == "FEASIBLE"
    assert len(calls) == 1 and calls[0]["physical_evidence"] == scene
    assert result["evidence"]["calculation"]["source"] == scene["source"]
    assert result["evidence"]["command_sent"] is False
    assert (scene, captured) == original


@pytest.mark.parametrize("primitive", ["pause_job", "resume_job", "cancel_job"])
@pytest.mark.parametrize("material,expected", [("loaded", "FEASIBLE"), ("empty", "INFEASIBLE"), (None, "NEEDS_CONTEXT")])
def test_saved_printer_uses_owner_capability_evaluator_without_live_agent(captured, monkeypatch, primitive, material, expected):
    from cais_spade_llm.agents.resource_agent.printing_agent import PrintingAgent

    monkeypatch.setattr(PrintingAgent, "__init__", lambda *args, **kwargs: pytest.fail("live printer construction"))
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    entry["resource_type"] = "printer"
    entry["recovery_des_model"]["primitive_catalog"] = [{"name": primitive}]
    snapshot = {"current_state": "printing", "material_state": material, "bed_state": "ready"}
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    row = resource.check_recovery_primitive_feasibility(primitive=primitive, params={}, task={},
        recovery_snapshot=snapshot, part_context={}, grounded_action={})
    assert row["feasibility_status"] == expected
    assert row["evidence"]["evidence_source"] == "saved_observations"
    assert row["evidence"]["command_sent"] is False


def test_saved_capability_rejects_conflicting_named_pose_values_and_missing_frame(captured):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    entry["static_capabilities"]["named_poses"] = {"home": "different"}
    entry["recovery_snapshot"]["named_poses"] = {"home": "home"}
    entry["recovery_des_model"]["primitive_catalog"].append({"name": "move_cartesian"})
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    common = {"task": {}, "recovery_snapshot": resource.get_recovery_snapshot(), "part_context": {}, "grounded_action": {}}
    named = resource.check_recovery_primitive_feasibility(primitive="move_to_named_pose", params={"pose_name": "home"}, **common)
    cartesian = resource.check_recovery_primitive_feasibility(primitive="move_cartesian", params={"x": 0., "y": 0., "z": 1.}, **common)
    assert named["feasibility_status"] == cartesian["feasibility_status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("held,attachment,status", [
    ("gear_small", True, "FEASIBLE"), ("other_part", True, "INFEASIBLE"),
    ("gear_small", False, "NEEDS_CONTEXT"), (None, True, "INFEASIBLE")])
def test_saved_release_capability_uses_recorded_custody(captured, held, attachment, status):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    entry["recovery_des_model"]["primitive_catalog"].append({"name": "release_part"})
    snapshot = {**entry["recovery_snapshot"], "held_part": held}
    evidence = {"inputs": {"geometry": {"parts": {"model_map": {"gear_small": "gear_small"}}}},
                "attachment": {"model_name": "gear_small", "physical_attachment_completed": attachment}}
    owner = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    row = owner.check_recovery_primitive_feasibility(primitive="release_part", params={"part_name": "gear_small"},
        task={"part_name": "gear_small"}, recovery_snapshot=snapshot, part_context={}, grounded_action={}, resource_evidence=evidence)
    assert row["feasibility_status"] == status
    assert row["evidence"]["command_sent"] is False
    if status == "FEASIBLE":
        assert "motion_landed_on_target" in row["evidence"]["composition_obligations"]


def test_saved_pose_restoration_preserves_null_and_detects_conflict(captured):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    captured["robot_poses"] = {"ur5e-4": {**entry["recovery_snapshot"]["current_pose"], "qx": 0., "qy": 0., "qz": 0., "qw": 1.}}
    entry["recovery_snapshot"]["current_pose"]["qx"] = None
    owner = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    evidence = owner.get_recovery_resource_evidence()
    assert evidence["snapshot"]["current_pose"]["qx"] is None
    assert evidence["snapshot"]["current_pose"]["qw"] == 1.
    assert "qw" not in owner.get_recovery_snapshot()["current_pose"]
    captured["robot_poses"]["ur5e-4"]["x"] += 1.
    owner = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    assert owner.get_recovery_resource_evidence()["conflicts"]


@pytest.mark.parametrize("pose_format", ["dict", "list"])
def test_explicit_owner_pose_is_not_replaced_or_conflicted_by_later_legacy_sample(captured, pose_format):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    pose = {**entry["recovery_snapshot"]["current_pose"], "qx": 0., "qy": 0., "qz": 0., "qw": 1.}
    entry["recovery_snapshot"]["current_pose"] = deepcopy(pose)
    recorded_pose = list(pose.values()) if pose_format == "list" else deepcopy(pose)
    entry["resource_evidence"] = {"snapshot": {"current_pose": recorded_pose}, "conflicts": [],
        "source": {"run_id": captured["run_id"], "kind": "recorded_owner_state", "captured_at": "owner_sample"}}
    entry["snapshot_sources"] = {"current_pose": {
        "kind": "observed", "ref": f"resource_evidence/{jid}/snapshot/current_pose"}}
    captured["robot_poses"] = {"ur5e-4": {**pose, "x": pose["x"] + .000003}}
    original = deepcopy(captured)
    owner = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    evidence = owner.get_recovery_resource_evidence()
    assert evidence["snapshot"]["current_pose"] == recorded_pose
    assert evidence["source"] == entry["resource_evidence"]["source"]
    assert evidence["conflicts"] == []
    assert captured == original


def test_saved_placement_calculation_uses_explicit_transform_and_preserves_source(captured):
    from cais_spade_llm.resources.robot.recovery_capability import calculate_recovery_place_target

    task = {"resource_jid": "resource@localhost", "part_name": "part", "expected_end_state": {"resource_location": "station"}}
    evidence = {"inputs": {"scene": {"station": {"workholding_pose": [.1, .2, 1.]}},
                            "geometry": {"parts": {"heights_m": {"part": .05}}}},
                "state": {"_task_ctx": {"part_name": "part", "pick_tcp_z": 1.07, "tz": 1.02, "tcp_offset_z": -.218}}}
    snapshot = {"current_pose": {"x": .1, "y": .2, "z": 1.5, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}}
    before = deepcopy(evidence)
    row = calculate_recovery_place_target(task=task, recovery_snapshot=snapshot, resource_evidence=evidence)
    assert row["status"] == "calculated"
    assert row["insert_pose"]["z"] == pytest.approx(1.268)
    assert row["target_origin_pose"] == {"x": .1, "y": .2, "z": 1.}
    assert evidence == before
    evidence["state"]["_task_ctx"]["tcp_offset_z"] = None
    missing = calculate_recovery_place_target(task=task, recovery_snapshot=snapshot, resource_evidence=evidence)
    assert missing["status"] == "NEEDS_CONTEXT"
    assert any(ref.endswith("/state/_task_ctx/tcp_offset_z") for ref in missing["context_requests"])
    assert "insert_pose" not in missing


def test_saved_placement_uses_observed_attachment_transform_without_tcp_default():
    from cais_spade_llm.resources.robot.recovery_capability import calculate_recovery_place_target

    task = {"resource_jid": "robot", "part_name": "part", "expected_end_state": {"resource_location": "station"}}
    snapshot = {"current_pose": {"x": .1, "y": .2, "z": 1.5, "qx": 0., "qy": 0., "qz": 0., "qw": 1.}}
    evidence = {"inputs": {"scene": {"station": {"workholding_pose": [.1, .2, 1.]}},
                            "geometry": {"parts": {"heights_m": {"part": .05}}}},
                "attachment": {"observed_attachment_complete": True, "part_name": "part",
                               "grasp_transform": [0., 0., -.2, 0., 0., 0., 1.]}}
    row = calculate_recovery_place_target(task=task, recovery_snapshot=snapshot, resource_evidence=evidence)
    assert row["status"] == "calculated", row
    assert row["insert_pose"]["z"] == pytest.approx(1.2)
    assert row["grasp_transform"] == evidence["attachment"]["grasp_transform"]
    assert row["command_sent"] is False


def test_saved_pick_calculation_requires_recorded_policy_and_matching_tcp_pose():
    from cais_spade_llm.resources.robot.recovery_capability import calculate_recovery_pick_target

    task = {"resource_jid": "robot", "part_name": "part"}
    snapshot = {"current_pose": {"x": .1, "y": .2, "z": 1.5, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}}
    evidence = {"execution_mode": "simulation", "frame": "world", "snapshot": deepcopy(snapshot),
        "observed_parts": {"part": {"current_pose": [.2, .1, 1., 0., 0., 0., 1.]}},
        "motion_config": {"pick_tcp_z_bias_min_m": .003, "pick_tcp_z_bias_max_m": .02,
            "approach_height_m": .2, "min_pick_tcp_z_m": 1.07, "tcp_offset_z": -.218,
            "tcp_transform_source": "recorded test observation", "controlled_link": "tool0", "tcp_link": "tcp"},
        "pick_geometry": {"part": {"part_height_m": .05, "board_center": {"z": .975}}},
        "pick_policy": {"part": {"surface_clearance_override_m": 0., "pick_z_adjustment_m": 0.,
                                  "use_global_min_pick_tcp_z": True}}}
    before = deepcopy(evidence)
    row = calculate_recovery_pick_target(task=task, recovery_snapshot=snapshot, resource_evidence=evidence)
    assert row["status"] == "calculated", row
    assert row["target_pose"]["z"] == pytest.approx(1.288)
    assert row["approach_pose"]["z"] == 1.5
    assert row["command_sent"] is False
    assert evidence == before
    public_snapshot = {"current_pose": {axis: snapshot["current_pose"][axis] for axis in ("x", "y", "z")}}
    restored = calculate_recovery_pick_target(task=task, recovery_snapshot=public_snapshot, resource_evidence=evidence)
    assert restored["status"] == "calculated"
    public_snapshot["current_pose"]["x"] += .01
    mismatched = calculate_recovery_pick_target(task=task, recovery_snapshot=public_snapshot, resource_evidence=evidence)
    assert mismatched["status"] == "NEEDS_CONTEXT"
    projected = deepcopy(evidence)
    projected["recorded_snapshot"] = deepcopy(snapshot)
    rotated = {"current_pose": {**snapshot["current_pose"], "qy": 0., "qw": 1.}}
    projected["snapshot"] = deepcopy(rotated)
    changed_orientation = calculate_recovery_pick_target(task=task, recovery_snapshot=rotated, resource_evidence=projected)
    assert changed_orientation["status"] == "NEEDS_CONTEXT"
    assert any(ref.endswith("tcp_offset_z") for ref in changed_orientation["context_requests"])
    projected["motion_config"]["tcp_transform"] = {
        "translation": [0., 0., .218], "rotation": [0., 0., 0., 1.], "fixed": True,
        "parent_frame": "tool0", "child_frame": "tcp", "source": {"model_fingerprint": "fixture"}}
    transformed = calculate_recovery_pick_target(task=task, recovery_snapshot=rotated, resource_evidence=projected)
    assert transformed["status"] == "calculated"
    assert transformed["tcp_offset_z"] == pytest.approx(.218)
    assert transformed["target_pose"]["z"] == pytest.approx(.852)
    projected["motion_config"]["tcp_transform"]["fixed"] = False
    unfixed = calculate_recovery_pick_target(task=task, recovery_snapshot=rotated, resource_evidence=projected)
    assert unfixed["status"] == "NEEDS_CONTEXT"
    assert any(ref.endswith("tcp_transform") for ref in unfixed["context_requests"])
    projected["motion_config"]["tcp_transform"]["fixed"] = True
    projected["motion_config"]["tcp_transform"]["parent_frame"] = "other_link"
    conflicting = calculate_recovery_pick_target(task=task, recovery_snapshot=rotated, resource_evidence=projected)
    assert conflicting["status"] == "NEEDS_CONTEXT"
    evidence["snapshot"]["current_pose"]["qy"] = 0.
    missing = calculate_recovery_pick_target(task=task, recovery_snapshot=snapshot, resource_evidence=evidence)
    assert missing["status"] == "NEEDS_CONTEXT"
    assert "target_pose" not in missing
    assert any(ref.endswith("tcp_offset_z") for ref in missing["context_requests"])


def test_saved_projected_custody_remains_distinct_from_observations():
    from cais_spade_llm.resources.recovery_parameter_evidence import projected_resource_evidence
    from cais_spade_llm.resources.robot.recovery_capability import calculate_recovery_place_target

    observed = {"resource_type": "robot", "state": {"_task_ctx": {"part_name": "old-part"}},
        "attachment": {"part_name": "old-part", "observed_attachment_complete": True},
        "inputs": {"scene": {"station": {"workholding_pose": [.1, .2, 1.]}},
                   "geometry": {"parts": {"heights_m": {"part": .05}}}}}
    projected = {"provenance": {"kind": "projected", "contract_fingerprint": "fixture-contract"},
        "jids": {"robot": "robot@localhost"}, "snapshot": {
            "resources": {"robot": {"held_part": "part", "current_pose": [.1, .2, 1.5, 0., 0., 0., 1.],
                                     "grasp_transform": [0., 0., -.2, 0., 0., 0., 1.]}},
            "parts": {"part": {"current_pose": [.1, .2, 1.3, 0., 0., 0., 1.], "model_name": "part"}}}}
    snapshot = deepcopy(projected["snapshot"]["resources"]["robot"])
    before = deepcopy(observed)
    facts = projected_resource_evidence(resource_jid="robot@localhost", resource_evidence=observed,
        physical_evidence=projected, recovery_snapshot=snapshot)
    assert "observed_attachment_complete" not in facts["attachment"]
    assert facts["attachment"]["projected_attachment_complete"] is True
    assert facts["recorded_attachment"]["part_name"] == "old-part"
    assert "_task_ctx" not in facts["state"]
    result = calculate_recovery_place_target(task={"resource_jid": "robot@localhost", "part_name": "part",
        "expected_end_state": {"resource_location": "station"}}, recovery_snapshot=snapshot, resource_evidence=facts)
    assert result["status"] == "calculated", result
    assert result["insert_pose"]["z"] == pytest.approx(1.2)
    assert result["source"]["attachment_source"]["kind"] == "projected"
    assert observed == before
    unchanged = deepcopy(observed)
    unchanged["snapshot"] = deepcopy(snapshot)
    unchanged["state"]["_task_ctx"] = {"part_name": "part", "travel_z": 1.5}
    peer_projection = projected_resource_evidence(resource_jid="robot@localhost", resource_evidence=unchanged,
        physical_evidence=projected, recovery_snapshot=snapshot)
    assert peer_projection["state"]["_task_ctx"] == unchanged["state"]["_task_ctx"]
    snapshot["held_part"] = "different"
    conflicts = projected_resource_evidence(resource_jid="robot@localhost", resource_evidence=observed,
        physical_evidence=projected, recovery_snapshot=snapshot)
    assert conflicts["conflicts"]


@pytest.mark.parametrize("defect", [None, "moved", "invalidated", "revision", "missing_source",
                                   "missing_run", "missing_revision", "nonfinite"])
def test_detection_retains_only_matching_recorded_part_observations(defect):
    from cais_spade_llm.resources.robot.recovery_capability import check_robot_primitive_capability

    pose = dict(x=.1, y=.2, z=1., qx=0., qy=0., qz=0., qw=1.)
    facts = {"source": {"run_id": "captured", "revision": 4}, "frame": "world",
        "observed_parts": {"part": {"current_pose": deepcopy(pose), "source": {"kind": "projected"}}},
        "recorded_observed_parts": {"part": {"current_pose": deepcopy(pose)}}}
    context = {"observed_pose": deepcopy(pose), "observed_pose_source": {
        "frame": "world", "run_id": "captured", "revision": 4, "ref": "/physical_evidence/parts/part/pose"}}
    if defect == "moved":
        facts["observed_parts"]["part"]["current_pose"]["x"] += 1
    elif defect == "invalidated":
        context["observed_pose_current"] = False
    elif defect == "revision":
        context["observed_pose_source"]["revision"] += 1
    elif defect == "missing_source":
        context.pop("observed_pose_source")
    elif defect in {"missing_run", "missing_revision"}:
        field = "run_id" if defect == "missing_run" else "revision"
        context["observed_pose_source"].pop(field)
        facts["source"].pop(field)
    elif defect == "nonfinite":
        for value in (facts["observed_parts"]["part"]["current_pose"],
                      facts["recorded_observed_parts"]["part"]["current_pose"], context["observed_pose"]):
            value["x"] = float("inf")
    original = deepcopy(facts)
    result = check_robot_primitive_capability(primitive="detect_parts", params={"part_name": "part"},
        task={"part_name": "part"}, recovery_snapshot={}, resource_evidence=facts,
        static_capabilities={}, primitive_catalog=[], resource_jid="robot@localhost", part_context=context)
    assert result["feasibility_status"] == ("FEASIBLE" if defect is None else "NEEDS_CONTEXT")
    assert result["evidence"]["command_sent"] is False
    assert facts == original


@pytest.mark.parametrize("saved_inputs", [{"geometry": {"recorded": [1, 2]}}, None])
def test_saved_resource_does_not_copy_unused_capture_inputs(captured, saved_inputs):
    class UnusedInputs(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("Existing recorded inputs must not copy their unused fallback")

    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    entry["resource_evidence"] = {"inputs": deepcopy(saved_inputs)}
    captured["inputs"] = UnusedInputs({"different": [3, 4]})
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured, _copy_observations=False)
    evidence = resource.get_recovery_resource_evidence()
    assert evidence["inputs"] == saved_inputs
    if saved_inputs is not None:
        evidence["inputs"]["geometry"]["recorded"].append(5)
        assert resource._entry["resource_evidence"]["inputs"] == saved_inputs
    assert captured["inputs"] == {"different": [3, 4]}


def test_saved_resource_detaches_required_capture_inputs(captured):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    captured["inputs"] = {"geometry": {"recorded": [1, 2]}}
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    evidence = resource.get_recovery_resource_evidence()
    evidence["inputs"]["geometry"]["recorded"].append(3)
    assert captured["inputs"] == resource._observations["inputs"] == {"geometry": {"recorded": [1, 2]}}
    assert "inputs" not in resource._entry.get("resource_evidence", {})


@pytest.mark.parametrize("proposed_pose", [None, {"x": 0., "y": 0., "z": 1.1}])
def test_saved_capture_projected_release_does_not_reuse_old_drop_pose(captured, monkeypatch, proposed_pose):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_outline_state import (
        _apply_outline_task_effects,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_validation_service import (
        build_recovery_physical_validation_input,
        compile_grounded_recovery_outline_task,
        projected_outline_validation_context,
    )
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    path = Path(__file__).parents[1] / "cais_spade_llm/monitor/environment_runs/recovery/1c3bec8faca6495cb1b0e22046277d5c/observations.json"
    if not path.is_file():
        pytest.skip("The user-owned October 8 observations are not available")
    source = path.read_bytes()
    saved = json.loads(source)
    live = Mock(side_effect=AssertionError("No controller or perception calls are allowed"))
    monkeypatch.setattr(RobotAgent, "check_recovery_physical_feasibility", live)
    monkeypatch.setattr(RobotAgent, "get_recovery_snapshot", live)
    monkeypatch.setattr("cais_spade_llm.recovery_framework.failure_effects._observe_part", live)
    prepared = deepcopy(saved["prepared_recovery_request"])
    drop_pose = deepcopy(saved["failure"]["observed_drop_pose"])
    part_facts = prepared["llm_input"]["part_facts"]
    gear = next(row for row in part_facts if row["part_name"] == "gear_small")
    gear.update(observed_pose=deepcopy(drop_pose), observed_pose_frame="world", pose=deepcopy(drop_pose))
    resources, parts = projected_outline_validation_context(session_state={}, prepared_recovery_request=prepared)
    jid3, jid4 = saved["jids"]["ur5e-3"], saved["jids"]["ur5e-4"]
    release = {"resource_jid": jid3, "part_name": "gear_small", "expected_end_state": {
        "resource_state": "idle", "resource_location": "assembly_board-v1", "held_part": None,
        "part_state": "ready", "part_location": "assembly_board-v1"}}
    if proposed_pose is not None:
        release["expected_end_state"]["pose"] = deepcopy(proposed_pose)
    _apply_outline_task_effects(release, resources_by_jid=resources, parts_by_name=parts, task_type="part_handling")
    state = {"symbolic_resources": resources, "symbolic_parts": parts, "accepted_outline_prefix": [release],
             "observation_store": {"saved_drop": {"part_name": "gear_small", "pose": deepcopy(drop_pose)}}}
    resources, parts = projected_outline_validation_context(session_state=state, prepared_recovery_request=prepared)
    assert parts["gear_small"]["observed_pose"] is None
    assert parts["gear_small"]["observed_pose_current"] is False
    assert parts["gear_small"]["last_observed_pose"] == drop_pose
    assert parts["gear_small"]["projected_pose"] == proposed_pose
    task = {"outline_id": "candidate", "event_name": "anything", "resource_jid": jid4, "part_name": "gear_small",
            "expected_end_state": {"resource_state": "picked", "resource_location": "assembly_board-v1",
                                   "held_part": "gear_small", "part_state": "in_gripper", "part_location": jid4}}
    compiled = compile_grounded_recovery_outline_task(
        task, resources_by_jid=resources, parts_by_name=parts, validate_binding_tokens=False)
    assert compiled["grounded_action"] is not None, compiled
    physical = build_recovery_physical_validation_input(
        task=task, grounded_action=compiled["grounded_action"], session_state=state, prepared_recovery_request=prepared)
    assert not (physical["grounded_action"]["preconditions"].get("source_ref") or {}).get("pose")
    assert path.read_bytes() == source
    assert saved == json.loads(source)
    live.assert_not_called()


@pytest.mark.parametrize("carried_move", [False, True])
def test_projection_preserves_unmoved_part_measurements(captured, carried_move):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_outline_state import (
        _apply_outline_task_effects,
    )

    jid = captured["prepared_recovery_request"]["ra_jid"]
    measured = deepcopy(captured["part_tracker"]["gear_small"]["observed_pose"])
    resources = {jid: {"resource_location": "home", "held_part": "gear_small"}}
    parts = {"gear_small": {"part_location": jid, "part_holder_resource_jid": jid,
                             "observed_pose": deepcopy(measured)},
             "unrelated": {"observed_pose": deepcopy(measured)}}
    end = {"part_state": "processed"}
    if carried_move:
        end["resource_location"] = "assembly_board-v1"
    task = {"resource_jid": jid, "part_name": "gear_small", "expected_end_state": end}
    _apply_outline_task_effects(task, resources_by_jid=resources, parts_by_name=parts, task_type="part_handling")
    assert parts["unrelated"] == {"observed_pose": measured}
    if carried_move:
        assert parts["gear_small"]["observed_pose_current"] is False
        assert parts["gear_small"]["observed_pose"] is None
        assert parts["gear_small"]["last_observed_pose"] == measured
    else:
        assert parts["gear_small"]["observed_pose"] == measured
        assert "observed_pose_current" not in parts["gear_small"]


@pytest.mark.parametrize("custody_field", ["held_part", "part_holder_resource_jid"])
@pytest.mark.parametrize("runtime_commit", [False, True])
def test_resource_only_carried_move_invalidates_part_measurement(captured, custody_field, runtime_commit):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import (
        _apply_task_effects_to_symbolic_state,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_outline_state import (
        _apply_outline_task_effects,
        _classify_outline_task,
    )

    jid = captured["prepared_recovery_request"]["ra_jid"]
    measured = deepcopy(captured["part_tracker"]["gear_small"]["observed_pose"])
    resources = {jid: {"resource_location": "assembly_board-v1", "current_location": "assembly_board-v1"}}
    parts = {"gear_small": {"part_location": jid, "observed_pose": deepcopy(measured)},
             "unrelated": {"observed_pose": deepcopy(measured)}}
    if custody_field == "held_part":
        resources[jid][custody_field] = "gear_small"
    else:
        parts["gear_small"][custody_field] = jid
    task = {"resource_jid": jid, "expected_end_state": {"resource_location": "home"}}
    task_type = _classify_outline_task(task, resources_by_jid=resources, parts_by_name=parts)
    assert task_type == "resource_only"
    if runtime_commit:
        state = {"symbolic_resources": resources, "symbolic_parts": parts}
        _apply_task_effects_to_symbolic_state(task, state)
        parts = state["symbolic_parts"]
    else:
        _apply_outline_task_effects(task, resources_by_jid=resources, parts_by_name=parts, task_type=task_type)
    assert parts["gear_small"]["observed_pose"] is None
    assert parts["gear_small"]["observed_pose_current"] is False
    assert parts["gear_small"]["last_observed_pose"] == measured
    assert parts["gear_small"]["projected_pose"] is None
    assert parts["unrelated"] == {"observed_pose": measured}


def test_projected_validation_facts_remain_independent_of_saved_inputs(captured):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_validation_service import (
        projected_outline_validation_context,
    )

    prepared = captured["prepared_recovery_request"]
    jid = prepared["ra_jid"]
    entry = prepared["recovery_resources"][jid]
    entry["static_capabilities"]["named_poses"] = {"home": {"joints": [0, 1]}}
    entry["recovery_des_model"]["current_valuation"] = {"calibration": {"offset": [0.1]}}
    prepared["llm_input"] = {
        "observed_runtime_state": {"resources": [{"resource_jid": "recorded-resource", "current_pose": {"x": 1}}]},
        "part_facts": [{"part_name": "recorded-part", "observed_pose": {"x": 2}}],
    }
    session = {
        "symbolic_resources": {jid: {"resource_state": "idle", "current_pose": {"x": 3}}},
        "symbolic_parts": {"gear_small": {"part_state": "misplaced", "part_location": None}},
        "observation_store": {"saved": {"part_name": "gear_small", "pose": {"x": 4}}},
    }
    original_prepared, original_session = deepcopy(prepared), deepcopy(session)
    resources, parts = projected_outline_validation_context(session_state=session, prepared_recovery_request=prepared)
    assert resources[jid]["current_state"] == "idle"
    assert resources[jid]["current_pose"] == {"x": 3}
    assert resources[jid]["named_poses"] == {"home": {"joints": [0, 1]}}
    assert parts["gear_small"]["observed_pose"] == {"x": 4}
    resources[jid]["current_pose"]["x"] = 99
    resources[jid]["named_poses"]["home"]["joints"].append(99)
    resources[jid]["calibration"]["offset"].append(99)
    resources["recorded-resource"]["current_pose"]["x"] = 99
    parts["gear_small"]["observed_pose"]["x"] = 99
    parts["recorded-part"]["observed_pose"]["x"] = 99
    assert prepared == original_prepared
    assert session == original_session


@pytest.mark.parametrize("validate", [False, True])
def test_outline_validation_does_not_copy_unrelated_diagnostic_records(captured, validate):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (
        recovery_validation_service,
    )

    class DiagnosticEvidence:
        def __deepcopy__(self, memo):
            raise AssertionError("Diagnostic evidence must not be copied for projected validation")

    prepared = captured["prepared_recovery_request"]
    prepared["recovery_debug"] = {"saved_records": DiagnosticEvidence()}
    jid = prepared["ra_jid"]
    session = {"symbolic_resources": {jid: {"resource_state": "failed", "held_part": None}}}
    if validate:
        findings, grounded = recovery_validation_service.validate_recovery_outline_task(
            planner=object(), session_state=session, prepared_recovery_request=prepared,
            task={"outline_id": "review", "event_name": "move_home", "resource_jid": jid,
                  "expected_end_state": {"resource_state": "idle"}, "rationale": "Use recorded support"},
        )
        assert findings == []
        assert grounded["resource_jid"] == jid
    else:
        resources, _ = recovery_validation_service.projected_outline_validation_context(
            session_state=session, prepared_recovery_request=prepared)
        assert resources[jid]["resource_state"] == "failed"


def test_observation_reload_keeps_content_identity_and_run_separation(captured, tmp_path):
    root = tmp_path / "saved"
    directory = root / captured["run_id"]
    directory.mkdir(parents=True)
    path = directory / "observations.json"
    write_record(path, captured)
    first = read_observations(path, root=root)
    second = root / "two" / "observations.json"
    second.parent.mkdir()
    write_record(second, first)
    restored = read_observations(second, root=root)
    assert observation_fingerprint(first) == observation_fingerprint(restored)
    assert first["source_path"] != restored["source_path"]
    third = root / "three" / "observations.json"
    third.parent.mkdir()
    write_record(third, restored)
    original_files = {item: item.read_bytes() for item in (path, second, third)}
    records = list_observations(root=root)
    assert len(records) == 1
    assert records[0]["source_path"] == str(path)
    assert set(records[0]["related_observation_paths"]) == {str(path), str(second), str(third)}
    assert all(item.read_bytes() == content for item, content in original_files.items())

    changed = root / "changed" / "observations.json"
    changed.parent.mkdir()
    revised = deepcopy(captured)
    revised["part_tracker"]["gear_small"]["observed_pose"]["x"] = 0.2
    write_record(changed, revised)
    assert len(list_observations(root=root)) == 2
    path.unlink()
    records = list_observations(root=root)
    assert len(records) == 2
    assert next(row for row in records if row["source_path"] != str(changed))["source_path"] in {str(second), str(third)}
    with pytest.raises(ValueError, match="Select a saved"):
        read_observations(tmp_path / "observations.json", root=root)


def test_legacy_observations_remain_visible_and_missing_context_is_explicit(tmp_path):
    from cais_spade_llm.recovery_framework.recovery_observations import offline_preparation_missing

    directory = tmp_path / "saved" / "old"
    directory.mkdir(parents=True)
    path = directory / "environment_run.json"
    write_record(path, {"run_id": "old", "conveyor_fault": {"scenario": "Part slippage", "evidence": {"observed_drop_pose": {"x": 1}}},
                        "final_product_states": {"gear_small": {"state": "misplaced"}}, "final_valuation": {}, "inputs": {}})
    record = read_observations(path, root=directory.parent)
    assert list_observations(root=directory.parent) == []
    assert record["part_tracker"]["gear_small"]["state"] == "misplaced"
    assert record["captured_at"] is None
    assert offline_preparation_missing(record)


@pytest.fixture
def saved_execution(captured, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import failure_effects
    from cais_spade_llm.recovery_framework.part_slippage_example import example_setup
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    captured.update(inputs={}, program_revision="saved-programs", pending_nodes=[])
    controller = SimpleNamespace(execution_mode="simulation")
    owner = SimpleNamespace(jid=jid, agent_name="ur5e-4", execution_mode="simulation", _controller=controller,
                            get_recovery_snapshot=Mock(return_value=deepcopy(entry["recovery_snapshot"])),
                            recovery_execution_primitive_catalog=Mock(return_value=deepcopy(entry["execution_primitive_catalog"])),
                            check_recovery_primitive_feasibility=Mock(return_value={"allowed": True, "mocked": True}),
                            execute_recovery_macro=AsyncMock())
    owner.get_recovery_snapshot.side_effect = lambda: {
        **deepcopy(owner.get_recovery_snapshot.return_value), "current_pose_captured_at": time.time(),
    }
    context = SimpleNamespace(run_id="live-run", revision=0, pending_tasks={}, reservations={}, inputs={},
                              part_tracker=deepcopy(captured["part_tracker"]), permitted_resources=["ur5e-4"],
                              resources={rid: SimpleNamespace(valuation=deepcopy(values))
                                         for rid, values in captured["resource_states"].items()},
                              geometry={"gear_small": {"model_name": "gear_small"}})
    runtime = SimpleNamespace(context=context, agent=SimpleNamespace(), resource_agents=[owner],
                              setup=example_setup(), diagnostic_cca_bypass=True, active_executions=set(),
                              jids=deepcopy(captured["jids"]), program_revision="saved-programs", path=tmp_path,
                              queue_save=Mock())
    recovery = PartSlippageRecovery(runtime)
    recovery._new_artifact_directory()
    task = {"outline_id": "RECOVERY_SEQ1", "resource_jid": jid, "expected_snapshot": {"current_state": "failed"},
            "primitive_steps": [{"primitive": "move_to_named_pose", "params": {"pose_name": "home"}}]}
    proposal = {"offline_recovery": True, "source_observation_fingerprint": observation_fingerprint(captured),
                "macro_tasks": [task]}
    monkeypatch.setattr(failure_effects, "_observe_part", lambda *args: deepcopy(captured["part_tracker"]["gear_small"]["observed_pose"]))
    return recovery, proposal, captured, owner


def test_saved_composition_validation_never_executes_it(saved_execution):
    from cais_spade_llm.recovery_framework.saved_recovery_execution import validate_saved_proposal

    recovery, proposal, observations, owner = saved_execution
    before = deepcopy(proposal)
    asyncio.run(validate_saved_proposal(recovery, proposal, observations))
    assert recovery.snapshot()["status"] == "ready_to_execute"
    assert recovery._proposal["offline_recovery"] is False
    owner.execute_recovery_macro.assert_not_called()
    assert proposal == before
    report = json.loads((recovery.path / "generation_0001/gazebo_validation.json").read_text())
    assert report["status"] == "passed" and report["safety_status"] == "skipped"


def test_gazebo_handoff_retains_original_generation_and_imported_captures(saved_execution, tmp_path):
    from cais_spade_llm.recovery_framework.saved_recovery_execution import validate_saved_proposal

    recovery, proposal, observations, owner = saved_execution
    original = Path(recovery.snapshot()["artifact_directory"])
    write_record(original / "proposal.json", {"earlier_generation": True})
    source = tmp_path / "offline_source"
    directory = source / "generation_0001"
    captures = directory / "recovery_outline"
    captures.mkdir(parents=True)
    raw = '{"messages": [{"role": "user", "content": "recorded prompt"}]}\n'
    (captures / "request.json").write_text(raw)
    write_record(source / "observations.json", observations)
    write_record(directory / "proposal.json", proposal)
    source_state = {"artifact_directory": str(directory), "observation_path": str(source / "observations.json"),
                    "outline": [{"outline_id": "RECOVERY_SEQ1"}], "primitives": proposal["macro_tasks"]}
    asyncio.run(validate_saved_proposal(recovery, proposal, observations, source_state=source_state))
    state = recovery.snapshot()
    imported = Path(state["artifact_directory"])
    assert imported != original
    assert json.loads((original / "proposal.json").read_text()) == {"earlier_generation": True}
    assert (Path(state["outline_artifact_directory"]) / "request.json").read_text() == raw
    assert (imported / "source_proposal.json").read_bytes() == (directory / "proposal.json").read_bytes()
    assert state["outline"] == source_state["outline"] and state["imported_recovery"] is True
    assert recovery._outline_request == {}
    asyncio.run(validate_saved_proposal(recovery, proposal, observations))
    assert len(list(imported.glob("gazebo_validation*.json"))) == 2
    owner.execute_recovery_macro.assert_not_called()


@pytest.mark.parametrize("change", ["custody", "configuration", "hardware", "controller", "catalog", "fingerprint", "stale_pose", "tasks"])
def test_saved_composition_rejects_incompatible_live_state(saved_execution, change):
    from cais_spade_llm.recovery_framework.saved_recovery_execution import validate_saved_proposal

    recovery, proposal, observations, owner = saved_execution
    if change == "custody":
        recovery.runtime.context.part_tracker["gear_small"]["location"] = "ur5e-3"
    elif change == "configuration":
        recovery.runtime.context.inputs = {"scene": "changed"}
    elif change == "hardware":
        owner.execution_mode = owner._controller.execution_mode = "physical"
    elif change == "controller":
        owner.get_recovery_snapshot.return_value["controller_ready"] = False
    elif change == "catalog":
        owner.recovery_execution_primitive_catalog.return_value = []
    elif change == "stale_pose":
        owner.get_recovery_snapshot.side_effect = None
    elif change == "tasks":
        observations["pending_nodes"] = [{"function_name": "move_home"}]
        proposal["source_observation_fingerprint"] = observation_fingerprint(observations)
    else:
        proposal["source_observation_fingerprint"] = "changed"
    with pytest.raises(ValueError):
        asyncio.run(validate_saved_proposal(recovery, proposal, observations))
    assert recovery._proposal == {}
    assert recovery.snapshot()["status"] == "blocked"
    owner.execute_recovery_macro.assert_not_called()


@pytest.fixture
def offline_example(captured, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example

    root = tmp_path / "recovery"
    source = root / "captured-run"
    source.mkdir(parents=True)
    path = source / "observations.json"
    write_record(path, captured)
    write_record(source / "run.json", {"status": "generating_outline"})
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda p: read_observations(p, root=root))
    forbidden = AssertionError("offline workflow contacted a live interface")
    bridge = SimpleNamespace(system_running=True, ros2_start=Mock(side_effect=forbidden),
                             start_system=AsyncMock(side_effect=forbidden), stop_system=AsyncMock(side_effect=forbidden))
    example = part_slippage_example.example_for_bridge(bridge)
    example.set_options(source="offline", scenario="Part slippage", safety_files=[], observation_path=str(path))
    return example, path, bridge


def test_observations_sorted_by_capture_time_and_readiness_is_derived(captured, tmp_path):
    root = tmp_path / "captures"
    for name, timestamp in (("zzzz", "2026-10-08T18:58:00+00:00"), ("aaaa", "2026-10-08T16:46:46-04:00"),
                            ("mmmm", None)):
        path = root / name / "observations.json"
        path.parent.mkdir(parents=True)
        record = {**deepcopy(captured), "run_id": name, "captured_at": timestamp}
        if name == "aaaa":
            record.update(offline_ready=False, missing=["obsolete assessment"])
        if name == "mmmm":
            record["prepared_recovery_request"] = {}
        write_record(path, record)
    records = list_observations(root=root)
    assert [row["run_id"] for row in records] == ["aaaa", "zzzz", "mmmm"]
    assert records[0]["offline_ready"] is True and records[0]["missing"] == []
    assert records[-1]["offline_ready"] is False
    assert records[-1]["missing"] == ["prepared_recovery_request"]


def test_missing_capture_context_explains_disabled_generate(offline_example):
    example, path, _ = offline_example
    record = json.loads(path.read_text())
    del record["prepared_recovery_request"]["recovery_resources"]
    write_record(path, record)
    state = example.prepare_offline()
    assert state["status"] == "blocked" and not state["can_generate_outline"]
    assert "prepared_recovery_request.recovery_resources" in state["outline_generate_reason"]
    assert state["observation_path"] == str(path)


@pytest.mark.parametrize("outline_mode,primitive_mode", [("auto", "auto"), ("manual", "manual"), ("auto", "manual")])
def test_offline_manual_policy_restores_gazebo_modes(outline_mode, primitive_mode):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample

    example = PartSlippageExample(SimpleNamespace(system_running=False))
    example.set_modes(outline_mode, primitive_mode)
    for _ in range(2):
        state = example.set_options(source="offline", scenario="Part slippage", safety_files=[])
        assert example.outline_mode == example.primitive_mode == "manual"
        assert state["outline_mode"] == state["primitive_mode"] == "manual"
        for modes in (("auto", "auto"), ("auto", "manual"), ("manual", "auto")):
            with pytest.raises(ValueError, match="Offline recovery always uses Manual"):
                example.set_modes(*modes)
        state = example.set_options(source="gazebo", scenario="Part slippage", safety_files=[])
        assert (state["outline_mode"], state["primitive_mode"]) == (outline_mode, primitive_mode)


@pytest.mark.parametrize("change,status", [
    ({}, "FEASIBLE"), ({"width_m": .3}, "INFEASIBLE"),
    ({"width_m": None}, "NEEDS_CONTEXT"), ({"holder": "other@localhost"}, "INFEASIBLE"),
    ({"holder_missing": True}, "NEEDS_CONTEXT"), ({"held_part": "other_part"}, "INFEASIBLE"),
    ({"controller_ready": False}, "INFEASIBLE"), ({"controller_ready": None}, "NEEDS_CONTEXT"),
    ({"execution_mode": "physical"}, "NEEDS_CONTEXT"),
])
def test_saved_grasp_capability_checks_recorded_tool_part_custody_without_motion(change, status):
    from cais_spade_llm.resources.robot.recovery_capability import check_robot_primitive_capability

    snapshot = {"held_part": change.get("held_part"), "controller_ready": change.get("controller_ready", True)}
    evidence = {"execution_mode": change.get("execution_mode", "simulation"),
        "controller_config": {"gripper": {"open": .11, "close": .015},
            "attach": {"robot_model_name": "robot", "primary_attach_link": "tool0"}, "services": {"attach": "/attach"}},
        "inputs": {"geometry": {"parts": {"model_map": {"part": "part_model"}}}},
        "pick_geometry": {"part": {"width_m": change.get("width_m", .022)}}}
    context = {} if change.get("holder_missing") else {"part_holder_resource_jid": change.get("holder")}
    before = deepcopy((snapshot, evidence, context))
    row = check_robot_primitive_capability(primitive="grasp_part", params={"model_name": "part_model", "part_name": "part"},
        task={"resource_jid": "robot", "part_name": "part"}, recovery_snapshot=snapshot,
        resource_evidence=evidence, part_context=context, static_capabilities={}, primitive_catalog=[], resource_jid="robot")
    assert row["feasibility_status"] == status
    assert row["evidence"]["command_sent"] is False
    if status == "FEASIBLE":
        assert row["evidence"]["composition_obligations"] == ["pick_target_grounded", "motion_landed_on_target"]
    assert (snapshot, evidence, context) == before


def test_saved_pick_support_calculates_declared_travel_and_optional_retreat_independently():
    from cais_spade_llm.resources.robot.recovery_capability import prepare_robot_primitive_parameters
    from cais_spade_llm.resources.recovery_parameter_evidence import evaluate_resource_primitive_condition

    task = {"resource_jid": "robot", "part_name": "part", "expected_start_state": {"resource_state": "idle"},
            "expected_end_state": {"resource_state": "picked"}}
    snapshot = {"held_part": None, "current_pose": {"x": .1, "y": .2, "z": 1.5, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}}
    evidence = {"resource_type": "robot", "execution_mode": "simulation", "frame": "world", "snapshot": deepcopy(snapshot),
        "observed_parts": {"part": {"current_pose": [.2, .1, 1., 0., 0., 0., 1.]}},
        "motion_config": {"pick_tcp_z_bias_min_m": .003, "pick_tcp_z_bias_max_m": .02,
            "approach_height_m": .2, "min_pick_tcp_z_m": 1.07, "tcp_offset_z": -.218,
            "tcp_transform_source": "recorded fixture", "controlled_link": "tool0", "tcp_link": "tcp"},
        "pick_geometry": {"part": {"part_height_m": .05, "board_center": {"z": .975}, "width_m": .022}},
        "pick_policy": {"part": {"surface_clearance_override_m": 0., "pick_z_adjustment_m": 0., "use_global_min_pick_tcp_z": True}},
        "controller_config": {"gripper": {"open": .11, "close": .015}}}
    move = {"primitive": "move_relative", "params": {"dx": 0., "dy": 0.,
        "dz": {"$sub": [{"$state": "_task_ctx", "path": ["travel_z"]},
                         {"$state": "_position", "path": ["z"]}]}}}
    retreat = {"primitive": "move_cartesian", "params": {
        "x": {"$state": "_task_ctx", "path": ["access_retreat_pose", "x"]}},
        "when": [{"predicate": "always", "condition": {"field": "task_ctx.access_retreat_pose", "operator": "exists"}}]}
    model = {"state_variables": {"resource_state": {"scope": "resource", "domain": ["idle", "picked"]}},
        "events": [{"event_name": "declared", "guards": {}, "updates": {"resource_state": {"set": "picked"}},
                    "primitive_support": [move, retreat]}],
        "primitive_catalog": [{"name": "move_relative", "required_params": ["dx", "dy", "dz"]},
                              {"name": "move_cartesian", "required_params": ["x"]}]}
    before = deepcopy((model, task, snapshot, evidence))
    row = prepare_robot_primitive_parameters(task=task, recovery_des_model=model, recovery_snapshot=snapshot,
                                             resource_evidence=evidence, resource_jid="robot")
    assert row["missing_facts"] == []
    assert len(row["parameter_witnesses"]) == 1
    assert row["parameter_witnesses"][0]["params"]["dz"] == 0.
    calculated = row["calculated_resource_evidence"]
    assert calculated["state"]["_task_ctx"]["access_retreat_pose"] is None
    assert "gripper_close_position" not in calculated["state"]["_task_ctx"]
    assert "held_part" not in calculated["state"]["_task_ctx"]
    condition = evaluate_resource_primitive_condition(step=model["events"][0]["primitive_support"][1], bindings={},
        snapshot=snapshot, resource_evidence=calculated, resource_jid="robot")
    assert condition["status"] == "not_applicable"
    assert (model, task, snapshot, evidence) == before


@pytest.mark.parametrize("part_name", [None, "part"])
def test_owner_parameters_exclude_other_part_context_and_calculate_only_required_fields(monkeypatch, part_name):
    from cais_spade_llm.resources.robot import recovery_capability as owner

    task = {"resource_jid": "robot", "part_name": part_name,
            "expected_start_state": {"resource_state": "failed"}, "expected_end_state": {"resource_state": "idle"}}
    primitive = {"name": "move_to_named_pose", "required_params": ["pose_name"]}
    model = {"state_variables": {"resource_state": {"scope": "resource", "domain": ["idle"]}},
        "events": [{"event_name": "declared", "updates": {"resource_state": {"set": "idle"}},
                    "primitive_support": [{"primitive": "move_to_named_pose", "params": {"pose_name": "home"}}]}],
        "primitive_catalog": [primitive]}
    snapshot = {"held_part": None, "controller_ready": True, "named_poses": ["home"]}
    evidence = {"state": {"_task_ctx": {"part_name": "other_part", "model_name": "other_model",
                    "travel_z": 100., "insert_pose": {"x": 99.}}}}
    before = deepcopy((task, model, snapshot, evidence))
    forbidden = Mock(side_effect=AssertionError("No part calculation is required for this declared primitive"))
    monkeypatch.setattr(owner, "calculate_recovery_pick_target", forbidden)
    monkeypatch.setattr(owner, "calculate_recovery_place_target", forbidden)
    result = owner.prepare_robot_primitive_parameters(task=task, recovery_des_model=model,
        recovery_snapshot=snapshot, resource_evidence=evidence, resource_jid="robot")
    assert result["missing_facts"] == result["conflicts"] == []
    assert result["parameter_witnesses"][0]["params"] == {"pose_name": "home"}
    calculated = result["calculated_resource_evidence"]
    assert calculated["state"]["_task_ctx"] == {}
    assert calculated["parameter_context_exclusions"][0]["recorded"] == evidence["state"]["_task_ctx"]
    checked = owner.check_robot_primitive_capability(primitive="move_to_named_pose", params={"pose_name": "home"},
        task=task, recovery_snapshot=snapshot, resource_evidence=calculated, static_capabilities={},
        primitive_catalog=[primitive], resource_jid="robot")
    assert checked["feasibility_status"] == "FEASIBLE"
    assert (task, model, snapshot, evidence) == before
    forbidden.assert_not_called()


@pytest.mark.parametrize("recorded_model", [False, True])
def test_owner_parameters_never_reuse_another_parts_declared_model(recorded_model):
    from cais_spade_llm.resources.robot.recovery_capability import prepare_robot_primitive_parameters

    task = {"resource_jid": "robot", "part_name": "part",
            "expected_start_state": {"resource_state": "idle"}, "expected_end_state": {"resource_state": "picked"}}
    model = {"state_variables": {"resource_state": {"scope": "resource", "domain": ["picked"]}},
        "events": [{"event_name": "declared", "updates": {"resource_state": {"set": "picked"}},
            "primitive_support": [{"primitive": "grasp_part", "params": {
                "model_name": {"$state": "_task_ctx", "path": ["model_name"]}}}]}],
        "primitive_catalog": [{"name": "grasp_part", "required_params": ["model_name"]}]}
    evidence = {"state": {"_task_ctx": {"part_name": "other_part", "model_name": "other_model"}},
        "inputs": {"geometry": {"parts": {"model_map": {"part": "part_model"} if recorded_model else {}}}}}
    before = deepcopy(evidence)
    result = prepare_robot_primitive_parameters(task=task, recovery_des_model=model, recovery_snapshot={},
        resource_evidence=evidence, resource_jid="robot")
    if recorded_model:
        assert result["parameter_witnesses"][0]["params"] == {"model_name": "part_model"}
        assert result["calculated_resource_evidence"]["sources"]["state/_task_ctx/model_name"].endswith("model_map/part")
    else:
        assert not result["parameter_witnesses"]
        assert "/resources/robot/resource_evidence/inputs/geometry/parts/model_map/part" in result["context_requests"]
        assert "model_name" not in result["calculated_resource_evidence"]["state"]["_task_ctx"]
    assert evidence == before


def test_owner_parameters_exclude_other_parts_step_output_without_reusing_coordinates():
    from cais_spade_llm.resources.robot.recovery_capability import prepare_robot_primitive_parameters

    model = {"state_variables": {"resource_state": {"scope": "resource", "domain": ["picked"]}},
        "events": [{"event_name": "declared", "updates": {"resource_state": {"set": "picked"}},
            "primitive_support": [{"primitive": "move_cartesian", "params": {
                "x": {"$step": "pick_targets", "path": ["tx"]}}}]}],
        "primitive_catalog": [{"name": "move_cartesian", "required_params": ["x"]}]}
    evidence = {"state": {"_task_ctx": {"part_name": "other_part"}},
                "step_outputs": {"pick_targets": {"tx": 123.}}}
    original = deepcopy(evidence)
    result = prepare_robot_primitive_parameters(task={"resource_jid": "robot", "part_name": "part",
        "expected_start_state": {"resource_state": "idle"},
        "expected_end_state": {"resource_state": "picked"}}, recovery_des_model=model,
        recovery_snapshot={}, resource_evidence=evidence, resource_jid="robot")
    calculated = result["calculated_resource_evidence"]
    assert "pick_targets" not in calculated["step_outputs"]
    assert calculated["excluded_step_outputs"]["pick_targets"] == {"tx": 123.}
    assert result["parameter_witnesses"] == []
    assert result["missing_facts"]
    assert evidence == original


@pytest.mark.parametrize("readiness,known,status", [(False, True, "INFEASIBLE"),
    (None, True, "NEEDS_CONTEXT"), (None, False, "NEEDS_CONTEXT")])
def test_named_pose_capability_distinguishes_false_from_missing_readiness(readiness, known, status):
    from cais_spade_llm.resources.robot.recovery_capability import check_robot_primitive_capability

    snapshot = {"named_poses": ["home"]}
    if known:
        snapshot["controller_ready"] = readiness
    result = check_robot_primitive_capability(primitive="move_to_named_pose", params={"pose_name": "home"},
        task={"resource_jid": "robot", "part_name": None}, recovery_snapshot=snapshot,
        resource_evidence={}, static_capabilities={}, primitive_catalog=[], resource_jid="robot")
    assert result["feasibility_status"] == status
    assert bool(result["context_requests"]) is (status == "NEEDS_CONTEXT")


def test_saved_assembly_target_uses_recorded_slot_origin_target_and_attachment():
    import math
    from cais_spade_llm.resources.robot.recovery_capability import calculate_recovery_place_target

    task = {"resource_jid": "robot", "part_name": "part", "expected_end_state": {"resource_location": "assembly_board-v1"}}
    snapshot = {"current_pose": {"x": .1, "y": .2, "z": 1.5, "qx": 0., "qy": 0., "qz": 0., "qw": 1.}}
    evidence = {"execution_mode": "simulation", "inputs": {"geometry": {
        "target_reference": {"source": "fixture", "surface_role": "assembly_slot", "target_point": "inserted_part_origin"},
        "parts": {"heights_m": {"part": .02}, "model_map": {"part": "part_model"}, "assembly_target_map": {"part": "fixture/slot"}},
        "assembly_board": {"center": {"x": .1, "y": .2, "z": 1.}, "slots": {"part": [.03, .04]},
            "slot_floor_z_m": 1.01, "target_origin_z_m": {"part": 1.02},
            "mating_contact_by_part": {"part": {"target_yaw_rad": math.pi / 2}}}}},
        "attachment": {"part_name": "part", "observed_attachment_complete": True,
                       "grasp_transform": [0., 0., -.2, 0., 0., 0., 1.]}}
    before = deepcopy(evidence)
    row = calculate_recovery_place_target(task=task, recovery_snapshot=snapshot, resource_evidence=evidence)
    assert row["status"] == "calculated"
    assert row["target"] == "fixture/slot"
    assert row["target_reference"] == evidence["inputs"]["geometry"]["target_reference"]
    assert row["target_origin_pose"] == {"x": .13, "y": .24000000000000002, "z": 1.02}
    assert row["insert_pose"]["z"] == pytest.approx(1.22)
    assert row["insert_pose"]["qz"] == pytest.approx(math.sin(math.pi / 4))
    assert row["slot_x"] == .13 and row["board_top_z"] == 1.01
    assert evidence == before
    del evidence["inputs"]["geometry"]["assembly_board"]["target_origin_z_m"]
    assert calculate_recovery_place_target(task=task, recovery_snapshot=snapshot, resource_evidence=evidence)["status"] == "NEEDS_CONTEXT"


def test_robot_snapshot_uses_supplied_capture_without_polling_or_cache_fallback():
    from cais_spade_llm.resources.robot.robot_profile import _robot_snapshot_builder

    live = Mock(side_effect=AssertionError("must not poll another pose"))
    owner = SimpleNamespace(_controller=SimpleNamespace(get_current_pose=live), _position={"x": 8., "y": 9., "z": 10.},
                            _current_state="picked", _held_part="part", _gripper_state="closed", executables={})
    pose = {"x": .1, "y": .2, "z": 1.5, "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    row = _robot_snapshot_builder(owner, recorded_pose=pose)
    assert row["current_pose"] == pose
    assert row["held_part"] == "part"
    assert _robot_snapshot_builder(owner, recorded_pose={})["current_pose"] is None
    assert owner._position == {"x": 8., "y": 9., "z": 10.}
    live.assert_not_called()


def test_observation_display_cache_invalidates_changed_capture(captured, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import recovery_observations

    path = tmp_path / "captures" / "source" / "observations.json"
    path.parent.mkdir(parents=True)
    write_record(path, captured)
    reader = Mock(wraps=recovery_observations.read_observations)
    monkeypatch.setattr(recovery_observations, "read_observations", reader)
    assert list_observations(root=path.parent.parent)[0]["offline_ready"]
    assert list_observations(root=path.parent.parent)[0]["offline_ready"]
    assert reader.call_count == 1
    write_record(path, {**captured, "prepared_recovery_request": {}})
    assert not list_observations(root=path.parent.parent)[0]["offline_ready"]
    assert reader.call_count == 2


def _native_archive_fixture(tmp_path):
    from cais_spade_llm.recovery_framework.recovery_observations import import_native_preview_model
    import hashlib

    capture = tmp_path / 'captured'
    capture.mkdir()
    observations = {'run_id': 'run', 'planning_scene_capture': {'model_fingerprint': 'model', 'scene_fingerprint': 'scene'},
                    'inputs': {'scene': {}}}
    source = capture / 'observations.json'
    source.write_text(json.dumps(observations))
    archive = tmp_path / 'recorded'
    archive.mkdir()
    parameters = archive / 'parameters.yaml'
    parameters.write_text('/**:\n  ros__parameters:\n    robot_description: exact\n')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    verification = archive / 'verification.json'
    verification.write_text(json.dumps({'parameters_sha256': digest(parameters), 'model_fingerprint': 'model',
        'fingerprint_verified': True, 'native_exit_code': 0, 'command_sent': False, 'execution_authorized': False}))
    provenance = archive / 'provenance.json'
    provenance.write_text(json.dumps({'source_sha256': digest(source), 'parameters_sha256': digest(parameters),
        'expected_model_fingerprint': 'model', 'exact_native_model_match': True,
        'source_verification_sha256': digest(verification)}))
    configuration = archive / 'configuration.json'
    configuration.write_text(json.dumps({'planning_scene_mounting_contacts': {'fixed_joints': ['exact_joint']}}))
    supplemental = archive / 'supplementary.json'
    supplemental.write_text(json.dumps({'source': str(configuration), 'sha256': digest(configuration),
        'declarations': {'fixed_joints': ['exact_joint']}}))
    kwargs = {'parameters_path': parameters, 'verification_path': verification, 'provenance_path': provenance,
              'supplementary_path': supplemental, 'supplementary_source_path': configuration}
    model = import_native_preview_model(source, **kwargs)
    return source, observations, model, kwargs


def test_native_preview_companion_preserves_capture_and_exact_provenance(tmp_path):
    from cais_spade_llm.recovery_framework.recovery_observations import (
        import_native_preview_model, read_native_preview_model, native_preview_observations)
    source, observations, archive, kwargs = _native_archive_fixture(tmp_path)
    original = source.read_bytes()
    assert import_native_preview_model(source, **kwargs) == archive
    assert read_native_preview_model(source, observations) == archive
    view = native_preview_observations(observations, archive)
    assert view['inputs']['scene']['planning_scene_mounting_contacts'] == {'fixed_joints': ['exact_joint']}
    assert 'planning_scene_mounting_contacts' not in observations['inputs']['scene']
    assert source.read_bytes() == original
    assert Path(archive['parameters_path']).read_bytes() == kwargs['parameters_path'].read_bytes()


@pytest.mark.parametrize('defect', ['source', 'model', 'null', 'parameters', 'manifest', 'symlink', 'escape', 'supplemental'])
def test_native_preview_companion_rejects_changed_or_unbound_facts(tmp_path, defect):
    from cais_spade_llm.recovery_framework.recovery_observations import read_native_preview_model
    import hashlib
    source, observations, archive, _ = _native_archive_fixture(tmp_path)
    manifest_path = Path(archive['manifest_path'])
    manifest = json.loads(manifest_path.read_text())
    if defect == 'source':
        source.write_text(source.read_text() + ' ')
    elif defect == 'model':
        observations['planning_scene_capture']['model_fingerprint'] = 'changed'
    elif defect == 'null':
        observations['inputs']['scene']['planning_scene_mounting_contacts'] = None
    elif defect == 'parameters':
        Path(archive['parameters_path']).write_text('changed')
    elif defect == 'manifest':
        manifest['binding']['run_id'] = 'other'
    elif defect == 'symlink':
        target = Path(archive['parameters_path'])
        data = target.read_bytes()
        target.unlink()
        outside = tmp_path / 'outside.yaml'
        outside.write_bytes(data)
        target.symlink_to(outside)
    elif defect == 'escape':
        manifest['files']['parameters']['path'] = '../parameters.yaml'
    else:
        path = manifest_path.parent / manifest['files']['supplementary_source']['path']
        path.write_text('{}')
        manifest['files']['supplementary_source']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_native_preview_model(source, observations)


def test_native_preview_archive_import_rejects_unverified_parameters(tmp_path):
    from cais_spade_llm.recovery_framework.recovery_observations import import_native_preview_model
    source, _, _, kwargs = _native_archive_fixture(tmp_path)
    kwargs['parameters_path'].write_text('different')
    with pytest.raises(ValueError, match='provenance'):
        import_native_preview_model(source, **kwargs)


@pytest.mark.parametrize('kind,value,field', [(1, True, 'bool_value'), (2, 3, 'integer_value'),
    (3, 3., 'double_value'), (4, 'exact', 'string_value'), (6, [False], 'bool_array_value'),
    (7, [3], 'integer_array_value'), (8, [3.], 'double_array_value'), (9, ['exact'], 'string_array_value')])
def test_captured_native_parameter_values_preserve_declared_types(kind, value, field):
    from cais_spade_llm.recovery_framework.recovery_observations import _captured_native_parameters
    scene = {'robot_description': 'urdf', 'robot_description_semantic': 'srdf', 'native_model_parameters': {
        'robot_description': {'type': 4, 'string_value': 'urdf'},
        'robot_description_semantic': {'type': 4, 'string_value': 'srdf'},
        'robot_description_planning.exact_parameter': {'type': kind, field: value}}}
    assert _captured_native_parameters(scene)['robot_description_planning.exact_parameter'] == value
    assert type(_captured_native_parameters(scene)['robot_description_planning.exact_parameter']) is type(value)


@pytest.mark.parametrize('row', [{'type': 0, 'string_value': 'not unset'}, {'type': 1, 'bool_value': 0}, {'type': 2, 'integer_value': False},
    {'type': 3, 'double_value': 1}, {'type': 3, 'double_value': float('nan')},
    {'type': 5, 'byte_array_value': [0]}, {'type': 7, 'integer_array_value': []}, {'type': None}])
def test_captured_native_parameters_do_not_fill_missing_or_change_types(row):
    from cais_spade_llm.recovery_framework.recovery_observations import _captured_native_parameters
    scene = {'robot_description': 'urdf', 'robot_description_semantic': 'srdf', 'native_model_parameters': {
        'robot_description': {'type': 4, 'string_value': 'urdf'},
        'robot_description_semantic': {'type': 4, 'string_value': 'srdf'},
        'robot_description_planning.exact_parameter': row}}
    with pytest.raises(ValueError):
        _captured_native_parameters(scene)


def test_native_archive_failed_verification_is_not_published(tmp_path):
    from cais_spade_llm.recovery_framework.recovery_observations import import_native_preview_model
    import hashlib
    import shutil
    source, _, _, kwargs = _native_archive_fixture(tmp_path)
    shutil.rmtree(source.parent / 'native_preview_model')
    verification = json.loads(kwargs['verification_path'].read_text())
    verification['fingerprint_verified'] = False
    kwargs['verification_path'].write_text(json.dumps(verification))
    provenance = json.loads(kwargs['provenance_path'].read_text())
    provenance['source_verification_sha256'] = hashlib.sha256(kwargs['verification_path'].read_bytes()).hexdigest()
    kwargs['provenance_path'].write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='independently verified'):
        import_native_preview_model(source, **kwargs)
    assert not (source.parent / 'native_preview_model').exists()


def test_native_companion_distinguishes_boolean_and_integer_declarations(tmp_path):
    from cais_spade_llm.recovery_framework.recovery_observations import read_native_preview_model
    import hashlib
    source, observations, archive, _ = _native_archive_fixture(tmp_path)
    manifest_path = Path(archive['manifest_path'])
    manifest = json.loads(manifest_path.read_bytes())
    observations['inputs']['scene']['planning_scene_mounting_contacts'] = {'flag': False}
    source.write_text(json.dumps(observations))
    manifest['binding']['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest['binding']['observation_fingerprint'] = observation_fingerprint(observations)
    for key, obj in [('supplementary_source', {'planning_scene_mounting_contacts': {'flag': 0}}),
                     ('supplementary_model_declarations', {'declarations': {'flag': 0}})]:
        path = manifest_path.parent / manifest['files'][key]['path']
        if key == 'supplementary_model_declarations':
            obj['sha256'] = manifest['files']['supplementary_source']['sha256']
        path.write_text(json.dumps(obj))
        manifest['files'][key]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='captured values'):
        read_native_preview_model(source, observations)


@pytest.mark.parametrize('schema', [1, 2])
@pytest.mark.parametrize('unset', [False, True])
def test_new_capture_native_archive_verifies_and_preserves_typed_source(tmp_path, monkeypatch, unset, schema):
    import hashlib
    from cais_spade_llm.recovery_framework import recovery_motion_preview
    from cais_spade_llm.recovery_framework.capture_files import read_capture, write_capture
    from cais_spade_llm.recovery_framework.recovery_observations import (
        read_native_preview_model,
        save_native_preview_model_from_capture,
    )

    source = tmp_path / 'observations.json'
    record = {'run_id': 'new-capture', 'planning_scene_capture': {'scene_fingerprint': 'scene',
        'model_fingerprint': 'model', 'robot_description': 'urdf', 'robot_description_semantic': 'srdf',
        'native_model_parameters': {'robot_description': {'type': 4, 'string_value': 'urdf'},
            'robot_description_semantic': {'type': 4, 'string_value': 'srdf'},
            'robot_description_planning.exact': {'type': 3, 'double_value': 3.}}}}
    unset_name = 'robot_description_kinematics.KMR_iiwa_arm.kinematics_solver_ik_links'
    if unset:
        record['planning_scene_capture']['native_model_parameters'][unset_name] = {'type': 0}
    if schema == 2:
        # Raw facts and derived views are separate files; the archive binds to both.
        write_capture(source, {**record, 'schema_version': 2, 'prepared_recovery_request': {'derived': True}})
    else:
        source.write_text(json.dumps(record))
    original = source.read_bytes()

    async def verify(path, *, expected_model_fingerprint, artifact_directory, cancel_event):
        import yaml

        assert expected_model_fingerprint == 'model'
        assert unset_name not in yaml.safe_load(path.read_text())['/**']['ros__parameters']
        (artifact_directory / 'fingerprint-goal.cdr').write_bytes(b'exact goal')
        (artifact_directory / 'fingerprint-result.cdr').write_bytes(b'exact result')
        return {'parameters_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'model_fingerprint': 'model',
                'fingerprint_verified': True, 'native_exit_code': 0, 'command_sent': False, 'execution_authorized': False}
    monkeypatch.setattr(recovery_motion_preview, 'verify_native_model_parameters', verify)
    monkeypatch.setattr(recovery_motion_preview, '_new_node', lambda: pytest.fail('Live interface called'))
    result = asyncio.run(save_native_preview_model_from_capture(source))
    assert source.read_bytes() == original
    directory = Path(result['manifest_path']).parent
    assert json.loads((directory / 'typed_parameters.json').read_text()) == record['planning_scene_capture']['native_model_parameters']
    verification = json.loads((directory / 'verification.json').read_text())
    assert verification['unset_parameter_names'] == ([unset_name] if unset else [])
    assert (directory / 'fingerprint-goal.cdr').read_bytes() == b'exact goal'
    assert asyncio.run(save_native_preview_model_from_capture(source)) == result
    assert len(list(tmp_path.glob('native_model_capture_*'))) == 1
    assert read_native_preview_model(source, read_capture(source)) == result


def test_unset_native_description_cannot_supply_a_model():
    from cais_spade_llm.recovery_framework.recovery_observations import _captured_native_parameters

    scene = {'robot_description': 'urdf', 'robot_description_semantic': 'srdf', 'native_model_parameters': {
        'robot_description': {'type': 0}, 'robot_description_semantic': {'type': 4, 'string_value': 'srdf'}}}
    with pytest.raises(ValueError, match='description is unset'):
        _captured_native_parameters(scene)


@pytest.mark.parametrize('defect', [None, 'content', 'source_hash', 'foreign', 'cycle'])
def test_native_model_reconnects_only_identical_hash_bound_saved_copies(tmp_path, defect):
    import hashlib
    from cais_spade_llm.recovery_framework.recovery_observations import read_native_preview_model
    source, observations, archive, _ = _native_archive_fixture(tmp_path)
    copied = tmp_path / 'offline' / 'observations.json'
    copied.parent.mkdir()
    saved = {**deepcopy(observations), 'source_path': str(source),
             'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    if defect == 'content':
        saved['planning_scene_capture']['scene_fingerprint'] = 'different'
    elif defect == 'source_hash':
        saved['source_sha256'] = 'different'
    elif defect == 'foreign':
        external = tmp_path / 'another' / 'run' / 'observations.json'
        external.parent.mkdir(parents=True)
        external.write_bytes(source.read_bytes())
        saved['source_path'] = str(external)
    elif defect == 'cycle':
        saved['source_path'] = str(copied)
    copied.write_text(json.dumps(saved))
    loaded = deepcopy(saved)
    loaded['source_path'] = str(copied)
    loaded['source_sha256'] = hashlib.sha256(copied.read_bytes()).hexdigest()
    if defect is None:
        assert read_native_preview_model(copied, loaded) == archive
    else:
        with pytest.raises(ValueError):
            read_native_preview_model(copied, loaded)
