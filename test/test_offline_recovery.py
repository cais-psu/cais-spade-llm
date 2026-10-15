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
    monkeypatch.setattr(offline_recovery, "RECORDS", tmp_path / "recovery")
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
    with pytest.raises(ValueError, match='Conflicting recorded resource state'):
        offline_recovery.create_offline_recovery(captured, outline_mode='manual', primitive_mode='manual')
    assert captured == before


def test_saved_m1_without_descriptor_does_not_block_unrelated_preparation(captured):
    from cais_spade_llm.recovery_framework.outline_validation import validate_resource_outline_payload

    jid = "recovery-resource-5@localhost"
    entry = {"resource_jid": jid, "resource_type": "resource", "recovery_des_model": {},
        "static_capabilities": {}, "recovery_snapshot": {"resource_type": "resource", "current_state": "unknown",
            "current_location": "M1", "availability": "unknown", "occupancy": {}}}
    captured["jids"]["M1"] = jid
    captured["prepared_recovery_request"]["recovery_resources"][jid] = entry
    captured["resource_evidence"] = {jid: {"snapshot": {"resource_state": "idle", "part_name": None}}}
    before = deepcopy(captured)
    session = offline_recovery.create_offline_recovery(captured, outline_mode="manual", primitive_mode="manual")

    async def prepare():
        session.begin(captured["failure"], generate_outline=False, defer_observation_save=True)
        await session._task

    asyncio.run(prepare())
    assert session.snapshot()["status"] == "outline_ready", session.snapshot()["message"]
    assert session._prepared["recovery_resources"][jid]["recovery_des_model"] == {}
    assert "recovery_des_model_fingerprint" not in session._prepared["recovery_resources"][jid]
    assert captured == before
    owner = session.agent.resources[jid]
    with pytest.raises(ValueError, match="descriptor_fingerprint"):
        owner.recovery_des_model(snapshot=owner.get_recovery_snapshot())
    task = {"outline_id": "M1_event", "resource_jid": jid, "expected_end_state": {"resource_state": "completed"}}
    result = validate_resource_outline_payload(owner, {"primitive_support_semantics": "successor_primitive_union",
        "candidates": [{"task": task}]}, offline=True)["results"][0]
    assert result["allowed"] is False and result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["candidate_fingerprint"] == recovery_validation_fingerprint(task)
    assert result["geometry_evidence"]["status"] == "needs_context"
    assert result["execution_authorized"] is False
    assert captured == before


def test_fresh_generate_with_all_checks_uses_manual_session(captured, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example, part_slippage_recovery

    path = tmp_path / "observations.json"
    write_record(path, captured)
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda source: read_observations(source, root=tmp_path))
    generated = AsyncMock()
    monkeypatch.setattr(part_slippage_recovery.PartSlippageRecovery, "_generate_outline", generated)
    example = part_slippage_example.PartSlippageExample(SimpleNamespace(system_running=False))
    example.set_options(source="offline", scenario="Part slippage", safety_files=[], observation_path=str(path),
        check_ra_feasibility=True, check_physical_evidence=True, check_cca_safety=True, defer_preparation=True)

    async def run():
        state = await example.action("generate_outline_fresh")
        assert state["busy"] and state["validation_checks_locked"]
        await example._task
        assert example._draft is None and example._offline is not None
        assert example._offline.snapshot()["outline_mode"] == "manual"
        assert example._offline.snapshot()["primitive_mode"] == "manual"
        generated.assert_awaited_once()
        assert not example.snapshot()["validation_checks_locked"]

    asyncio.run(run())


def test_fresh_full_preparation_reports_needed_context_without_error_trace(captured, tmp_path, monkeypatch, caplog):
    from cais_spade_llm.recovery_framework import part_slippage_example, part_slippage_recovery

    path = tmp_path / "observations.json"
    write_record(path, captured)
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda source: read_observations(source, root=tmp_path))
    generated = AsyncMock()
    monkeypatch.setattr(part_slippage_recovery.PartSlippageRecovery, "_generate_outline", generated)

    def needs_context(self, *args, **kwargs):
        raise ValueError("NEEDS_CONTEXT: conflicting recorded observed_pose for gear_small")

    monkeypatch.setattr(part_slippage_recovery.PartSlippageRecovery, "_install_prepared", needs_context)
    example = part_slippage_example.PartSlippageExample(SimpleNamespace(system_running=False))
    example.set_options(source="offline", scenario="Part slippage", safety_files=[], observation_path=str(path),
        check_ra_feasibility=True, check_physical_evidence=True, check_cca_safety=True, defer_preparation=True)

    async def run():
        await example.action("generate_outline_fresh")
        await example._task

    asyncio.run(run())
    assert example.snapshot()["status"] == "NEEDS_CONTEXT"
    assert example.snapshot()["outline"] == []
    assert not example.snapshot()["validation_checks_locked"]
    generated.assert_not_called()
    assert not any(record.exc_info for record in caplog.records)


def test_saved_generation_retains_missing_physical_compatibility(captured, monkeypatch):
    from cais_spade_llm.recovery_framework.recovery_observations import (
        offline_preparation_missing, processing_capture_compatibility,
    )

    captured['processing_observations'] = {'M2': {
        'run_id': captured['run_id'], 'resource_id': 'M2', 'model_fingerprint': 'local-fixture',
        'simulation_clock_sec': 84.0, 'revision': 4, 'active_job': None,
        'selected_program': 'circle', 'jobs': {},
    }}
    captured['physical_evidence'] = {'resources': {'ur5e-4': {'simulation_time': 85.65}}}
    original = deepcopy(captured)
    assert not offline_preparation_missing(captured)
    compatibility = processing_capture_compatibility(captured)
    assert compatibility['status'] == 'NEEDS_CONTEXT'
    session = offline_recovery.create_offline_recovery(captured, outline_mode='manual', primitive_mode='manual')
    generate = AsyncMock(side_effect=AssertionError('Preparation must not call the LLM'))
    monkeypatch.setattr(session, '_generate_outline', generate)

    async def prepare():
        session.begin(captured['failure'], generate_outline=False)
        await session._task

    asyncio.run(prepare())
    assert session.snapshot()['status'] == 'outline_ready'
    assert 'Physical validation remains NEEDS_CONTEXT' in session.snapshot()['message']
    assert session._prepared['physical_evidence']['processing_capture_compatibility'] == compatibility
    assert session.snapshot()['execution_authorized'] is False
    assert session.snapshot()['outline_mode'] == session.snapshot()['primitive_mode'] == 'manual'
    generate.assert_not_called()
    assert captured == original


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
    session = offline_recovery.create_offline_recovery(captured, outline_mode="auto", primitive_mode="auto", **overrides)
    assert (session.agent.model, session.agent.reasoning_effort) == expected
    state = session.snapshot()
    assert (state["llm_model"], state["llm_reasoning_effort"]) == expected
    settings = state["llm_settings"]
    assert (settings["model_source"], settings["reasoning_effort_source"]) == sources
    assert settings["model_override"] == overrides.get("model")
    assert settings["reasoning_effort_override"] == overrides.get("reasoning_effort")
    assert settings["saved_model"] == saved.get("llm_model")
    assert settings["saved_reasoning_effort"] == saved.get("llm_reasoning_effort")
    assert json.loads((session.path / "run.json").read_text())["llm_settings"] == settings
    assert state["outline_mode"] == state["primitive_mode"] == "manual"
    assert captured == original
    assert session._observations == original


@pytest.mark.parametrize("overrides", [
    {"model": ""}, {"model": " "}, {"model": " gpt-5.6"}, {"model": 5.6},
    {"reasoning_effort": ""}, {"reasoning_effort": "automatic"}, {"reasoning_effort": "LOW"},
    {"reasoning_effort": 0}, {"model": "gpt-5.4", "reasoning_effort": "minimal"},
    {"model": "gpt-4o", "reasoning_effort": "low"}, {"model": "gpt-4o", "reasoning_effort": "none"},
])
def test_offline_attempt_rejects_invalid_explicit_model_settings(captured, overrides):
    original = deepcopy(captured)
    with pytest.raises(ValueError, match="model|reasoning_effort"):
        offline_recovery.create_offline_recovery(captured, outline_mode="manual", primitive_mode="manual", **overrides)
    assert captured == original
    assert not offline_recovery.RECORDS.exists()


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
    session = offline_recovery.create_offline_recovery(
        captured, outline_mode="manual", primitive_mode="manual", model=model, reasoning_effort=effort)
    with capture_requests(directory=tmp_path / "capture", run_id=session.snapshot()["run_id"], stage="outline", turn=1):
        response = asyncio.run(session.agent.ask_llm_structured(
            "Read recorded facts", response_format={"name": "test", "strict": False,
                                                    "schema": {"type": "object"}}))
    assert response == {"decision": "need_context"}
    provider.assert_called_once()
    request = json.loads(next((tmp_path / "capture").glob("requests/*/*_request.json")).read_text())
    assert request["payload"] == provider.call_args.kwargs
    assert request["payload"]["model"] == model
    if model == "gpt-4o":
        assert "reasoning_effort" not in request["payload"]
        assert "reasoning_effort" not in session.agent._last_structured_request
        assert session.snapshot()["llm_settings"]["reasoning_effort_source"] == "not_applicable"
    else:
        assert request["payload"]["reasoning_effort"] == effort
    raw_response = json.loads(next((tmp_path / "capture").glob("requests/*/*_response.json")).read_text())
    assert raw_response["raw_response"]["model"] == model + "-returned"
    assert captured == original


def test_offline_resources_share_one_owned_capture_and_keep_accessors_detached(captured):
    entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].values()))
    for index in range(1, 13):
        jid = f"recovery-resource-{index}@localhost"
        captured["jids"][f"recorded-resource-{index}"] = jid
        captured["prepared_recovery_request"]["recovery_resources"][jid] = deepcopy(entry)
        captured["prepared_recovery_request"]["recovery_resources"][jid]["resource_jid"] = jid
    captured["context_revision"] = 7
    original = deepcopy(captured)
    session = offline_recovery.create_offline_recovery(captured, outline_mode="manual", primitive_mode="manual")
    resources = list(session.agent.resources.values())
    assert len(resources) == 12
    shared = resources[0]._observations
    assert shared == original and shared is not captured
    assert all(resource._observations is shared for resource in resources)
    assert session._observations is not shared
    for resource in resources:
        snapshot = resource.get_recovery_snapshot()
        snapshot["current_pose"]["x"] = 99.
        evidence = resource.get_recovery_resource_evidence()
        evidence["snapshot"]["current_pose"]["x"] = 99.
        detected = resource.detect_parts("gear_small")
        detected[0]["x"] = 99.
    assert shared == original
    captured["context_revision"] = 8
    captured["part_tracker"]["gear_small"]["observed_pose"]["x"] = 99.
    assert shared == original
    with pytest.raises(ValueError, match="another source capture"):
        resources[0].install_recovery_resource_evidence(
            observations=captured, source_fingerprint=observation_fingerprint(captured))
    assert shared == original
    retry = offline_recovery.create_offline_recovery(original, outline_mode="manual", primitive_mode="manual")
    assert next(iter(retry.agent.resources.values()))._observations is not shared


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


def test_saved_ra_support_uses_local_declared_named_pose_and_preserves_capture(captured, monkeypatch):
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
    from cais_spade_llm.recovery_framework.outline_validation import validate_resource_outline_payload

    monkeypatch.setattr(RobotAgent, "__init__", lambda *args, **kwargs: pytest.fail("live robot construction"))
    monkeypatch.setattr(RobotAgent, "check_recovery_primitive_feasibility", lambda *args, **kwargs: pytest.fail("live capability evaluator"))
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    original = deepcopy(captured)
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    task = {"outline_id": "RECOVERY1", "resource_jid": jid, "event_name": "new_event",
            "expected_start_state": {"resource_state": "failed", "held_part": None},
            "expected_end_state": {"resource_state": "idle", "held_part": None}}
    row = validate_resource_outline_payload(resource, {
        "validation_contract_version": 2, "primitive_support_semantics": "successor_primitive_union",
        "candidates": [{"task": task, "physical_input": {}}]}, offline=True)["results"][0]
    assert row["allowed"] is True, row
    assert row["feasibility_status"] == "FEASIBLE"
    assert row["primitive_support"]["semantics"] == "successor_primitive_union"
    assert row["primitive_support"]["primitives"][0]["evidence"]["evidence_source"] == "saved_observations"
    assert row["geometry_evidence"]["status"] == "needs_context"
    assert "trace" not in row["primitive_support"]
    assert captured == original


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


def test_saved_parameter_witness_requires_recorded_provenance(captured):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    row = resource.validate_recovery_primitive_support(task={}, recovery_des_model=entry["recovery_des_model"],
        recovery_snapshot=resource.get_recovery_snapshot(), physical_input={"primitive_support_semantics": "successor_primitive_union",
            "parameter_witnesses": [{"primitive": "move_to_named_pose", "event_name": "move_home", "step_index": 0,
                                     "params": {"pose_name": "home"}}]})
    assert row["feasibility_status"] == "NEEDS_CONTEXT"
    assert row["context_requests"] == [f"/resources/{jid}/parameter_witnesses"]


def test_saved_owner_resolves_scoped_state_expressions_independently(captured):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    model = entry["recovery_des_model"]
    model["primitive_catalog"] = [{"name": "move_relative", "frame": "world", "required_params": ["dx", "dy", "dz"],
                                  "params": {axis: {"type": "number"} for axis in ("dx", "dy", "dz")}}]
    model["events"][0]["primitive_support"] = [{"primitive": "move_relative", "params": {
        "dx": 0., "dy": 0., "dz": {"$sub": [{"$state": "_task_ctx", "path": ["travel_z"]},
                                               {"$state": "_position", "path": ["z"]}]}}}]
    entry["static_capabilities"]["gripper_reach"] = {"frame": "world", "origin_pose": {"x": 0., "y": 0., "z": 1.},
        "max_xy_radius_m": .7, "z_min_m": .8, "z_max_m": 1.6, "tolerance_m": .01}
    entry["recovery_snapshot"]["current_pose_frame"] = "world"
    entry["resource_evidence"] = {"state": {"_task_ctx": {"travel_z": 1.3}}, "source": {"run_id": captured["run_id"]}}
    before = deepcopy(captured)
    owner = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    task = {"resource_jid": jid, "event_name": "new_event",
            "expected_start_state": {"resource_state": "failed"}, "expected_end_state": {"resource_state": "idle"}}
    row = owner.validate_recovery_primitive_support(task=task, recovery_des_model=model,
        recovery_snapshot=owner.get_recovery_snapshot(), physical_input={"primitive_support_semantics": "successor_primitive_union"})
    assert row["feasibility_status"] == "FEASIBLE", row
    witness = row["primitive_support"]["primitives"][0]
    assert witness["params"]["dz"] == pytest.approx(.1)
    assert witness["parameter_witness"]["source_capture"]["run_id"] == captured["run_id"]
    assert witness["parameter_witness"]["source_kind"] == "resource_owned_calculation"
    assert "trace" not in row["primitive_support"]
    assert captured == before


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
def test_explicit_owner_pose_is_not_replaced_or_conflicted_by_later_legacy_sample(captured, monkeypatch, pose_format):
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
    calls = []

    def capability(resource, **kwargs):
        calls.append(kwargs)
        return {"allowed": False, "feasibility_status": "NEEDS_CONTEXT", "reason": "separate capability evidence required"}

    monkeypatch.setattr(offline_recovery.ResourceAgent, "validate_recovery_primitive_support", capability)
    physical = {"primitive_support_semantics": "successor_primitive_union", "resource_evidence": deepcopy(evidence)}
    result = owner.validate_recovery_primitive_support(task={}, recovery_des_model=entry["recovery_des_model"],
        recovery_snapshot=owner.get_recovery_snapshot(), physical_input=physical)
    assert len(calls) == 1 and result["reason"] == "separate capability evidence required"
    physical["resource_evidence"]["source"]["run_id"] = "unrelated_capture"
    result = owner.validate_recovery_primitive_support(task={}, recovery_des_model=entry["recovery_des_model"],
        recovery_snapshot=owner.get_recovery_snapshot(), physical_input=physical)
    assert len(calls) == 1 and result["reason"] == "Resource evidence is not linked to this saved resource capture"
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


def test_saved_supplement_requires_matching_checkpoint_and_preserves_source(captured):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    captured.update(inputs={"model": "fixture"}, program_revision="fixture", pending_nodes=[],
                    context_revision=4, acknowledgements={}, transitions=[], monitor_history={"active": False})
    captured["physical_evidence"] = {"snapshot": {"resources": {"ur5e-4": {"held_part": None}}},
        "resource_models": {"ur5e-4": {"model": "fixture"}}, "geometry": {"model": "fixture"},
        "resources": {"ur5e-4": {"joint_names": ["joint"], "joint_positions": [0.]}}}
    before = deepcopy(captured)
    owner = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    supplement = deepcopy(captured)
    supplement["captured_at"] = "2026-10-09T00:00:00+00:00"
    supplement["resource_evidence"] = {jid: {"state": {"_task_ctx": {"travel_z": 1.3}},
        "source": {"captured_at": supplement["captured_at"]}}}
    with pytest.raises(ValueError, match="another source capture"):
        owner.install_recovery_resource_evidence(observations=supplement, source_fingerprint="wrong")
    facts = owner.install_recovery_resource_evidence(observations=supplement,
        source_fingerprint=observation_fingerprint(captured))
    assert facts["state"]["_task_ctx"]["travel_z"] == 1.3
    assert facts["source"]["captured_at"] == supplement["captured_at"]
    assert owner._observations == captured == before
    changed = deepcopy(supplement)
    changed["physical_evidence"]["resources"]["ur5e-4"]["joint_positions"] = [1.]
    with pytest.raises(ValueError, match="changes the checkpoint"):
        owner.install_recovery_resource_evidence(observations=changed,
            source_fingerprint=observation_fingerprint(captured))
    assert owner.get_recovery_resource_evidence() == facts
    restored = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    product = offline_recovery.SavedRecoveryProduct(captured, [restored])
    session = {"validation_resource_evidence_captures": [{
        "source_observation_fingerprint": observation_fingerprint(captured), "observations": supplement}]}
    product.restore_recovery_resource_evidence(session, source_fingerprint=observation_fingerprint(captured))
    assert restored.get_recovery_resource_evidence() == facts
    assert restored._observations == before
    product.restore_recovery_resource_evidence({}, source_fingerprint=observation_fingerprint(captured))
    assert "_task_ctx" not in restored.get_recovery_resource_evidence()["state"]
    with pytest.raises(ValueError, match="another source capture"):
        product.restore_recovery_resource_evidence(session, source_fingerprint="wrong")


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


def test_saved_parameter_witness_inherits_only_actual_capture_metadata(captured, monkeypatch):
    from cais_spade_llm.resources import recovery_feasibility

    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    witness = {"primitive": "move_to_named_pose", "event_name": "move_home", "step_index": 0, "params": {"pose_name": "home"}}
    entry["parameter_witnesses"] = [witness]
    captured["context_revision"] = 7
    before = deepcopy(captured)
    received = []
    monkeypatch.setattr(recovery_feasibility, "validate_primitive_support", lambda **kwargs: received.append(kwargs) or {})
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    resource.validate_recovery_primitive_support(task={}, recovery_des_model=entry["recovery_des_model"],
        recovery_snapshot=resource.get_recovery_snapshot(), physical_input={"primitive_support_semantics": "successor_primitive_union",
            "parameter_witnesses": [witness]})
    row = received[0]["parameter_witnesses"][0]
    assert row["params"] == witness["params"]
    assert row["source"] == f"/resources/{jid}/parameter_witnesses"
    assert row["captured_at"] == captured["captured_at"] and row["revision"] == 7
    assert row["source_capture"]["run_id"] == captured["run_id"]
    assert captured == before


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


def test_saved_resource_preserves_owner_and_primitive_input_isolation(captured, monkeypatch):
    from cais_spade_llm.resources import recovery_feasibility

    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    witness = {"primitive": "move_to_named_pose", "event_name": "move_home", "step_index": 0,
        "parameter_bindings": {"pose_name": "home"}, "params": {"pose_name": "home"}}
    entry["parameter_witnesses"] = [witness]
    captured["context_revision"] = 7
    resource = offline_recovery.SavedRecoveryResource(jid, entry, captured)
    physical = {"primitive_support_semantics": "successor_primitive_union", "parameter_witnesses": [witness],
        "part_context": {"part_state": "ready"}, "physical_evidence": {
            "snapshot": {"resources": {"peer": {"held_part": None}}, "parts": {}},
            "resource_models": {"peer": {"recorded": [1]}}, "jids": {"peer": "peer@localhost"}}}
    original, original_capture, original_entry = deepcopy(physical), deepcopy(captured), deepcopy(resource._entry)
    received, checked = [], []

    def evaluate(*, primitive, params, task, recovery_snapshot, part_context, grounded_action,
                 resource_evidence, physical_evidence):
        assert physical_evidence == original["physical_evidence"]
        assert part_context == original["part_context"]
        assert recovery_snapshot == resource.get_recovery_snapshot()
        checked.append(deepcopy(resource_evidence))
        if len(checked) == 2:
            assert checked[0] == checked[1]
        physical_evidence["snapshot"]["resources"]["peer"]["held_part"] = "gear_small"
        physical_evidence["resource_models"]["peer"]["recorded"].append(2)
        resource_evidence["source"]["run_id"] = "changed"
        recovery_snapshot["held_part"] = "gear_small"
        part_context["part_state"] = "changed"
        return {"allowed": True, "feasibility_status": "FEASIBLE"}

    def validate_support(**kwargs):
        received.extend(deepcopy(kwargs["parameter_witnesses"]))
        for _ in range(2):
            assert kwargs["validate_primitive"](primitive="move_to_named_pose", params={"pose_name": "home"})["allowed"]
        return {"allowed": True, "feasibility_status": "FEASIBLE"}

    monkeypatch.setattr(resource, "check_recovery_primitive_feasibility", evaluate)
    monkeypatch.setattr(recovery_feasibility, "validate_primitive_support", validate_support)
    result = resource.validate_recovery_primitive_support(task={}, recovery_des_model=entry["recovery_des_model"],
        recovery_snapshot=resource.get_recovery_snapshot(), physical_input=physical)
    assert result["allowed"] is True and len(checked) == 2
    assert received[0]["parameter_bindings"] == witness["parameter_bindings"]
    assert received[0]["source"] == f"/resources/{jid}/parameter_witnesses"
    assert received[0]["source_capture"] == {"run_id": captured["run_id"], "captured_at": captured["captured_at"],
                                              "context_revision": 7}
    assert received[0]["captured_at"] == captured["captured_at"] and received[0]["revision"] == 7
    assert physical == original
    assert captured == original_capture
    assert resource._entry == original_entry
    assert resource._observations == original_capture


def test_offline_transition_checks_do_not_claim_physical_pass(captured):
    session = offline_recovery.create_offline_recovery(captured, outline_mode="manual", primitive_mode="manual")
    product = session.agent
    jid = next(iter(product.resources))
    task = {"outline_id": "RECOVERY_SEQ1", "resource_jid": jid, "event_name": "move_home",
            "expected_start_state": {"resource_state": "failed", "held_part": None},
            "expected_end_state": {"resource_state": "idle"}}
    async def check():
        reply = await product.request_recovery_outline_physical_validation(resource_jid=jid,
                    payload={"candidates": [{"candidate_index": 1, "task": task}]})
        row = reply["results"][0]
        assert row["transition_feasibility"]["allowed"] is True
        assert row["allowed"] is False
        assert row["physical_feasibility"]["status"] == "deferred"
        task["expected_start_state"]["held_part"] = "gear_small"
        rejected = await product.request_recovery_outline_physical_validation(resource_jid=jid,
                    payload={"candidates": [{"candidate_index": 1, "task": task}]})
        assert rejected["results"][0]["transition_feasibility"]["allowed"] is False
    asyncio.run(check())


@pytest.fixture
def recorded_robot(captured):
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    entry["static_capabilities"]["gripper_reach"] = {
        "frame": "world", "origin_pose": {"x": 0., "y": -.5, "z": 1.021},
        "max_xy_radius_m": .7, "z_min_m": .85, "z_max_m": 1.6, "tolerance_m": .01,
    }
    entry["static_capabilities"]["reachability"] = ["assembly_board-v1"]
    entry["recovery_snapshot"].update(current_location="home", held_part=None, gripper_state="open")
    task = {"event_name": "unregistered_event", "resource_jid": jid, "part_name": "gear_small",
            "expected_end_state": {"held_part": "gear_small", "part_location": jid,
                                   "resource_location": "home", "resource_state": "picked"}}
    physical = {"part_context": {"part_name": "gear_small", "part_location": None,
                                 "current_holder_resource_jid": None, "observed_pose_frame": "world",
                                 "observed_pose": deepcopy(captured["part_tracker"]["gear_small"]["observed_pose"])}}
    return captured, jid, entry, task, physical


@pytest.mark.parametrize("event_name", ["recover_pick", "arbitrary_transition", "place_insert"])
@pytest.mark.parametrize("combined_release", [False, True])
def test_recorded_reach_rejects_acquisition_without_using_event_name(recorded_robot, event_name, combined_release, monkeypatch):
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    capture, jid, entry, task, physical = recorded_robot
    task["event_name"] = event_name
    if combined_release:
        task["expected_end_state"].update(held_part=None, part_location="assembly_board-v1", part_state="assembled")
    live = Mock(side_effect=AssertionError("Recorded checks must not call a live feasibility interface"))
    monkeypatch.setattr(RobotAgent, "check_recovery_physical_feasibility", live)
    original = deepcopy(capture)
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    result = resource.check_recorded_recovery_feasibility(
        task=task, recovery_snapshot=entry["recovery_snapshot"], physical_input=physical)
    finding = next(row for row in result["findings"] if row["constraint_code"] == "workspace_unreachable")
    assert finding["feasibility_status"] == "INFEASIBLE"
    assert "0.8000" in finding["reason"]
    assert finding["evidence"]["checked_pose"] == physical["part_context"]["observed_pose"]
    assert finding["evidence"]["gripper_reach"] == entry["static_capabilities"]["gripper_reach"]
    assert finding["evidence"]["source_run_id"] == capture["run_id"]
    assert capture == original
    live.assert_not_called()


def test_recorded_reach_pass_preserves_other_part_custody_conflict(recorded_robot):
    capture, jid, entry, task, physical = recorded_robot
    entry["static_capabilities"]["gripper_reach"]["origin_pose"]["y"] = .5
    entry["recovery_snapshot"]["held_part"] = "KET4_Square_4mm"
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    result = resource.check_recorded_recovery_feasibility(
        task=task, recovery_snapshot=entry["recovery_snapshot"], physical_input=physical)
    assert [row["constraint_code"] for row in result["findings"]] == ["holder_conflict"]
    assert result["checks"][0]["check"] == "gripper_reach"
    assert result["checks"][0]["status"] == "passed"


@pytest.mark.parametrize("projected_holder, alias_holder, conflict", [
    (None, "stale-holder", False), ("recorded-holder", None, True),
])
def test_recorded_custody_prefers_explicit_projected_holder(recorded_robot, projected_holder, alias_holder, conflict):
    capture, jid, entry, task, physical = recorded_robot
    entry["static_capabilities"]["gripper_reach"]["origin_pose"]["y"] = .5
    physical["part_context"].update(part_holder_resource_jid=projected_holder, current_holder_resource_jid=alias_holder)
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    result = resource.check_recorded_recovery_feasibility(
        task=task, recovery_snapshot=entry["recovery_snapshot"], physical_input=physical)
    assert any(row["constraint_code"] == "holder_conflict" for row in result["findings"]) is conflict
    assert any(row["check"] == "gripper_reach" for row in result["checks"])


@pytest.mark.parametrize("change", ["missing_bounds", "missing_pose_frame", "other_pose_frame", "missing_bounds_frame",
                                    "invalid_bounds", "conflicting_bounds", "missing_pose", "nonfinite_pose"])
def test_recorded_reach_missing_facts_stay_needs_context(recorded_robot, change):
    capture, jid, entry, task, physical = recorded_robot
    if change == "missing_bounds":
        del entry["static_capabilities"]["gripper_reach"]
    elif change == "missing_pose_frame":
        del physical["part_context"]["observed_pose_frame"]
    elif change == "other_pose_frame":
        physical["part_context"]["observed_pose_frame"] = "other_frame"
    elif change == "missing_bounds_frame":
        del entry["static_capabilities"]["gripper_reach"]["frame"]
    elif change == "invalid_bounds":
        entry["static_capabilities"]["gripper_reach"]["max_xy_radius_m"] = None
    elif change == "conflicting_bounds":
        entry["recovery_adapter"] = {"gripper_reach": {**entry["static_capabilities"]["gripper_reach"], "max_xy_radius_m": 2.}}
    elif change == "missing_pose":
        physical["part_context"].pop("observed_pose")
    else:
        physical["part_context"]["observed_pose"]["x"] = float("nan")
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    result = resource.check_recorded_recovery_feasibility(
        task=task, recovery_snapshot=entry["recovery_snapshot"], physical_input=physical)
    assert result["findings"]
    assert all(row["feasibility_status"] == "NEEDS_CONTEXT" for row in result["findings"])


@pytest.mark.parametrize("source", ["recovery_snapshot", "recovery_adapter"])
def test_recorded_reach_uses_declared_snapshot_or_adapter_bounds(recorded_robot, source):
    capture, jid, entry, task, physical = recorded_robot
    entry.setdefault(source, {})["gripper_reach"] = entry["static_capabilities"].pop("gripper_reach")
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    result = resource.check_recorded_recovery_feasibility(
        task=task, recovery_snapshot=entry["recovery_snapshot"], physical_input=physical)
    assert result["findings"][0]["constraint_code"] == "workspace_unreachable"


@pytest.mark.parametrize("distance, rejected", [(.71, False), (.710001, True)])
def test_recorded_reach_respects_captured_tolerance(recorded_robot, distance, rejected):
    capture, jid, entry, task, physical = recorded_robot
    entry["static_capabilities"]["gripper_reach"]["origin_pose"] = {"x": 0., "y": 0., "z": 1.021}
    physical["part_context"]["observed_pose"].update(x=distance, y=0.)
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    result = resource.check_recorded_recovery_feasibility(
        task=task, recovery_snapshot=entry["recovery_snapshot"], physical_input=physical)
    assert bool(result["findings"]) is rejected


def test_recorded_robot_checks_use_projected_custody_and_check_target(recorded_robot):
    capture, jid, entry, task, physical = recorded_robot
    projected = {**entry["recovery_snapshot"], "held_part": "gear_small"}
    physical["part_context"].update(current_holder_resource_jid=jid)
    task["expected_end_state"].update(held_part=None, part_location="assembly_board-v1")
    physical["grounded_action"] = {"target": {"pose": {"x": 5., "y": 0., "z": 1.}, "frame": "world"}}
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    result = resource.check_recorded_recovery_feasibility(
        task=task, recovery_snapshot=projected, physical_input=physical)
    assert [row["evidence"].get("role") for row in result["findings"]] == ["target"]
    assert result["findings"][0]["constraint_code"] == "workspace_unreachable"


def test_recorded_carried_part_does_not_turn_named_resource_pose_into_station(recorded_robot):
    capture, jid, entry, task, physical = recorded_robot
    entry["recovery_snapshot"].update(held_part="gear_small", current_location="assembly_board-v1")
    physical["part_context"].update(part_holder_resource_jid=jid, part_location=jid)
    task["expected_end_state"].update(resource_location="home", held_part="gear_small", part_location=jid)
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    result = resource.check_recorded_recovery_feasibility(
        task=task, recovery_snapshot=entry["recovery_snapshot"], physical_input=physical)
    assert result["findings"] == []
    assert result["checks"][0]["check"] == "named_poses"
    assert result["checks"][0]["named_pose"] == "home"


def test_recorded_nonrobot_checks_do_not_require_robot_fields(recorded_robot):
    capture, jid, entry, task, physical = recorded_robot
    entry["resource_type"] = "machining"
    entry["static_capabilities"] = {}
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    result = resource.check_recorded_recovery_feasibility(task=task, recovery_snapshot={}, physical_input=physical)
    assert result["findings"] == result["checks"] == []


@pytest.mark.parametrize("conflicting", [False, True])
def test_recorded_destination_uses_snapshot_reachable_locations(recorded_robot, conflicting):
    capture, jid, entry, task, physical = recorded_robot
    entry["recovery_snapshot"].update(held_part="gear_small", reachable_locations=["assembly_board-v1"])
    physical["part_context"]["current_holder_resource_jid"] = jid
    task["expected_end_state"].update(held_part=None, part_location="assembly_board-v1")
    entry["static_capabilities"].pop("reachability")
    if conflicting:
        entry["recovery_adapter"] = {"reachability": ["Buffer For Machined parts"]}
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    result = resource.check_recorded_recovery_feasibility(
        task=task, recovery_snapshot=entry["recovery_snapshot"], physical_input=physical)
    if conflicting:
        assert result["findings"][0]["feasibility_status"] == "NEEDS_CONTEXT"
        declarations = result["findings"][0]["evidence"]["declarations"]
        assert declarations[0][0] == "recovery_snapshot.reachable_locations"
        assert declarations[1][0] == "recovery_adapter.reachability"
    else:
        assert result["findings"] == []
        assert result["checks"][0]["destination"] == "assembly_board-v1"


@pytest.mark.parametrize("pose_kind", ["observed", "requested", "other_part"])
def test_recorded_world_frame_requires_exact_gazebo_observation_provenance(recorded_robot, pose_kind):
    capture, jid, entry, task, physical = recorded_robot
    physical["part_context"].pop("observed_pose_frame")
    pose = physical["part_context"]["observed_pose"]
    capture.update(evidence_source="Gazebo")
    capture["failure"].update(part_name="gear_small", observed_drop_pose=deepcopy(pose),
                              requested_drop_pose={**pose, "z": 1.04})
    if pose_kind == "requested":
        physical["part_context"]["observed_pose"] = capture["failure"]["requested_drop_pose"]
    elif pose_kind == "other_part":
        capture["failure"]["part_name"] = "other_part"
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    result = resource.check_recorded_recovery_feasibility(
        task=task, recovery_snapshot=entry["recovery_snapshot"], physical_input=physical)
    assert result["findings"][0]["constraint_code"] == (
        "workspace_unreachable" if pose_kind == "observed" else "resource_validation_unavailable")


def test_recorded_part_state_only_change_does_not_invent_acquisition(recorded_robot):
    capture, jid, entry, task, physical = recorded_robot
    task["expected_end_state"] = {"part_state": "processed"}
    physical["part_context"].pop("observed_pose")
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    result = resource.check_recorded_recovery_feasibility(
        task=task, recovery_snapshot=entry["recovery_snapshot"], physical_input=physical)
    assert result["findings"] == []


@pytest.mark.parametrize("project_release", [False, True])
def test_saved_validation_projects_release_before_later_acquisition(recorded_robot, monkeypatch, project_release):
    from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent

    capture, jid, entry, task, physical = recorded_robot
    entry["static_capabilities"]["gripper_reach"]["origin_pose"]["y"] = .5
    entry["recovery_snapshot"].update(gripper_state="closed", held_part=None)
    if project_release:
        entry["recovery_snapshot"]["held_part"] = "KET4_Square_4mm"
    physical.update(use_projected_recovery_snapshot=project_release,
                    projected_recovery_snapshot={"held_part": None})
    monkeypatch.setattr(ResourceAgent, "check_recovery_transition_feasibility", lambda *_, **__: {"allowed": True})
    monkeypatch.setattr(offline_recovery, "derive_primitive_union", lambda **_: {
        "primitives": [{"primitive": "move_to_named_pose"}], "missing_support": []})
    resource = offline_recovery.SavedRecoveryResource(jid, entry, capture)
    product = offline_recovery.SavedRecoveryProduct(capture, [resource])
    reply = asyncio.run(product.request_recovery_outline_physical_validation(
        resource_jid=jid, payload={"candidates": [{"task": task, "physical_input": physical}]}))
    row = reply["results"][0]
    if project_release:
        assert row["findings"] == []
        assert row["physical_feasibility"]["status"] == "deferred"
        assert row["allowed"] is False
    else:
        assert row["physical_feasibility"]["status"] == "rejected"
        assert row["findings"][0]["constraint_code"] == "gripper_occupancy_conflict"


def test_actual_saved_capture_blocks_ur5e4_repick_without_live_calls(captured, monkeypatch):
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
    session = offline_recovery.create_offline_recovery(saved, outline_mode="manual", primitive_mode="manual")
    jid = saved["jids"]["ur5e-4"]
    task = {"outline_id": "candidate", "event_name": "anything", "resource_jid": jid, "part_name": "gear_small",
            "expected_start_state": {"resource_state": "idle", "resource_location": "home", "held_part": None,
                                     "part_location": None, "part_state": "misplaced"},
            "expected_end_state": {"resource_state": "picked", "resource_location": "home", "held_part": "gear_small",
                                   "part_location": jid, "part_state": "in_gripper"}}
    physical = {"use_projected_recovery_snapshot": True,
                "projected_recovery_snapshot": {"resource_state": "idle", "current_state": "idle", "held_part": None,
                                                "resource_location": "home", "current_location": "home"},
                "part_context": {"part_name": "gear_small", "observed_pose": saved["failure"]["observed_drop_pose"],
                                 "part_location": None, "part_state": "misplaced", "current_holder_resource_jid": None}}
    reply = asyncio.run(session.agent.request_recovery_outline_physical_validation(
        resource_jid=jid, payload={"candidates": [{"task": task, "physical_input": physical}]}))
    row = reply["results"][0]
    assert row["transition_feasibility"]["allowed"] is True
    assert row["physical_feasibility"]["status"] == "rejected"
    finding = next(row for row in row["findings"] if row["constraint_code"] == "workspace_unreachable")
    assert "0.7998" in finding["reason"]
    assert finding["evidence"]["pose_source"].startswith("failure.observed_drop_pose")
    jid3 = saved["jids"]["ur5e-3"]
    resource3 = session.agent.resources[jid3]
    task3 = deepcopy(task)
    task3.update(resource_jid=jid3)
    task3["expected_end_state"].update(part_location=jid3, resource_location="Buffer For Machined parts")
    result3 = resource3.check_recorded_recovery_feasibility(
        task=task3, recovery_snapshot=resource3.get_recovery_snapshot(), physical_input=physical)
    assert [item["constraint_code"] for item in result3["findings"]] == ["holder_conflict"]
    assert any(item["check"] == "gripper_reach" and item["status"] == "passed" for item in result3["checks"])
    assert path.read_bytes() == source
    assert saved == json.loads(source)
    live.assert_not_called()


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
    resource = offline_recovery.SavedRecoveryResource(jid4, prepared["recovery_resources"][jid4], saved)
    result = resource.check_recorded_recovery_feasibility(
        task=task, recovery_snapshot=resource.get_recovery_snapshot(), physical_input=physical)
    assert any(row["constraint_code"] == "source_reference_unavailable"
               and row["feasibility_status"] == "NEEDS_CONTEXT" for row in result["findings"])
    assert not any(row["constraint_code"] == "workspace_unreachable" for row in result["findings"])
    assert not any(row["check"] == "gripper_reach" and row.get("role") == "source" for row in result["checks"])
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


@pytest.mark.parametrize("finding_location", ["top", "physical"])
@pytest.mark.parametrize("contradictory_allowed", [False, True])
def test_offline_deferral_preserves_recorded_findings(captured, monkeypatch, finding_location, contradictory_allowed):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_generation,
    )

    captured["prepared_recovery_request"]["recovery_session"].update(
        outline_mode="incremental_candidates_validated", recovery_selection_mode="neurosymbolic")
    jid, entry = next(iter(captured["prepared_recovery_request"]["recovery_resources"].items()))
    captured["prepared_recovery_request"]["llm_input"]["observed_runtime_state"] = {
        "resources": [{**entry["recovery_snapshot"], "resource_jid": jid, "resource_state": "failed"}]}

    async def run():
        recovery = offline_recovery.create_offline_recovery(captured, outline_mode="manual", primitive_mode="manual")
        recovery.begin(captured["failure"], generate_outline=False)
        await recovery._task
        request = recovery.agent.request_recovery_outline_physical_validation
        finding = {"validation_category": "physical_feasibility", "constraint_code": "workspace_unreachable",
                   "reason": "Recorded pose is outside the declared bound", "resource_jid": captured["prepared_recovery_request"]["ra_jid"]}

        async def recorded_failure(**kwargs):
            reply = await request(**kwargs)
            for row in reply["results"]:
                row["allowed"] = contradictory_allowed
                target = row if finding_location == "top" else row["physical_feasibility"]
                target["findings"] = [finding]
            return reply

        monkeypatch.setattr(recovery.agent, "request_recovery_outline_physical_validation", recorded_failure)
        seed = deepcopy(recovery._prepared["multi_turn_session_seed"])
        # Replay the historical candidate handler with its recorded policy.
        seed.update(validation_contract_version=None, outline_mode="incremental_candidates_validated",
                    recovery_selection_mode="neurosymbolic", candidate_count=1)
        candidate = {"outline_id": "candidate", "event_name": "move_home", "resource_jid": finding["resource_jid"],
                     "expected_end_state": {"resource_state": "idle"}, "rationale": "Recorded candidate"}
        _, turn = await multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=seed, prepared_recovery_request=recovery._prepared, planner=recovery.agent.process_planner,
            parsed_response={"thought": "Recorded replay", "candidate_events": [candidate]})
        evaluation = turn["candidate_evaluations"][0]
        assert evaluation["valid"] is False
        assert any(row["constraint_code"] == "workspace_unreachable" for row in evaluation["validation_findings"]), evaluation["validation_findings"]
        assert seed["accepted_outline_prefix"] == []
        monkeypatch.setattr(multi_turn_outline_generation, "_symbolically_enabled_recovery_event_instances", lambda **_: [
            {"event_id": "recorded-event", "resource_jid": finding["resource_jid"], "task": candidate,
             "physical_input": {}}])
        enabled = await multi_turn_outline_generation._agent_filtered_recovery_enabledness(
            session_state=seed, prepared_recovery_request=recovery._prepared,
            unresolved_condition_ids=set(), planner=recovery.agent.process_planner)
        assert enabled["admissible_event_ids"] == []
        assert enabled["event_evaluations"]
        assert all(row["ra_status"] == "rejected" for row in enabled["event_evaluations"])
        assert all("workspace_unreachable" in row["constraint_codes"] for row in enabled["event_evaluations"])
    asyncio.run(run())


@pytest.mark.parametrize("missing_place_support", [False, True])
def test_saved_repick_progress_requires_supported_continuation(captured, monkeypatch, missing_place_support):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_generation,
    )
    from cais_spade_llm.resources.robot.robot_task_recovery import robot_recovery_des_descriptor

    prepared = captured["prepared_recovery_request"]
    jid, entry = next(iter(prepared["recovery_resources"].items()))
    entry["recovery_snapshot"]["current_location"] = "3D Printing Station"
    model = robot_recovery_des_descriptor(
        resource_jid=jid, snapshot=entry["recovery_snapshot"],
        reachable_locations=["3D Printing Station", "assembly_board-v1"], named_poses=["home"],
    )
    catalog = [{"name": name} for name in sorted({
        step["primitive"] for event in model["events"] for step in event["primitive_support"]
        if not (missing_place_support and step["primitive"] == "compute_place_targets")
    })]
    model["primitive_catalog"] = catalog
    model["descriptor_fingerprint"] = recovery_validation_fingerprint(model)
    entry.update(recovery_des_model=model, recovery_des_model_fingerprint=model["descriptor_fingerprint"],
                 execution_primitive_catalog=catalog, primitive_catalog=catalog,
                 static_capabilities={"resource_type": "robot", "reachability": ["3D Printing Station", "assembly_board-v1"],
                                      "gripper_reach": {"frame": "world", "origin_pose": {"x": 0., "y": 0., "z": 1.},
                                                        "max_xy_radius_m": .7, "z_min_m": .85, "z_max_m": 1.6,
                                                        "tolerance_m": .01}})
    pose = deepcopy(captured["part_tracker"]["gear_small"]["observed_pose"])
    part = {"part_name": "gear_small", "current_state": "misplaced", "current_location": None,
            "current_holder_resource_jid": None, "observed_pose": pose, "observed_pose_frame": "world",
            "origin_location": "3D Printing Station", "goal_location": "assembly_board-v1"}
    prepared["llm_input"] = {
        "observed_runtime_state": {"resources": [{"resource_jid": jid, **entry["recovery_snapshot"]}]},
        "part_facts": [part],
        "goal_conditions": [{"entity_kind": "part", "entity": "gear_small", "field": field,
                             "expected": value} for field, value in (("state", "assembled"), ("location", "assembly_board-v1"))],
    }
    prepared["grounding_context"] = {"parts": {"gear_small": part}}
    prepared["recovery_session"].update(outline_mode="incremental_candidates_validated",
                                         recovery_selection_mode="neurosymbolic")
    candidate = {"outline_id": "candidate-3", "event_name": "recover_pick_gear_small",
                 "resource_jid": jid, "part_name": "gear_small", "rationale": "Acquire observed part custody.",
                 "expected_end_state": {"resource_state": "picked", "held_part": "gear_small",
                                        "resource_location": "3D Printing Station", "part_state": "in_gripper",
                                        "part_location": jid}}
    live = AsyncMock(side_effect=AssertionError("Offline candidate validation must not call live interfaces"))
    monkeypatch.setattr(offline_recovery.SavedRecoveryProduct, "ask_llm_structured", live)
    monkeypatch.setattr("cais_spade_llm.recovery_framework.failure_effects.observe_robot", live)
    monkeypatch.setattr("spade.agent.Agent.start", live)
    original = deepcopy(captured)

    async def run():
        session = offline_recovery.create_offline_recovery(captured, outline_mode="manual", primitive_mode="manual")
        session.begin(captured["failure"], generate_outline=False)
        await session._task
        state = deepcopy(session._prepared["multi_turn_session_seed"])
        # This fixture checks the retained historical candidate-selection path.
        state.update(validation_contract_version=None, outline_mode="incremental_candidates_validated",
                     recovery_selection_mode="neurosymbolic", candidate_count=1)
        decision, turn = await multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=state, prepared_recovery_request=session._prepared,
            planner=session.agent.process_planner,
            parsed_response={"thought": "Recorded candidate replay.", "candidate_events": [candidate]},
        )
        evaluation = turn["candidate_evaluations"][0]
        assert evaluation["valid"] is True
        assert evaluation["selection_evidence"]["cleared_recovery_obligation_ids"] == []
        after = evaluation["recovery_enabledness_validation_after"]
        continuation = next(row for row in after["event_evaluations"]
                            if json.loads(row["event_id"])["event_name"] == "place_approach")
        if missing_place_support:
            assert continuation["constraint_codes"] == ["resource_validation_unavailable"]
            assert "NEEDS_CONTEXT" in continuation["findings"][0]["reason"]
            assert evaluation["selection_status"] == "excluded_no_progress"
            assert state["accepted_outline_prefix"] == []
        else:
            assert decision == "need_next_task"
            assert evaluation["selection_evidence"]["newly_enabled_recovery_event_ids"] == [continuation["event_id"]]
            assert continuation["ra_status"] == "deferred"
            assert continuation["cca_status"] == "skipped"
            assert continuation["execution_authorized"] is False
            assert state["accepted_outline_prefix"][0]["event_name"] == "recover_pick_gear_small"
            assert state["symbolic_parts"]["gear_small"]["observed_pose"] is None
            assert state["symbolic_parts"]["gear_small"]["observed_pose_current"] is False
            assert state["symbolic_parts"]["gear_small"]["last_observed_pose"] == pose
            release = {"outline_id": "candidate-release", "event_name": "recover_release",
                       "resource_jid": jid, "part_name": "gear_small", "rationale": "Recorded continuation replay.",
                       "expected_end_state": {"resource_state": "placed", "held_part": None,
                                              "resource_location": "assembly_board-v1", "part_state": "assembled",
                                              "part_location": "assembly_board-v1"}}
            _, release_turn = await multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
                session_state=state, prepared_recovery_request=session._prepared,
                planner=session.agent.process_planner,
                parsed_response={"thought": "Recorded candidate replay.", "candidate_events": [release]},
            )
            assert len(state["accepted_outline_prefix"]) == 2, [
                row.get("validation_findings") for row in release_turn["candidate_evaluations"]]
            assert state["symbolic_parts"]["gear_small"]["part_location"] == "assembly_board-v1"
            from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_validation_service import (
                build_recovery_physical_validation_input,
                compile_grounded_recovery_outline_task,
                projected_outline_validation_context,
            )
            state["observation_store"]["saved_drop"] = {"part_name": "gear_small", "pose": deepcopy(pose)}
            resources, parts = projected_outline_validation_context(
                session_state=state, prepared_recovery_request=session._prepared)
            reacquire = deepcopy(candidate)
            reacquire["expected_start_state"] = deepcopy(release["expected_end_state"])
            compiled = compile_grounded_recovery_outline_task(
                reacquire, resources_by_jid=resources, parts_by_name=parts, validate_binding_tokens=False)
            assert compiled["grounded_action"] is not None, compiled
            physical = build_recovery_physical_validation_input(
                task=reacquire, grounded_action=compiled["grounded_action"],
                session_state=state, prepared_recovery_request=session._prepared)
            reply = await session.agent.request_recovery_outline_physical_validation(
                resource_jid=jid, payload={"candidates": [{"task": reacquire, "physical_input": physical}]})
            findings = reply["results"][0]["findings"]
            assert any(row["constraint_code"] == "source_reference_unavailable"
                       and row["feasibility_status"] == "NEEDS_CONTEXT" for row in findings), findings
            assert not any(row["constraint_code"] == "workspace_unreachable" for row in findings)
        assert session.snapshot()["execution"] == []
    asyncio.run(run())
    live.assert_not_called()
    assert captured == original


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


@pytest.mark.parametrize("outline_mode", ["auto", "manual"])
@pytest.mark.parametrize("primitive_mode", ["auto", "manual"])
def test_offline_stages_never_access_live_observations(captured, monkeypatch, outline_mode, primitive_mode):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.recovery_framework import failure_effects

    observe = AsyncMock(side_effect=AssertionError("offline test queried Gazebo"))
    monkeypatch.setattr(failure_effects, "observe_robot", observe)
    original = deepcopy(captured)
    session = offline_recovery.create_offline_recovery(captured, outline_mode=outline_mode, primitive_mode=primitive_mode)
    async def generate(planner, prepared, *, session_state):
        assert prepared["offline_recovery"] is True
        assert prepared["recovery_resources"] == original["prepared_recovery_request"]["recovery_resources"]
        state = deepcopy(session_state or {})
        if prepared["_stop_after_multi_turn_phase"] == "outline":
            state.update(current_phase="primitive_generation", accepted_outline_prefix=[{"outline_id": "RECOVERY_SEQ1",
                "resource_jid": prepared["ra_jid"], "event_name": "move_home", "part_name": None,
                "expected_start_state": {"resource_state": "failed"}, "expected_end_state": {"resource_state": "idle"}}])
        else:
            state.update(final_output={"primitive_program_complete": True}, accepted_primitive_program=[{"outline_id": "RECOVERY_SEQ1"}])
        prepared["multi_turn_session_state"] = state
    monkeypatch.setattr(multi_turn, "execute_multi_turn_recovery", generate)
    monkeypatch.setattr(multi_turn, "build_multi_turn_recovery_proposal", lambda **kwargs: {
        "accepted": True, "recovery_proposal": {"macro_tasks": [{"outline_id": "RECOVERY_SEQ1"}]}})
    async def run():
        session.begin(captured["failure"])
        await session._task
        assert session.snapshot()["outline_mode"] == session.snapshot()["primitive_mode"] == "manual"
        assert session.runtime.setup["runtime_recovery_mode"] == session.runtime.setup["runtime_recovery_primitive_mode"] == "manual"
        assert session.snapshot()["status"] == "outline_ready"
        session.action("generate_outline")
        await session._task
        assert session.snapshot()["status"] == "outline_review"
        session.action("continue_outline")
        await session._task
        assert session.snapshot()["status"] == "primitive_ready"
        session.action("generate_primitives")
        await session._task
        assert session.snapshot()["status"] == "offline_completed"
        with pytest.raises(ValueError, match="runtime CCA integration is pending"):
            session.action("execute")
        with pytest.raises(ValueError, match="runtime CCA integration is pending"):
            await session._execute()
        saved = json.loads((session.path / "observations.json").read_text())
        assert saved == original
        assert session._proposal["source_observation_fingerprint"] == observation_fingerprint(original)
        previous = session.snapshot()["artifact_directory"]
        session.action("regenerate_primitives", feedback="")
        await session._task
        assert session.snapshot()["status"] == "offline_completed"
        directory = Path(session.snapshot()["artifact_directory"])
        assert str(directory) != previous
        assert (directory / "outline_request.json").is_file()
        assert json.loads((directory / "primitive_input_preview.json").read_text())["request_sent"] is False
        session.stop()
        assert session.snapshot()["status"] == "offline_completed"
    asyncio.run(run())
    observe.assert_not_called()
    assert captured == original


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
    directory = tmp_path / "saved" / "old"
    directory.mkdir(parents=True)
    path = directory / "environment_run.json"
    write_record(path, {"run_id": "old", "conveyor_fault": {"scenario": "Part slippage", "evidence": {"observed_drop_pose": {"x": 1}}},
                        "final_product_states": {"gear_small": {"state": "misplaced"}}, "final_valuation": {}, "inputs": {}})
    record = read_observations(path, root=directory.parent)
    assert list_observations(root=directory.parent) == []
    assert record["part_tracker"]["gear_small"]["state"] == "misplaced"
    assert record["captured_at"] is None
    with pytest.raises(ValueError, match="NEEDS_CONTEXT"):
        offline_recovery.create_offline_recovery(record, outline_mode="auto", primitive_mode="auto")


def test_offline_instances_have_separate_output_directories(captured):
    first = offline_recovery.create_offline_recovery(captured, outline_mode="manual", primitive_mode="manual")
    second = offline_recovery.create_offline_recovery(captured, outline_mode="manual", primitive_mode="manual")
    assert first.path != second.path
    assert first.snapshot()["run_id"] != second.snapshot()["run_id"]


@pytest.mark.parametrize("select_original", [False, True])
def test_saved_result_reloads_after_restart_without_generation(captured, tmp_path, monkeypatch, select_original):
    from cais_spade_llm.recovery_framework import part_slippage_example

    root = tmp_path / "recovery"
    source = root / "old_offline_run"
    directory = source / "generation_0001"
    directory.mkdir(parents=True)
    (directory / "recovery_outline").mkdir()
    raw_request = '{"prompt": "captured exactly", "request_sent": true}\n'
    (directory / "recovery_outline/request.json").write_text(raw_request)
    observation_path = source / "observations.json"
    write_record(observation_path, captured)
    prepared = deepcopy(captured["prepared_recovery_request"])
    prepared["multi_turn_session_state"] = {"current_phase": "primitive_generation"}
    write_record(directory / "outline_request.json", prepared)
    proposal = {"macro_tasks": [{"outline_id": "RECOVERY_SEQ1"}], "offline_recovery": True,
                "source_observation_fingerprint": observation_fingerprint(captured)}
    write_record(directory / "proposal.json", proposal)
    write_record(source / "run.json", {"status": "offline_completed", "artifact_directory": str(directory),
                                       "outline_mode": "auto", "primitive_mode": "auto",
                                       "outline": [{"outline_id": "RECOVERY_SEQ1"}], "primitives": []})
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda p: read_observations(p, root=root))
    monkeypatch.setattr(part_slippage_example, "list_observations", lambda: list_observations(root=root))
    if select_original:
        observation_path = root / captured["run_id"] / "observations.json"
        observation_path.parent.mkdir()
        write_record(observation_path, captured)
        records = list_observations(root=root)
        assert len(records) == 1
        assert records[0]["source_path"] == str(observation_path)
        assert records[0]["result_available"] is True
        assert records[0]["result_observation_path"] == str(source / "observations.json")
    llm = AsyncMock(side_effect=AssertionError("loading a result started LLM generation"))
    monkeypatch.setattr(offline_recovery.SavedRecoveryProduct, "ask_llm_structured", llm)
    bridge = SimpleNamespace(system_running=False, ros2_start=Mock(), start_system=AsyncMock())
    example = part_slippage_example.example_for_bridge(bridge)
    example.set_options(source="offline", scenario="Part slippage", safety_files=[], observation_path=str(observation_path))
    state = example.load_offline_result()
    assert state["status"] == "offline_completed"
    assert state["outline_mode"] == state["primitive_mode"] == "manual"
    assert json.loads((source / "run.json").read_text())["outline_mode"] == "auto"
    assert state["run_id"] != "old_offline_run"
    assert example._offline._proposal == proposal
    assert observation_fingerprint(example._offline._observations) == observation_fingerprint(captured)
    assert example._offline.path != source
    assert (Path(state["outline_artifact_directory"]) / "request.json").read_text() == raw_request
    assert part_slippage_example.example_for_bridge(bridge) is example
    llm.assert_not_called()
    bridge.ros2_start.assert_not_called()
    bridge.start_system.assert_not_called()
    assert json.loads((directory / "proposal.json").read_text()) == proposal


def test_offline_duplicate_stop_reset_preserve_capture(captured, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    root = tmp_path / "recovery"
    source = root / "source"
    source.mkdir(parents=True)
    path = source / "observations.json"
    write_record(path, captured)
    original_bytes = path.read_bytes()
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda p: read_observations(p, root=root))
    started = asyncio.Event()
    async def pending(self, stage):
        started.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(PartSlippageRecovery, "_run_stage", pending)
    bridge = SimpleNamespace(system_running=False, stop_system=AsyncMock())
    example = part_slippage_example.PartSlippageExample(bridge)
    example.set_options(source="offline", scenario="Part slippage", safety_files=[], observation_path=str(path))
    async def run():
        example.start_offline()
        with pytest.raises(ValueError):
            example.start_offline()
        await example._offline._task
        assert not started.is_set()
        assert example.snapshot()["status"] == "outline_ready"
        await example.action("generate_outline")
        await asyncio.wait_for(started.wait(), timeout=3)
        session = example._offline
        await example.action("stop")
        await asyncio.gather(session._task, return_exceptions=True)
        assert example.snapshot()["status"] == "stopped"
        await example.action("reset")
        await example._offline._task
        assert example.snapshot()["status"] == "outline_ready"
        assert example._offline.path != session.path
        assert (session.path / "observations.json").is_file()
    asyncio.run(run())
    assert path.read_bytes() == original_bytes
    bridge.stop_system.assert_not_called()


def test_real_outline_validation_records_physical_deferral(captured, monkeypatch):
    from test_case3_recovery_dryrun import (
        _fixture_outline_responses,
        _load_case3_runtime_context,
        _prepare_recovery_dryrun_harness,
    )

    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn,
        multi_turn_outline_generation,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_session import (
        RecoverySessionMixin,
    )

    monkeypatch.setattr(RecoverySessionMixin, "_recovery_loaded_safety_rules", lambda self: [])
    _, _, _, prepared = asyncio.run(_prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses()))
    prepared["loaded_safety_rules"] = []
    prepared["llm_input"]["loaded_safety_rules"] = []
    captured["prepared_recovery_request"] = prepared
    context = _load_case3_runtime_context()
    captured["part_tracker"] = deepcopy(context["part_tracker"])
    for snapshot in context["resource_snapshots"]:
        for name, observation in snapshot.get("observations", {}).items():
            captured["part_tracker"][name]["observed_pose"] = deepcopy(observation["pose"])
    session = offline_recovery.create_offline_recovery(captured, outline_mode="manual", primitive_mode="manual")
    session._install_prepared(deepcopy(prepared), session._new_artifact_directory(), "")
    state = deepcopy(session._prepared["multi_turn_session_seed"])
    seeded = next(row for row in state["observation_fact_ledger"].values() if row["entity"] == "KET4_Square_4mm")
    assert seeded["output"]["pose"] == captured["part_tracker"]["KET4_Square_4mm"]["observed_pose"]
    assert seeded["freshness"] == "saved_observations"
    # Exercise the saved observation adapter as well as the seeded facts.
    state["observation_fact_ledger"] = {}
    observed, error = asyncio.run(multi_turn._execute_observe_requests(
        session.agent.process_planner, session._prepared, state, _fixture_outline_responses()[0]["observe_requests"]))
    assert error is None and observed
    assert observed[0]["output"]["pose"] == captured["part_tracker"]["KET4_Square_4mm"]["observed_pose"]
    assert observed[0]["output"]["captured_at"] == captured["captured_at"]
    multi_turn._record_observation_fact(state, observed[0])
    assert state["observation_fact_ledger"][observed[0]["fact_key"]]["freshness"] == "saved_observations"
    candidate = deepcopy(_fixture_outline_responses()[2]["candidate_events"][0])
    evaluation = asyncio.run(multi_turn_outline_generation._validate_candidate_sequence(
        candidate={"candidate_index": 0, "surface_events": [candidate]}, sequence_index=1,
        action_horizon="1", action_horizon_k=1, session_state=state,
        prepared_recovery_request=session._prepared, planner=session.agent.process_planner))
    assert evaluation["valid"] is True, evaluation
    stages = {row["validation_category"]: row["status"] for row in evaluation["validation_stages"]}
    assert stages["physical_feasibility"] == "deferred"
    assert stages["safety"] == "skipped"
    assert evaluation["committed_events"][0]["execution_authorized"] is False


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


@pytest.fixture
def scripted_generation(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    calls = []
    async def generate(planner, prepared, *, session_state):
        stage = prepared["_stop_after_multi_turn_phase"]
        calls.append(stage)
        state = deepcopy(session_state or {})
        if stage == "outline":
            state.update(current_phase="primitive_generation", accepted_outline_prefix=[{"outline_id": "RECOVERY_SEQ1",
                "resource_jid": prepared["ra_jid"], "event_name": "move_home", "part_name": None,
                "expected_start_state": {"resource_state": "failed"}, "expected_end_state": {"resource_state": "idle"}}])
        else:
            state.update(final_output={"primitive_program_complete": True}, accepted_primitive_program=[{"outline_id": "RECOVERY_SEQ1"}])
        prepared["multi_turn_session_state"] = state
    monkeypatch.setattr(multi_turn, "execute_multi_turn_recovery", generate)
    monkeypatch.setattr(multi_turn, "build_multi_turn_recovery_proposal", lambda **kwargs: {
        "accepted": True, "recovery_proposal": {"macro_tasks": [{"outline_id": "RECOVERY_SEQ1"}]}})
    return calls


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_direct_generate_prepares_without_llm_and_respects_shared_mode(offline_example, scripted_generation, mode):
    from cais_spade_llm.recovery_framework.part_slippage_example import example_for_bridge

    example, path, bridge = offline_example
    original = path.read_bytes()
    async def run():
        state = example.prepare_offline()
        assert state["busy"] and "Preparing" in state["outline_generate_reason"]
        assert not state["can_generate_outline"]
        session = example._offline
        example.prepare_offline()
        assert example._offline is session
        await session._task
        assert scripted_generation == []
        assert example.snapshot()["can_generate_outline"]
        preview = json.loads((Path(session.snapshot()["artifact_directory"]) / "input_preview.json").read_text())
        assert preview["request_sent"] is False
        if mode == "auto":
            with pytest.raises(ValueError, match="Offline recovery always uses Manual"):
                example.set_modes(mode, mode)
        else:
            example.set_modes(mode, mode)
        assert session.snapshot()["outline_mode"] == session.snapshot()["primitive_mode"] == "manual"
        assert example_for_bridge(bridge) is example
        await example.action("generate_outline")
        with pytest.raises(ValueError, match="already running"):
            await example.action("generate_outline")
        assert "already running" in example.snapshot()["outline_generate_reason"]
        await session._task
        assert session.snapshot()["status"] == "outline_review"
        assert scripted_generation == ["outline"]
        await example.action("continue_outline")
        await session._task
        assert session.snapshot()["status"] == "primitive_ready"
        await example.action("generate_primitives")
        await session._task
        assert session.snapshot()["status"] == "offline_completed"
        assert len(scripted_generation) == 2
        assert session.snapshot()["execution"] == []
        assert session.snapshot()["safety_status"] == "skipped"
    asyncio.run(run())
    assert path.read_bytes() == original
    bridge.ros2_start.assert_not_called()
    bridge.start_system.assert_not_called()
    bridge.stop_system.assert_not_called()


def test_stop_then_generate_creates_a_fresh_attempt(offline_example, monkeypatch, scripted_generation):
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    example, path, bridge = offline_example
    original = path.read_bytes()
    started = asyncio.Event()
    stage_runner = PartSlippageRecovery._run_stage
    async def pending(self, stage):
        started.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(PartSlippageRecovery, "_run_stage", pending)
    async def run():
        example.prepare_offline()
        first = example._offline
        assert first.runtime.source_observation_path == str(path)
        await first._task
        await example.action("generate_outline")
        await started.wait()
        await example.action("stop")
        await asyncio.gather(first._task, return_exceptions=True)
        assert example.snapshot()["can_generate_outline"]
        example.set_modes("manual", "manual")
        assert first.snapshot()["outline_mode"] == "manual"
        assert example.snapshot()["outline_mode"] == "manual"
        monkeypatch.setattr(PartSlippageRecovery, "_run_stage", stage_runner)
        await example.action("generate_outline")
        second = example._offline
        await second._task
        assert second.path != first.path
        assert second.snapshot()["status"] == "outline_review"
        assert first.snapshot()["status"] == "stopped"
        assert (first.path / "observations.json").is_file()
    asyncio.run(run())
    assert path.read_bytes() == original
    bridge.stop_system.assert_not_called()


@pytest.mark.parametrize("stop_during_drain", [False, True])
def test_changed_capture_cancels_obsolete_preparation_and_rejects_late_results(
        offline_example, captured, monkeypatch, stop_during_drain):
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    example, path, _ = offline_example
    second_path = path.parent.parent / "second" / "observations.json"
    second_path.parent.mkdir()
    second_record = {**deepcopy(captured), "run_id": "second"}
    write_record(second_path, second_record)
    waiting, released = asyncio.Event(), asyncio.Event()
    original_prepare = PartSlippageRecovery._prepare
    async def delayed(self, feedback=""):
        if self._observations["run_id"] == "captured-run":
            waiting.set()
            try:
                await released.wait()
            except asyncio.CancelledError:
                await released.wait()
        await original_prepare(self, feedback)
    monkeypatch.setattr(PartSlippageRecovery, "_prepare", delayed)
    llm = AsyncMock(side_effect=AssertionError("preparation called the LLM"))
    monkeypatch.setattr(offline_recovery.SavedRecoveryProduct, "ask_llm_structured", llm)
    async def run():
        example.prepare_offline()
        first = example._offline
        await waiting.wait()
        example.set_options(source="offline", scenario="Part slippage", safety_files=[], observation_path=str(second_path))
        example.prepare_offline()
        assert example._offline is None and example.snapshot()["busy"]
        assert not example.snapshot()["can_generate_outline"]
        if stop_during_drain:
            await example.action("stop")
        released.set()
        await example._task
        await asyncio.gather(first._task, return_exceptions=True)
        if stop_during_drain:
            assert example._offline is None
            example.prepare_offline()
        second = example._offline
        assert second is not None
        await second._task
        assert example._offline is second
        assert example.snapshot()["observations"]["run_id"] == "second"
        assert example.snapshot()["can_generate_outline"]
        assert first._prepared == {}
        assert second.snapshot()["status"] == "outline_ready"
    asyncio.run(run())
    llm.assert_not_called()


def test_preparation_failure_retains_capture_and_retry_does_not_generate(offline_example, monkeypatch):
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    example, path, _ = offline_example
    original = path.read_bytes()
    prepare = PartSlippageRecovery._prepare
    monkeypatch.setattr(PartSlippageRecovery, "_prepare", AsyncMock(side_effect=ValueError("NEEDS_CONTEXT: missing saved geometry")))
    async def run():
        example.prepare_offline()
        first = example._offline
        await first._task
        state = example.snapshot()
        assert state["status"] == "blocked"
        assert not state["can_generate_outline"]
        assert state["outline_generate_reason"] == "ValueError: NEEDS_CONTEXT: missing saved geometry"
        assert Path(state["observation_path"]).is_file()
        monkeypatch.setattr(PartSlippageRecovery, "_prepare", prepare)
        await example.action("prepare")
        await example._offline._task
        assert example._offline.path != first.path
        assert example.snapshot()["status"] == "outline_ready"
    asyncio.run(run())
    assert path.read_bytes() == original


@pytest.mark.parametrize("pose_source", ["part_tracker", "grounding_context"])
def test_twelve_saved_resources_and_null_prepared_pose_remain_retrievable(captured, monkeypatch, pose_source):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_context,
    )
    from cais_spade_llm.recovery_framework import failure_effects

    prepared = captured["prepared_recovery_request"]
    template = prepared["recovery_resources"][prepared["ra_jid"]]
    for number in range(1, 13):
        jid = f"recovery-resource-{number}@localhost"
        entry = deepcopy(template)
        entry.update(resource_jid=jid, resource_type="robot" if number <= 4 else "kmr" if number == 8 else "resource")
        if number > 4:
            entry.update(recovery_des_model=None, recovery_des_model_fingerprint=None,
                         execution_primitive_catalog=[], recovery_adapter={"supports_executable_recovery": number == 8})
        prepared["recovery_resources"][jid] = entry
    pose = deepcopy(captured["part_tracker"]["gear_small"]["observed_pose"])
    captured["failure"].update(observed_drop_pose=pose, requested_drop_pose={"x": 0, "y": 0.3, "z": 1.04})
    prepared["llm_input"]["part_facts"] = [{"part_name": "gear_small", "observed_pose": None}]
    prepared["part_tracker"] = {"gear_small": {"observed_pose": None}}
    prepared["grounding_context"] = {"parts": {"gear_small": {"observed_pose": pose}}}
    if pose_source == "grounding_context":
        captured["part_tracker"]["gear_small"]["observed_pose"] = None
    else:
        prepared["grounding_context"]["parts"]["gear_small"]["observed_pose"] = None
    original = deepcopy(captured)
    live = AsyncMock(side_effect=AssertionError("offline retrieval contacted a live interface"))
    monkeypatch.setattr(failure_effects, "observe_robot", live)
    monkeypatch.setattr("spade.agent.Agent.start", live)
    monkeypatch.setattr(offline_recovery.SavedRecoveryProduct, "ask_llm_structured", live)
    async def run():
        session = offline_recovery.create_offline_recovery(captured, outline_mode="auto", primitive_mode="auto")
        session.begin(captured["failure"], generate_outline=False)
        await session._task
        assert session.snapshot()["status"] == "outline_ready"
        assert len(session.agent.resources) == 12
        state = session._prepared["multi_turn_session_seed"]
        resolved = multi_turn_context.resolve_outline_context(session._prepared, state, "/parts/gear_small/observed_pose")
        assert resolved["value"] == pose and resolved["source"] == "saved_observations"
        assert resolved["captured_at"] == captured["captured_at"]
        assert len(multi_turn_context.resolve_outline_context(session._prepared, state, "/resources")["value"]) == 12
        for number in (5, 8):
            with pytest.raises(ValueError, match="NEEDS_CONTEXT.*recovery_des_model"):
                await session.agent.request_recovery_outline_physical_validation(
                    resource_jid=f"recovery-resource-{number}@localhost", payload={"candidates": [{"task": {}}]})
        assert json.loads((session.path / "observations.json").read_text()) == original
    asyncio.run(run())
    live.assert_not_called()
    assert captured == original


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


@pytest.mark.parametrize("stage", ["outline", "primitive"])
def test_api_failure_marks_stage_failed_and_keeps_error_across_retry(offline_example, scripted_generation, monkeypatch, stage):
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    example, _, _ = offline_example
    error = "LLM call failed: BadRequestError; status_code=400; Invalid schema for response_format"
    runner = PartSlippageRecovery._run_stage
    async def fail(self, requested_stage):
        if requested_stage == stage:
            raise RuntimeError(error)
        return await runner(self, requested_stage)
    monkeypatch.setattr(PartSlippageRecovery, "_run_stage", fail)
    async def run():
        example.prepare_offline()
        session = example._offline
        await session._task
        await example.action("generate_outline")
        await session._task
        if stage == "primitive":
            await example.action("continue_outline")
            await session._task
            await example.action("generate_primitives")
            await session._task
        state = example.snapshot()
        assert state["status"] == "blocked" and state[f"{stage}_status"] == "failed"
        assert state["message"] == "RuntimeError: " + error
        record = Path(state["artifact_directory"]) / "stage_error.json"
        before = record.read_bytes()
        assert json.loads(before)["error"] == "RuntimeError: " + error
        monkeypatch.setattr(PartSlippageRecovery, "_run_stage", runner)
        action = "regenerate_outline" if stage == "outline" else "regenerate_primitives"
        await example.action(action)
        await session._task
        state = example.snapshot()
        assert Path(state["artifact_directory"]) != record.parent
        assert record.read_bytes() == before
        assert state["status"] == ("outline_review" if stage == "outline" else "offline_completed")
        assert state["outline_mode"] == state["primitive_mode"] == "manual"
        assert state["execution"] == []
    asyncio.run(run())


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


def _saved_outline_review(captured, tmp_path, monkeypatch, *, proof_changes=None):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import recovery_artifacts
    from cais_spade_llm.recovery_framework import part_slippage_example

    root = tmp_path / "recovery"
    source = root / "saved_outline"
    directory = source / "generation_0001"
    directory.mkdir(parents=True)
    observations = deepcopy(captured)
    observations["pending_nodes"] = []
    write_record(source / "observations.json", observations)
    prepared = deepcopy(observations["prepared_recovery_request"])
    prepared.update(validation_contract_version=2, primitive_support_semantics="successor_primitive_union",
        nominal_continuation_artifact_version=1,
        source_observation_fingerprint=observation_fingerprint(observations), saved_pending_nodes=[],
        recovery_debug={"artifact_directory": str(directory), "per_turn_debug_dir": str(directory)})
    accepted = [{"outline_id": "proposed_event", "candidate_source": "llm", "part_name": None,
        "resource_jid": prepared["ra_jid"], "event_name": "move_home",
        "expected_start_state": {"resource_state": "failed"}, "expected_end_state": {"resource_state": "idle"}}]
    proof = {"complete": True, "status": "passed", "execution_authorized": False,
        "tasks": [], "replaced_task_ids": [], "replacement_evidence": {}}
    proof.update(proof_changes or {})
    session = {"validation_contract_version": 2, "current_phase": "primitive_generation",
        "status": "ready_for_primitive_generation", "turn_index": 2,
        "accepted_outline_prefix": accepted, "symbolic_resources": {}, "symbolic_parts": {},
        "projected_physical_evidence": {"snapshot": {"resources": {}, "parts": {}}},
        "nominal_continuation_evidence": deepcopy(proof)}
    prepared["multi_turn_session_state"] = session
    turn = {"phase": "outline", "turn_index": 2, "nominal_continuation_evidence": deepcopy(proof)}
    archive = recovery_artifacts.archive_nominal_continuation(prepared, session, turn, decision="outline_ready")
    assert archive is not None
    write_record(directory / "outline_request.json", prepared)
    write_record(directory / "outline.json", {"outline": accepted})
    provider = '{"request_sent":true,"messages":[{"role":"user","content":"Recorded facts"}]}\n'
    (directory / "recovery_outline/request.json").write_text(provider)
    write_record(source / "run.json", {"status": "outline_review", "outline_status": "completed",
        "primitive_status": "waiting", "validation_contract_version": 2,
        "artifact_directory": str(directory), "outline_artifact_directory": str(directory / "recovery_outline"),
        "selected_safety_files": [], "outline_mode": "auto", "primitive_mode": "auto",
        "outline": accepted, "primitives": [], "execution_authorized": False})
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda path: read_observations(path, root=root))
    monkeypatch.setattr(part_slippage_example, "list_observations", lambda: list_observations(root=root))
    llm = AsyncMock(side_effect=AssertionError("Saved result restoration called the provider"))
    monkeypatch.setattr(offline_recovery.SavedRecoveryProduct, "ask_llm_structured", llm)
    bridge = SimpleNamespace(system_running=False, ros2_start=Mock(), start_system=AsyncMock())
    example = part_slippage_example.PartSlippageExample(bridge)
    example.set_options(source="offline", scenario="Part slippage", safety_files=[],
        observation_path=str(source / "observations.json"))
    return example, source, directory, prepared, archive, llm


@pytest.mark.parametrize("select_original", [False, True])
@pytest.mark.parametrize("preparation_status", ["idle", "blocked"])
def test_saved_outline_review_loads_bound_proof_without_composition(captured, tmp_path, monkeypatch, select_original, preparation_status):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import recovery_artifacts
    from cais_spade_llm.agents.shared_information.llm_request_records import recovery_request_directory

    example, source, directory, prepared, archive, llm = _saved_outline_review(captured, tmp_path, monkeypatch)
    example._status.update(status=preparation_status)
    originals = {path: path.read_bytes() for path in source.rglob("*.json")}
    if select_original:
        original = source.parent / captured["run_id"] / "observations.json"
        original.parent.mkdir()
        original.write_bytes((source / "observations.json").read_bytes())
        example.observation_path = str(original)
        index = list_observations(root=source.parent)
        assert len(index) == 1 and index[0]["result_available"] is True
        assert index[0]["source_path"] == str(original)
        assert index[0]["result_observation_path"] == str(source / "observations.json")
    result = (asyncio.run(asyncio.to_thread(example.load_offline_result))
              if preparation_status == "blocked" else example.load_offline_result())
    restored = example._offline
    copied = Path(result["artifact_directory"])
    assert restored.runtime.source_observation_path == example.observation_path
    assert result["status"] == "outline_review"
    assert result["outline_status"] == "completed" and result["primitive_status"] == "waiting"
    assert result["outline_mode"] == result["primitive_mode"] == "manual"
    assert result["execution_authorized"] is False and result["primitives"] == []
    assert restored._proposal == {} and not (copied / "proposal.json").exists()
    assert result["outline"] == prepared["multi_turn_session_state"]["accepted_outline_prefix"]
    for request in (restored._prepared, restored._outline_request):
        assert recovery_request_directory(request, "outline") == copied / "recovery_outline"
        assert recovery_request_directory(request, "primitive_generation") == copied / "recovery_primitves"
        assert request["multi_turn_session_state"]["accepted_outline_prefix"][0]["part_name"] is None
    nominal = restored._prepared["multi_turn_session_state"]["nominal_continuation_evidence"]
    binding = recovery_artifacts.nominal_continuation_context_binding(
        restored._prepared["multi_turn_session_state"], restored._prepared)
    proof = recovery_artifacts.read_nominal_continuation(nominal,
        directory=copied / "recovery_outline", context_binding=binding)
    assert proof["complete"] is True and proof["execution_authorized"] is False
    assert (copied / "recovery_outline" / archive.relative_to(directory / "recovery_outline")).read_bytes() == originals[archive]
    assert (copied / "outline_request.json").read_bytes() == originals[directory / "outline_request.json"]
    assert (copied / "recovery_outline/request.json").read_bytes() == originals[directory / "recovery_outline/request.json"]
    assert {path: path.read_bytes() for path in originals} == originals
    llm.assert_not_called()
    example.bridge.ros2_start.assert_not_called()
    example.bridge.start_system.assert_not_called()
    with pytest.raises(ValueError, match="does not authorize execution"):
        restored.action("execute")


@pytest.mark.parametrize("changed", ["source", "prefix", "outline", "run_prefix", "phase", "status",
    "candidate_source", "missing_tasks", "task_index", "missing_reference", "tampered_proof", "missing_proof",
    "outside_path", "typed_prefix", "pending_capture", "session_type"])
def test_saved_outline_review_rejects_unbound_or_incomplete_records(captured, tmp_path, monkeypatch, changed):
    example, source, directory, prepared, archive, llm = _saved_outline_review(captured, tmp_path, monkeypatch)
    session = prepared["multi_turn_session_state"]
    if changed == "source":
        prepared["source_observation_fingerprint"] = "b" * 64
    elif changed == "session_type":
        prepared["multi_turn_session_state"] = None
    elif changed == "prefix":
        session["accepted_outline_prefix"][0]["outline_id"] = "another_event"
    elif changed == "outline":
        write_record(directory / "outline.json", {"outline": []})
    elif changed == "run_prefix":
        state = json.loads((source / "run.json").read_text())
        state["outline"] = []
        write_record(source / "run.json", state)
    elif changed == "phase":
        session["current_phase"] = "outline"
    elif changed == "status":
        session["status"] = "NEEDS_CONTEXT"
    elif changed == "candidate_source":
        session["accepted_outline_prefix"][0]["candidate_source"] = "supplied"
    elif changed == "missing_tasks":
        prepared.pop("saved_pending_nodes")
    elif changed == "task_index":
        prepared["saved_pending_nodes"] = [{"id": "unrecorded_task"}]
    elif changed == "pending_capture":
        observations = json.loads((source / "observations.json").read_text())
        observations.pop("pending_nodes")
        write_record(source / "observations.json", observations)
        prepared["source_observation_fingerprint"] = observation_fingerprint(observations)
    elif changed == "missing_reference":
        session["nominal_continuation_evidence"] = {"complete": True, "status": "passed", "execution_authorized": False}
    elif changed == "tampered_proof":
        archive.write_text(archive.read_text() + " ")
    elif changed == "missing_proof":
        archive.unlink()
    elif changed == "outside_path":
        session["nominal_continuation_evidence"]["artifact_reference"]["path"] = "../outside.json"
    elif changed == "typed_prefix":
        state = json.loads((source / "run.json").read_text())
        state["outline"][0]["part_name"] = ""
        write_record(source / "run.json", state)
    write_record(directory / "outline_request.json", prepared)
    with pytest.raises((ValueError, OSError)):
        example.load_offline_result()
    assert example._offline is None
    llm.assert_not_called()
    example.bridge.ros2_start.assert_not_called()
    example.bridge.start_system.assert_not_called()


@pytest.mark.parametrize("proof_changes", [{"complete": False}, {"status": "NEEDS_CONTEXT"},
    {"execution_authorized": True}, {"execution_authorized": None},
    {"tasks": None}, {"replaced_task_ids": None}, {"replacement_evidence": None}])
def test_saved_outline_review_requires_complete_nonexecuting_proof(captured, tmp_path, monkeypatch, proof_changes):
    example, _, _, _, _, llm = _saved_outline_review(captured, tmp_path, monkeypatch, proof_changes=proof_changes)
    with pytest.raises(ValueError, match="nominal continuation is incomplete"):
        example.load_offline_result()
    assert example._offline is None
    llm.assert_not_called()


def test_saved_outline_review_continue_keeps_original_request_and_requires_generation(captured, tmp_path, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    example, source, _, _, _, llm = _saved_outline_review(captured, tmp_path, monkeypatch)
    originals = {path: path.read_bytes() for path in source.rglob("*.json")}
    result = example.load_offline_result()
    restored = example._offline
    accepted = deepcopy(restored._prepared["multi_turn_session_state"]["accepted_outline_prefix"])
    monkeypatch.setattr(multi_turn, "_build_phase_prompt", lambda prepared, state: (
        {"accepted_outline_prefix": deepcopy(state["accepted_outline_prefix"])}, "Recorded primitive preview"))

    async def continue_review():
        restored.action("continue_outline")
        await restored._task

    asyncio.run(continue_review())
    state = restored.snapshot()
    assert state["status"] == "primitive_ready" and state["primitive_status"] == "ready"
    assert state["execution_authorized"] is False
    assert state["outline"] == accepted
    assert state["artifact_directory"] != result["artifact_directory"]
    assert state["composition_validation_version"] == 1
    preview = json.loads((Path(state["artifact_directory"]) / "primitive_input_preview.json").read_text())
    assert preview["request_sent"] is False
    assert preview["prompt_input"]["accepted_outline_prefix"] == accepted
    assert {path: path.read_bytes() for path in originals} == originals
    llm.assert_not_called()


def test_saved_outline_review_rejects_a_changed_copy(captured, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example

    example, _, _, _, archive, llm = _saved_outline_review(captured, tmp_path, monkeypatch)
    original = archive.read_bytes()
    copy_records = part_slippage_example.copy_generation_records

    def corrupt_copy(state, source_directory, directory):
        paths = copy_records(state, source_directory, directory)
        copied = directory / "recovery_outline" / "turns" / archive.name
        copied.write_bytes(copied.read_bytes() + b" ")
        return paths

    monkeypatch.setattr(part_slippage_example, "copy_generation_records", corrupt_copy)
    with pytest.raises(ValueError, match="Copied nominal continuation differs"):
        example.load_offline_result()
    assert example._offline is None and archive.read_bytes() == original
    llm.assert_not_called()


@pytest.mark.parametrize("tamper", [None, "program", "proposal", "source", "authorization"])
def test_versioned_composition_regeneration_reload_and_source_preservation(captured, tmp_path, monkeypatch, tamper):
    from cais_spade_llm.resources.recovery_composition import composition_start_views, validate_recovery_primitive_composition

    example, source, _, _, _, llm = _saved_outline_review(captured, tmp_path, monkeypatch)
    originals = {path: path.read_bytes() for path in source.rglob("*.json")}
    example.load_offline_result()
    restored = example._offline
    async def composed(stage):
        assert stage == "primitive"
        prepared = restored._prepared
        state = prepared["multi_turn_session_state"]
        task = state["accepted_outline_prefix"][0]
        entry = prepared["recovery_resources"][task["resource_jid"]]
        resource, part = composition_start_views(session=state, prepared=prepared, task=task)
        steps = [{"primitive": "move_to_named_pose", "params": {"pose_name": "home"}}]
        evidence = validate_recovery_primitive_composition(task=task, primitive_steps=steps,
            recovery_des_model=entry["recovery_des_model"], recovery_snapshot=resource,
            part_context=part, grounding_context={}, primitive_catalog=entry["execution_primitive_catalog"],
            validate_primitive=lambda **_: {"allowed": True, "feasibility_status": "FEASIBLE"})
        assert evidence["allowed"], evidence
        state["accepted_primitive_program"] = [{**deepcopy(task), "primitive_steps": steps, "composition_validation": evidence}]
        state["final_output"] = {"transition_trace": state["accepted_outline_prefix"],
            "accepted_primitive_program": state["accepted_primitive_program"], "primitive_program_complete": True}
        return state
    monkeypatch.setattr(restored, "_run_stage", composed)
    async def run():
        restored.action("continue_outline")
        await restored._task
        restored.action("generate_primitives")
        await restored._task
        first = Path(restored._state["artifact_directory"])
        records = {path: path.read_bytes() for path in first.rglob("*.json")}
        restored.action("regenerate_primitives", feedback="Review the exact bindings")
        await restored._task
        assert restored._state["status"] == "offline_completed", restored._state.get("message")
        assert Path(restored._state["artifact_directory"]) != first
        assert restored._prepared["recovery_feedback"] == "Review the exact bindings"
        assert {path: path.read_bytes() for path in records} == records
    asyncio.run(run())
    assert {path: path.read_bytes() for path in originals} == originals
    directory = Path(restored._state["artifact_directory"])
    composition = json.loads((directory / "composition.json").read_text())
    if tamper == "program":
        composition["accepted_primitive_program"][0]["composition_validation"]["projected_snapshot"]["current_state"] = "failed"
    elif tamper == "source":
        composition["source_outline_fingerprint"] = "changed"
    elif tamper == "authorization":
        composition["execution_authorized"] = True
    elif tamper == "proposal":
        proposal = json.loads((directory / "proposal.json").read_text())
        proposal["macro_tasks"][0]["primitive_steps"] = []
        write_record(directory / "proposal.json", proposal)
    write_record(directory / "composition.json", composition)
    example.set_options(source="offline", scenario="Part slippage", safety_files=[],
                        observation_path=str(restored.path / "observations.json"))
    if tamper:
        with pytest.raises(ValueError):
            example.load_offline_result()
    else:
        state = example.load_offline_result()
        assert state["status"] == "offline_completed" and state["composition_feasibility"] == "FEASIBLE"
        assert state["execution_authorized"] is False
        assert state["primitives"] == composition["accepted_primitive_program"]
        with pytest.raises(ValueError, match="does not authorize execution"):
            example._offline.action("execute")
    llm.assert_not_called()
    example.bridge.ros2_start.assert_not_called()
    example.bridge.start_system.assert_not_called()


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


@pytest.mark.parametrize("cancel_wait", [False, True])
def test_offline_reset_retains_worker_until_drained(offline_example, monkeypatch, cancel_wait):
    import threading

    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    example, path, _ = offline_example
    original = path.read_bytes()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    prepare = PartSlippageRecovery._prepare
    first_owner = []

    def work():
        entered.set()
        try:
            assert release.wait(5), "test did not release the old worker"
        finally:
            finished.set()

    async def delayed(self, feedback=""):
        if not first_owner:
            first_owner.append(self)
            await self._offline_work(work)
        await prepare(self, feedback)

    monkeypatch.setattr(PartSlippageRecovery, "_prepare", delayed)
    llm = AsyncMock(side_effect=AssertionError("Reset called the provider"))
    monkeypatch.setattr(offline_recovery.SavedRecoveryProduct, "ask_llm_structured", llm)

    async def run():
        example.prepare_offline()
        first = example._offline
        reset = None
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            reset = asyncio.create_task(example.action("reset"))
            await asyncio.sleep(0)
            assert example._offline is first and example.snapshot()["busy"]
            assert not finished.is_set()
            example.prepare_offline()
            assert example._offline is first
            with pytest.raises(ValueError, match="Wait for the stopped operation"):
                await example.action("reset")
            if cancel_wait:
                reset.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await reset
                assert example.snapshot()["busy"] and not example._task.done()
            release.set()
            if cancel_wait:
                await example._task
                assert example._offline is first
                await example.action("reset")
            else:
                await reset
            second = example._offline
            assert second is not first and finished.is_set() and first._task.done()
            await second._task
            assert first._prepared == {}
            assert second.snapshot()["status"] == "outline_ready"
            assert example.snapshot()["outline_mode"] == example.snapshot()["primitive_mode"] == "manual"
        finally:
            release.set()
            if reset is not None:
                await asyncio.gather(reset, return_exceptions=True)
            await asyncio.gather(first._task, return_exceptions=True)

    asyncio.run(run())
    assert path.read_bytes() == original
    llm.assert_not_called()


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


@pytest.mark.parametrize('unset', [False, True])
def test_new_capture_native_archive_verifies_and_preserves_typed_source(tmp_path, monkeypatch, unset):
    import hashlib
    from cais_spade_llm.recovery_framework import recovery_motion_preview
    from cais_spade_llm.recovery_framework.recovery_observations import save_native_preview_model_from_capture

    source = tmp_path / 'observations.json'
    record = {'run_id': 'new-capture', 'planning_scene_capture': {'scene_fingerprint': 'scene',
        'model_fingerprint': 'model', 'robot_description': 'urdf', 'robot_description_semantic': 'srdf',
        'native_model_parameters': {'robot_description': {'type': 4, 'string_value': 'urdf'},
            'robot_description_semantic': {'type': 4, 'string_value': 'srdf'},
            'robot_description_planning.exact': {'type': 3, 'double_value': 3.}}}}
    unset_name = 'robot_description_kinematics.KMR_iiwa_arm.kinematics_solver_ik_links'
    if unset:
        record['planning_scene_capture']['native_model_parameters'][unset_name] = {'type': 0}
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


@pytest.mark.parametrize('scenario', ['Part slippage', 'Conveyor breakdown', 'ur5e-1 breakdown', 'Machining breakdown during part processing'])
def test_all_failure_scenarios_prepare_manual_offline_without_live_calls(captured, monkeypatch, scenario):
    def forbidden(*args, **kwargs):
        pytest.fail('Offline preparation contacted a live interface')
    monkeypatch.setattr('spade.agent.Agent.start', forbidden)
    monkeypatch.setattr('cais_spade_llm.recovery_framework.failure_effects.observe_robot', forbidden)
    monkeypatch.setattr(offline_recovery.SavedRecoveryProduct, 'ask_llm_structured', forbidden)
    captured['scenario'] = scenario
    original = deepcopy(captured)
    session = offline_recovery.create_offline_recovery(captured, outline_mode='auto', primitive_mode='auto')
    assert session.runtime.setup['failure_scenario']['scenario'] == scenario
    assert session.runtime.setup['outline_only'] is False
    assert session.snapshot()['outline_mode'] == session.snapshot()['primitive_mode'] == 'manual'
    assert captured == original


@pytest.mark.parametrize('resource_id', ['ur5e-1', 'ur5e-2', 'ur5e-3', 'ur5e-4'])
@pytest.mark.parametrize('explicit', [False, True])
def test_cell_capability_calculation_preserves_explicit_limits_and_source(resource_id, explicit):
    from cais_spade_llm.recovery_framework import ROOT, SCENE_PATH
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_resource_adapter import (
        adapt_recovery_resource_snapshot,
    )

    scene = json.loads(SCENE_PATH.read_text())
    robot = next(row for row in scene['robots'] if row['resource_id'] == resource_id)
    template = json.loads((ROOT / 'cais_spade_llm/initialization/resources/robot_ur5e.json').read_text())
    capabilities = deepcopy(template['ur5e']['gazebo']['static_capabilities'])
    capabilities['gripper_reach']['origin_pose'] = dict(zip(('x', 'y', 'z'), robot['base_xyz'], strict=True))
    if explicit:
        capabilities['gripper_reach']['max_xy_radius_m'] = .3
    jid = resource_id + '@localhost'
    entry = {'resource_type': 'robot', 'static_capabilities': capabilities,
             'recovery_snapshot': deepcopy(capabilities)}
    observations = {'jids': {resource_id: jid}, 'inputs': {'scene': scene}}
    original = deepcopy((entry, observations))
    owner = offline_recovery.SavedRecoveryResource(jid, entry, observations)
    assert owner.static_capabilities == capabilities
    calculation = owner.prepare_current_scene_capabilities()
    if explicit:
        assert owner.static_capabilities['gripper_reach'] == capabilities['gripper_reach']
    else:
        assert 'gripper_reach' not in owner.static_capabilities
        assert 'gripper_reach' not in owner.get_recovery_snapshot()
    assert calculation['recorded_static_capabilities'] == capabilities
    assert calculation['execution_authorized'] is False
    snapshot = owner.get_recovery_snapshot()
    assert snapshot == adapt_recovery_resource_snapshot(resource_jid=jid, resource_type='robot',
        snapshot=snapshot, state_variables={})
    assert (entry, observations) == original
