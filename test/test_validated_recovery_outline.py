from __future__ import annotations

"""Single-event RA/CCA transactions and recorded nominal continuation."""

import asyncio
import json
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_validated_event as validated
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import validate_nominal_continuation
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import multi_turn_phase_response_schema
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
from cais_spade_llm.recovery_framework.outline_validation import (
    OUTLINE_VALIDATION_CONTRACT_VERSION, uses_outline_evidence,
    validate_resource_outline_payload, validate_safety_outline_payload,
)

JID = "ur5e-3@localhost"


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
    owner = SimpleNamespace(jid=JID, get_recovery_snapshot=lambda: deepcopy(snapshot),
                            recovery_des_model=lambda **kwargs: deepcopy(model))
    transport = _Transport(owner)
    planner = SimpleNamespace(product_agent=transport)
    event = {"outline_id": "RECOVERY1", "event_name": "unlisted_recovery_event", "resource_jid": JID,
             "expected_end_state": {"resource_state": "idle"}, "rationale": "Restore the interrupted resource state."}
    return prepared, session, planner, event


class _Transport:
    """Exercise real validators through replaceable in-process transport replies."""

    def __init__(self, owner):
        self.owner = owner
        self.ra_requests, self.cca_requests = [], []
        self.ra_reply = self.cca_reply = None
        self.environment_runtime = SimpleNamespace(context=SimpleNamespace(revision=4))

    async def request_recovery_outline_physical_validation(self, *, resource_jid, payload):
        assert resource_jid == str(self.owner.jid)
        self.ra_requests.append(deepcopy(payload))
        result = validate_resource_outline_payload(self.owner, payload, offline=True)
        return self.ra_reply(result, payload) if self.ra_reply else result

    async def request_recovery_outline_safety_validation(self, *, payload):
        self.cca_requests.append(deepcopy(payload))
        result = validate_safety_outline_payload(payload, validator_jid="cca@localhost")
        return self.cca_reply(result, payload) if self.cca_reply else result


def _run(prepared, session, planner, event, **fields):
    return asyncio.run(validated.handle_validated_event(prepared=prepared, session=session, planner=planner,
        response={"thought": "Use the recorded resource facts.", "next_transition": event, **fields}))


def test_novel_event_is_appended_only_after_real_ra_and_cca_validation():
    prepared, session, planner, event = _case()
    original = deepcopy(prepared)
    status, turn = _run(prepared, session, planner, event)
    assert status == "outline_ready", turn
    assert len(planner.product_agent.ra_requests) == len(planner.product_agent.cca_requests) == 1
    assert session["accepted_outline_prefix"][0]["event_name"] == event["event_name"]
    assert session["accepted_outline_prefix"][0]["candidate_source"] == "llm"
    assert session["symbolic_resources"][JID]["resource_state"] == "idle"
    assert session["outline_monitor_context"]["recorded-rule"]["observations"]
    assert turn["transition_validation"]["execution_authorized"] is False
    assert prepared == original


def test_rejection_feedback_survives_retrieval_and_clears_after_accepted_transition(monkeypatch):
    prepared, session, planner, event = _case()
    validate = validated.validate_event

    async def rejected(**kwargs):
        return {"allowed": False, "feasibility_status": "INFEASIBLE", "reason": "Recorded capability conflict",
                "evidence": {"capability_result": {"feasibility_status": "INFEASIBLE", "checks": []}}}, {}

    monkeypatch.setattr(validated, "validate_event", rejected)
    assert _run(prepared, session, planner, event)[0] == "need_revision"
    history = deepcopy(session["candidate_rejection_feedback"])
    assert history[0]["evidence"]["capability_result"]["feasibility_status"] == "INFEASIBLE"
    asyncio.run(validated.handle_validated_event(prepared=prepared, session=session, planner=planner,
        response={"thought": "Read recorded facts.", "decision": "need_context", "context_requests": ["/resources"]}))
    assert session["candidate_rejection_feedback"] == history
    monkeypatch.setattr(validated, "validate_event", validate)
    assert _run(prepared, session, planner, event)[0] == "outline_ready"
    assert session["candidate_rejection_feedback"] == []


def test_existing_projected_scene_does_not_copy_the_unused_capture(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import _derive_candidate_outline_task

    prepared, session, planner, event = _case()
    task, findings = _derive_candidate_outline_task(candidate_task=event, session_state=session,
                                                   prepared_recovery_request=prepared)
    assert findings == []
    session["projected_physical_evidence"] = deepcopy(prepared["physical_evidence"])
    captured = prepared["physical_evidence"]

    def copy_needed(value):
        if value is captured:
            raise AssertionError("The unused captured scene must not be copied during projected validation")
        return deepcopy(value)

    monkeypatch.setattr(validated, "deepcopy", copy_needed)
    ra, cca = asyncio.run(validated.validate_event(task=task, session=session, prepared=prepared,
        planner=planner, turn={}))
    assert ra["allowed"] and cca["is_safe"]


def _union_case():
    from cais_spade_llm.resources.recovery_feasibility import validate_primitive_support
    from cais_spade_llm.resources.recovery_geometry import project_recovery_outline_event
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )

    prepared, session, planner, event = _case()
    model = prepared["recovery_resources"][JID]["recovery_des_model"]
    model["events"] = [{"event_name": "reset_fault", "updates": {"resource_state": {"set": "idle"}},
                        "primitive_support": [{"primitive": "reset_fault", "params": {}}]}]
    scene = prepared["physical_evidence"]
    scene["resources"] = {rid: {"idle": True, "stationary_contract": {"kind": "recorded hold"}}
                          for rid in scene["snapshot"]["resources"]}
    scene["resource_models"]["ur5e-3"]["geometry_unchanged_events"] = ["reset_fault"]
    prepared["primitive_support_semantics"] = session["primitive_support_semantics"] = "successor_primitive_union"
    owner = planner.product_agent.owner
    owner.validate_recovery_primitive_support = lambda **kwargs: validate_primitive_support(
        task=kwargs["task"], recovery_des_model=kwargs["recovery_des_model"], recovery_snapshot=kwargs["recovery_snapshot"],
        validate_primitive=lambda **_: {"allowed": True, "feasibility_status": "FEASIBLE"})
    owner.project_recovery_outline_event = lambda task, **kwargs: project_recovery_outline_event(task=task,
        recovery_snapshot=kwargs["recovery_snapshot"], recovery_des_model=kwargs["recovery_des_model"],
        evidence=kwargs["physical_input"]["physical_evidence"])

    def save_preview(task):
        request = prepare_outline_event_preview_request(task=task,
            recovery_snapshot=prepared["recovery_resources"][JID]["recovery_snapshot"], recovery_des_model=model,
            physical_evidence=prepared["physical_evidence"], resource_evidence={})
        assert request["status"] == "ready", request
        result = finalize_outline_event_preview(prepared_preview=request, motion_results=[])
        assert result["status"] == "prepared", result
        prepared["physical_evidence"]["resource_models"] = result["resource_models"]
        prepared["physical_evidence"].setdefault("event_evidence", []).extend(result["event_evidence"])

    return prepared, session, planner, event, save_preview


def test_union_ra_and_cca_append_only_with_independent_event_preview():
    prepared, session, planner, event, save_preview = _union_case()
    before = deepcopy(session)
    decision, turn = _run(prepared, session, planner, event)
    assert decision == "need_context", turn
    assert planner.product_agent.ra_requests and planner.product_agent.cca_requests
    assert session["accepted_outline_prefix"] == before["accepted_outline_prefix"]
    assert session["symbolic_resources"] == before["symbolic_resources"]
    assert session["outline_monitor_context"] == before["outline_monitor_context"]
    pending = deepcopy(session["pending_outline_validation"])
    save_preview(pending["task"])
    decision, turn = asyncio.run(validated.retry_pending_event(prepared=prepared, session=session, planner=planner))
    assert decision == "outline_ready", turn
    assert turn["validation_only"] is True and turn["request_sent"] is False
    assert len(session["accepted_outline_prefix"]) == 1
    assert session["accepted_outline_prefix"][0]["event_name"] == event["event_name"]
    assert session["symbolic_resources"][JID]["resource_state"] == "idle"
    assert session["outline_monitor_context"]
    assert "pending_outline_validation" not in session


@pytest.mark.parametrize("explicit_null", [False, True])
def test_resource_only_proposal_preserves_null_or_omitted_binding_across_retry(explicit_null):
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import pending_outline_event

    prepared, session, planner, event, save_preview = _union_case()
    if explicit_null:
        event["part_name"] = None
    original = deepcopy(event)
    decision, _ = _run(prepared, session, planner, event)
    assert decision == "need_context"
    pending = session["pending_outline_validation"]
    assert ("part_name" in pending["task"]) == explicit_null
    recovery = SimpleNamespace(_prepared={"multi_turn_session_state": session})
    assert pending_outline_event(recovery) == pending["task"]
    save_preview(pending["task"])
    decision, _ = asyncio.run(validated.retry_pending_event(prepared=prepared, session=session, planner=planner))
    assert decision == "outline_ready"
    accepted = session["accepted_outline_prefix"][0]
    assert ("part_name" in accepted) == explicit_null
    if explicit_null:
        assert accepted["part_name"] is None
    assert event == original


@pytest.mark.parametrize("error", [TypeError("unexpected keyword argument"), KeyError("programming-key"),
                                  RuntimeError("implementation failed"), ValueError("invalid implementation result")])
def test_programming_error_is_not_recorded_as_missing_observations(monkeypatch, error):
    prepared, session, planner, event = _case()

    async def broken(**kwargs):
        raise error

    monkeypatch.setattr(validated, "validate_event", broken)
    with pytest.raises(type(error), match=str(error).strip("'")):
        _run(prepared, session, planner, event)
    assert session["accepted_outline_prefix"] == []
    assert "pending_outline_validation" not in session


def test_nominal_programming_error_retains_accepted_prefix_and_propagates(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_nominal_evidence as nominal

    prepared, session, planner, event = _case()

    async def broken(**kwargs):
        raise TypeError("nominal helper received unexpected keyword")

    monkeypatch.setattr(nominal, "validate_nominal_continuation", broken)
    with pytest.raises(TypeError, match="unexpected keyword"):
        _run(prepared, session, planner, event)
    assert len(session["accepted_outline_prefix"]) == 1
    assert "pending_outline_validation" not in session


def _rejected_motion_record(task, snapshot, model, physical):
    from cais_spade_llm.resources.recovery_geometry import outline_event_binding
    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record

    request = {"resource_id": "ur5e-3", "capture_fingerprint": "capture", "scene_fingerprint": "native-scene",
        "model_fingerprint": "native-model", "proposal_fingerprint": recovery_validation_fingerprint(task),
        "start_state": {"joint_state": {"name": ["joint"], "position": [0.]}}}
    collision = {"version": 1, "source": "PreviewRecoveryMotion", "phase": "joint_target", "state_index": 3,
        "kind": "unrelated_collision", "reason": "Recorded collision in supplied motion", "contacts_complete": True,
        "checked_contacts": 1, "contacts": [{"body_1": "tool", "body_2": "obstacle", "contacts": [
            {"depth": .01, "position": [0., 0., 0.], "normal": [1., 0., 0.]}]}], "support_pairs": [],
        "continuous_collision_verified": False, "joint_state": {"name": ["joint"], "position": [.03]},
        "request_binding": {**{field: request[field] for field in ("resource_id", "capture_fingerprint",
            "proposal_fingerprint", "scene_fingerprint", "model_fingerprint")}, "support_departure_fingerprint": ""}}
    response = {**request, "status": "NEEDS_CONTEXT", "path_complete": False, "fraction": 0.,
        "sampled_collision_rejected": True, "continuous_collision_verified": False,
        "reason": collision["reason"], "collision_rejection_evidence_json": json.dumps(collision)}
    motion = preview_result_record(request, response)
    return {"status": "rejected", "validation_status": "rejected", "physical_motion_rejected": True,
        "task": deepcopy(task), "rejected_motion": motion, "reason": motion["reason"],
        "event_record_binding": outline_event_binding(task, snapshot, model, physical)}


def test_rejected_supplied_motion_retries_with_factual_feedback_without_state_commit():
    prepared, session, planner, event, _ = _union_case()
    decision, _ = _run(prepared, session, planner, event)
    assert decision == "need_context"
    task = session["pending_outline_validation"]["task"]
    physical = session.setdefault("projected_physical_evidence", deepcopy(prepared["physical_evidence"]))
    entry = prepared["recovery_resources"][JID]
    record = _rejected_motion_record(task, entry["recovery_snapshot"], entry["recovery_des_model"], physical)
    physical["rejected_event_evidence"] = [record]
    before = deepcopy(session)
    decision, turn = asyncio.run(validated.retry_pending_event(prepared=prepared, session=session, planner=planner))
    assert decision == "need_revision"
    assert turn["transition_validation"]["status"] == "rejected"
    stages = {row["validator_role"]: row["evidence"] for row in turn["validation_stages"]}
    assert stages["ResourceAgent"]["feasibility_status"] == "FEASIBLE"
    assert stages["ResourceAgent"]["geometry_evidence"]["status"] == "rejected"
    assert stages["CentralControllerAgent"]["formal_safety_status"] == "not_checked"
    assert session["outline_validation_findings"][0]["constraint_code"] == "physical_motion_rejected"
    assert "pending_outline_validation" not in session
    for field in ("accepted_outline_prefix", "symbolic_resources", "symbolic_parts", "outline_monitor_context"):
        assert session[field] == before[field]


def test_workflow_retains_bound_motion_rejection_and_returns_to_outline_without_llm(tmp_path):
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    prepared, session, planner, event, _ = _union_case()
    decision, _ = _run(prepared, session, planner, event)
    assert decision == "need_context"
    task = session["pending_outline_validation"]["task"]
    entry = prepared["recovery_resources"][JID]
    record = _rejected_motion_record(task, entry["recovery_snapshot"], entry["recovery_des_model"],
                                    prepared["physical_evidence"])
    prepared["multi_turn_session_state"] = session
    recovery = SimpleNamespace(_stopped=False, _generation=1, _prepared=prepared,
        offline=True, _offline_cancel=threading.Event(), _observations={},
        runtime=SimpleNamespace(source_observation_path=None),
        agent=SimpleNamespace(process_planner=planner), _state={"artifact_directory": str(tmp_path)})
    recovery._publish = lambda **values: recovery._state.update(values)
    recovery._offline_work = PartSlippageRecovery._offline_work.__get__(recovery)
    recovery._offline_preview_observations = PartSlippageRecovery._offline_preview_observations.__get__(recovery)
    original = deepcopy(prepared)
    asyncio.run(PartSlippageRecovery.retry_outline_validation(recovery, record))
    current = recovery._prepared["multi_turn_session_state"]
    assert recovery._state["status"] == "outline_ready"
    assert current["accepted_outline_prefix"] == []
    assert current["projected_physical_evidence"]["rejected_event_evidence"] == [record]
    assert current["outline_monitor_context"] == original["multi_turn_session_state"]["outline_monitor_context"]
    artifacts = list((tmp_path / "recovery_outline" / "turns").glob("*validation*.json"))
    assert len(artifacts) == 1
    saved = json.loads(artifacts[0].read_text())
    assert saved["request_sent"] is False and saved["decision"] == "need_revision"
    assert json.loads((tmp_path / "validation_resume.json").read_text())["multi_turn_session_state"] == current


def _incomplete_motion_record(task, snapshot, model, physical, facts=None):
    from cais_spade_llm.resources.recovery_geometry import outline_event_binding
    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record

    state = {"is_diff": False, "joint_state": {"name": ["joint"], "position": [0.]}}
    request = {"resource_id": "ur5e-3", "capture_fingerprint": "capture", "scene_fingerprint": "native-scene",
        "model_fingerprint": "native-model", "proposal_fingerprint": recovery_validation_fingerprint(task),
        "planning_scene": {"is_diff": False, "robot_state": deepcopy(state)}, "start_state": state,
        "group_name": "arm", "link_name": "tool", "frame_id": "world", "mode": "cartesian",
        "waypoints": [{"position": {"x": 1., "y": 0., "z": 0.},
                       "orientation": {"x": 0., "y": 0., "z": 0., "w": 1.}}],
        "max_translation_step": .01, "max_rotation_step": .05, "max_joint_step": .35,
        "velocity_scaling": .1, "acceleration_scaling": .1, "timeout_seconds": 60.}
    diagnostic = {"version": 1, "source": "PreviewRecoveryMotion", "diagnostic_only": True,
        "complete_inputs": True, "failure_kind": "ik_search_failed", "fraction": .5,
        "request_binding": {key: request[key] for key in ("resource_id", "capture_fingerprint",
            "proposal_fingerprint", "scene_fingerprint", "model_fingerprint")},
        **{key: request[key] for key in ("group_name", "link_name", "frame_id", "max_translation_step",
                                       "max_rotation_step", "max_joint_step")},
        "ik_queries": [{"index": 0, "success": False, "error_code": -6, "timeout_seconds": .05,
                        "private_evaluator_data": "must stay outside the prompt"}],
        "waypoints": [{"index": 0, "query_end": 1, "last_accepted_state": {
            "joint_state": {"name": ["joint"], "position": [.99]}}}],
        "joint_bounds": [{"name": "joint", "position_bounded": True, "min_position": -1., "max_position": 1.}]}
    response = {**{key: request[key] for key in diagnostic["request_binding"]},
        "status": "NEEDS_CONTEXT", "path_complete": False, "fraction": .5,
        "sampled_collision_rejected": False, "continuous_collision_verified": False,
        "reason": "Cartesian path is incomplete; no complete motion evidence is available",
        "cartesian_diagnostics_json": json.dumps(diagnostic)}
    motion = preview_result_record(request, response)
    return {"status": "NEEDS_CONTEXT", "validation_status": "NEEDS_CONTEXT", "physical_motion_unresolved": True,
        "task": deepcopy(task), "incomplete_motion": motion, "reason": motion["reason"],
        "resource_evidence_fingerprint": recovery_validation_fingerprint(facts or {}),
        "event_record_binding": outline_event_binding(task, snapshot, model, physical)}


def test_incomplete_motion_preserves_unresolved_verdict_and_allows_one_factual_revision():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _outline_feedback_for_prompt, _render_outline_prompt,
    )

    prepared, session, planner, event, _ = _union_case()
    planner.product_agent.owner.get_recovery_resource_evidence = lambda: {}
    assert _run(prepared, session, planner, event)[0] == "need_context"
    task = session["pending_outline_validation"]["task"]
    entry = prepared["recovery_resources"][JID]
    physical = session.setdefault("projected_physical_evidence", deepcopy(prepared["physical_evidence"]))
    physical["incomplete_event_evidence"] = [_incomplete_motion_record(
        task, entry["recovery_snapshot"], entry["recovery_des_model"], physical)]
    before = deepcopy(session)
    decision, turn = asyncio.run(validated.retry_pending_event(prepared=prepared, session=session, planner=planner))
    assert decision == "need_revision"
    assert turn["transition_validation"]["validation_status"] == "NEEDS_CONTEXT"
    assert turn["transition_validation"]["status"] == "unresolved"
    stages = {row["validator_role"]: row["evidence"] for row in turn["validation_stages"]}
    assert stages["ResourceAgent"]["feasibility_status"] == "FEASIBLE"
    assert stages["CentralControllerAgent"]["validation_status"] == "NEEDS_CONTEXT"
    assert stages["CentralControllerAgent"]["formal_safety_status"] == "not_checked"
    assert "pending_outline_validation" not in session
    for field in ("accepted_outline_prefix", "symbolic_resources", "symbolic_parts", "outline_monitor_context"):
        assert session[field] == before[field]
    finding = session["outline_validation_findings"][0]
    public = _outline_feedback_for_prompt(finding, set(entry["recovery_des_model"]["state_variables"]))
    summary = public["evidence"]["solver_summary"]
    assert summary["failed_query"] == {"index": 0, "error_code": -6, "timeout_seconds": .05}
    assert summary["last_accepted_nearest_bound"] == {"name": "joint", "position": .99, "min_position": -1., "max_position": 1.}
    assert "private_evaluator_data" not in json.dumps(public)
    assert "ik_queries" not in json.dumps(public)
    prompt = _render_outline_prompt({"session_state": session, "llm_input": prepared["llm_input"],
                                    "recovery_resources": prepared["recovery_resources"]})
    assert '"error_code": -6' in prompt and '"timeout_seconds": 0.05' in prompt
    assert '"position": 0.99' in prompt and '"max_position": 1.0' in prompt
    assert "physical validation remains NEEDS_CONTEXT" in prompt
    assert "private_evaluator_data" not in prompt and "ik_queries" not in prompt
    renamed = {**event, "outline_id": "OTHER", "event_name": "another_identifier", "rationale": "Different text."}
    assert _run(prepared, session, planner, renamed)[0] == "need_context"
    assert session["status"] == "NEEDS_CONTEXT"
    assert len(session["incomplete_motion_revision_identities"]) == 1
    assert session["accepted_outline_prefix"] == []


def test_workflow_retains_incomplete_native_record_and_enables_revision_without_llm(tmp_path):
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    prepared, session, planner, event, _ = _union_case()
    owner = planner.product_agent.owner
    owner.get_recovery_resource_evidence = lambda: {}
    assert _run(prepared, session, planner, event)[0] == "need_context"
    task = session["pending_outline_validation"]["task"]
    entry = prepared["recovery_resources"][JID]
    record = _incomplete_motion_record(task, entry["recovery_snapshot"], entry["recovery_des_model"], prepared["physical_evidence"])
    prepared["multi_turn_session_state"] = session
    recovery = SimpleNamespace(_stopped=False, _generation=1, _prepared=prepared,
        offline=True, _offline_cancel=threading.Event(), _observations={},
        runtime=SimpleNamespace(source_observation_path=None),
        agent=SimpleNamespace(process_planner=planner, resources={JID: owner}), _state={"artifact_directory": str(tmp_path)})
    recovery._publish = lambda **values: recovery._state.update(values)
    recovery._offline_work = PartSlippageRecovery._offline_work.__get__(recovery)
    recovery._offline_preview_observations = PartSlippageRecovery._offline_preview_observations.__get__(recovery)
    asyncio.run(PartSlippageRecovery.retry_outline_validation(recovery, record))
    current = recovery._prepared["multi_turn_session_state"]
    assert recovery._state["status"] == "outline_ready"
    assert current["accepted_outline_prefix"] == []
    assert current["projected_physical_evidence"]["incomplete_event_evidence"] == [record]
    artifact = next((tmp_path / "recovery_outline" / "turns").glob("*validation*.json"))
    saved = json.loads(artifact.read_text())
    assert saved["request_sent"] is False and saved["decision"] == "need_revision"


@pytest.mark.parametrize("defect", ["solver_unavailable", "canceled_or_deadline", "unclassified", "missing_input",
                                    "stale_model", "stale_scene", "changed_policy", "forged_flag", "command"])
def test_incomplete_motion_revision_requires_complete_bound_native_evidence(defect):
    from cais_spade_llm.recovery_framework.outline_validation import incomplete_outline_motion_projection

    prepared, _, _, task, _ = _union_case()
    entry, physical = prepared["recovery_resources"][JID], prepared["physical_evidence"]
    record = _incomplete_motion_record(task, entry["recovery_snapshot"], entry["recovery_des_model"], physical)
    motion = record["incomplete_motion"]
    diagnostic = json.loads(motion["cartesian_diagnostics_json"])
    facts = {}
    if defect in {"solver_unavailable", "canceled_or_deadline", "unclassified"}:
        diagnostic["failure_kind"] = defect
    elif defect == "missing_input":
        diagnostic["complete_inputs"] = False
    elif defect == "stale_model":
        motion["model_fingerprint"] = "changed"
    elif defect == "stale_scene":
        physical["snapshot"]["resources"]["ur5e-4"]["current_pose"][0] += .1
    elif defect == "changed_policy":
        facts["controller_config"] = {"cartesian_motion": {"max_joint_step_rad": .2}}
    elif defect == "forged_flag":
        motion.pop("cartesian_diagnostics_json")
    else:
        motion["command_sent"] = True
    if defect != "forged_flag":
        motion["cartesian_diagnostics_json"] = json.dumps(diagnostic)
    with pytest.raises(ValueError):
        incomplete_outline_motion_projection(record=record, task=task, recovery_snapshot=entry["recovery_snapshot"],
            recovery_des_model=entry["recovery_des_model"], physical_evidence=physical, resource_evidence=facts)


def test_nominal_incomplete_motion_can_request_one_revision_but_does_not_claim_readiness(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_nominal_evidence as nominal

    prepared, session, planner, event = _case()
    evidence = {"complete": False, "status": "NEEDS_CONTEXT", "proposal_revision_identities": ["native-query"],
                "failures": [], "reason": "Incomplete supplied motion"}

    async def incomplete(**kwargs):
        return deepcopy(evidence)

    monkeypatch.setattr(nominal, "validate_nominal_continuation", incomplete)
    session["pending_outline_validation"] = {"kind": "nominal", "source_turn_index": 2}
    decision, _ = asyncio.run(validated.retry_pending_event(prepared=prepared, session=session, planner=planner))
    assert decision == "need_revision"
    assert session["nominal_continuation_evidence"] == evidence
    assert session["accepted_outline_prefix"] == []
    session["pending_outline_validation"] = {"kind": "nominal", "source_turn_index": 2}
    decision, _ = asyncio.run(validated.retry_pending_event(prepared=prepared, session=session, planner=planner))
    assert decision == "need_context"


@pytest.mark.parametrize("change", ["task", "binding", "scene", "result_hash", "collision_flag", "command_sent"])
def test_rejected_motion_requires_exact_current_binding_and_authoritative_result(change):
    from cais_spade_llm.recovery_framework.outline_validation import rejected_outline_motion_projection

    prepared, _, _, task, _ = _union_case()
    entry, physical = prepared["recovery_resources"][JID], prepared["physical_evidence"]
    record = _rejected_motion_record(task, entry["recovery_snapshot"], entry["recovery_des_model"], physical)
    if change == "task":
        record["task"]["part_name"] = None
    elif change == "binding":
        record["event_record_binding"]["snapshot_fingerprint"] = "stale"
    elif change == "scene":
        physical["snapshot"]["resources"]["ur5e-4"]["current_pose"][0] += .1
    elif change == "result_hash":
        record["rejected_motion"]["model_fingerprint"] = "stale"
    elif change == "collision_flag":
        record["rejected_motion"]["sampled_collision_rejected"] = False
    else:
        record["rejected_motion"]["command_sent"] = True
    with pytest.raises(ValueError):
        rejected_outline_motion_projection(record=record, task=task, recovery_snapshot=entry["recovery_snapshot"],
            recovery_des_model=entry["recovery_des_model"], physical_evidence=physical)


@pytest.mark.parametrize("obstructed", [True, False])
def test_nominal_obstruction_allows_another_proposal_without_passing_missing_evidence(monkeypatch, obstructed):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_nominal_evidence as nominal

    prepared, session, planner, _ = _case()
    prepared["llm_input"]["goal_conditions"] = []
    prepared["nominal_continuation_semantics"] = "recorded_unfinished_tasks"
    prepared["saved_pending_nodes"] = [{"id": name, "resource_jid": name, "function_name": "machine_part"}
                                       for name in ("M1", "M2")]
    monkeypatch.setattr(nominal, "_nominal_task", lambda node, *_: (
        {"outline_id": node["id"], "resource_jid": node["resource_jid"]}, ""))

    async def checked(*, task, turn, **kwargs):
        status = "INFEASIBLE" if obstructed and task["outline_id"] == "M1" else "NEEDS_CONTEXT"
        ra = {"allowed": False, "feasibility_status": status,
              "reason": "Recorded unavailable resource" if status == "INFEASIBLE" else "Missing owner evidence"}
        turn["validation_stages"] = [{"validator_role": "ResourceAgent", "evidence": ra}]
        return ra, {"is_safe": False}

    monkeypatch.setattr(validated, "validate_event", checked)
    original = deepcopy(session)
    result = asyncio.run(validate_nominal_continuation(session=session, prepared=prepared, planner=planner))
    assert result["status"] == ("not_ready" if obstructed else "NEEDS_CONTEXT")
    assert result["complete"] is False and len(result["failures"]) == 2
    assert result["failures"][1]["validation_stages"][0]["evidence"]["feasibility_status"] == "NEEDS_CONTEXT"
    for field in ("accepted_outline_prefix", "symbolic_resources", "outline_monitor_context"):
        assert session[field] == original[field]


@pytest.mark.parametrize("final_status", ["rejected", "NEEDS_CONTEXT", "incomplete", "capability"])
def test_nominal_result_uses_final_verdict_after_retrieval_and_reports_motion_rejection(monkeypatch, final_status):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_nominal_evidence as nominal
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _render_outline_prompt

    prepared, session, planner, task = _case()
    prepared["llm_input"]["goal_conditions"] = []
    prepared["nominal_continuation_semantics"] = "recorded_unfinished_tasks"
    prepared["saved_pending_nodes"] = [{"id": "nominal", "resource_jid": JID, "function_name": "reset_fault"}]
    monkeypatch.setattr(nominal, "_nominal_task", lambda *args: (task, ""))

    async def checked(*, turn, **kwargs):
        ra = {"allowed": True, "feasibility_status": "FEASIBLE"}
        cca = {"is_safe": False, "validation_status": "NEEDS_CONTEXT" if final_status == "incomplete" else final_status,
            "physical_motion_rejected": final_status == "rejected", "reason": "Recorded owner validation"}
        if final_status == "incomplete":
            cca.update(physical_motion_unresolved=True, revision_identity="recorded-motion",
                solver_summary={"source": "PreviewRecoveryMotion", "failure_kind": "ik_search_failed", "fraction": .5,
                                "failed_query": {"error_code": -6, "timeout_seconds": .05}})
        if final_status == "capability":
            reason = "pose is outside gripper_reach: xy_radius=2.0000 m, max=0.7000 m"
            cca.update(validation_status="rejected", physical_capability_rejected=True, reason=reason,
                capability_result={"feasibility_status": "INFEASIBLE", "checks": [{
                    "feasibility_status": "INFEASIBLE", "reason": reason,
                    "checked_pose": {"x": 2., "y": 0., "z": 1.}, "frame": "world",
                    "gripper_reach": {"max_xy_radius_m": .7}}]},
                capability_source={"kind": "recorded", "ref": f"/resources/{JID}/static_capabilities",
                                   "capture": {"revision": 4}})
        turn["validation_stages"] = [
            {"validator_role": "ResourceAgent", "evidence": {"allowed": False, "feasibility_status": "NEEDS_CONTEXT"}},
            {"validator_role": "ResourceAgent", "evidence": ra},
            {"validator_role": "CentralControllerAgent", "evidence": cca}]
        return ra, cca

    monkeypatch.setattr(validated, "validate_event", checked)
    session["pending_outline_validation"] = {"kind": "nominal", "source_turn_index": 2}
    decision, _ = asyncio.run(validated.retry_pending_event(prepared=prepared, session=session, planner=planner))
    assert decision == {"rejected": "need_next_task", "NEEDS_CONTEXT": "need_context", "incomplete": "need_revision",
                        "capability": "need_next_task"}[final_status]
    assert ("pending_outline_validation" in session) is (final_status == "NEEDS_CONTEXT")
    rendered = json.dumps(session["outline_validation_findings"])
    assert "resource_validation_unavailable" not in rendered
    if final_status == "rejected":
        assert "physical_motion_rejected" in rendered and "safety_rule_violation" not in rendered
        assert "alternative motions have not been exhausted" in rendered
    if final_status == "incomplete":
        assert session["nominal_continuation_evidence"]["status"] == "NEEDS_CONTEXT"
        assert "physical_motion_unresolved" in rendered and "safety_rule_violation" not in rendered
        assert '"error_code": -6' in rendered
    if final_status == "capability":
        assert "physical_capability_rejected" in rendered and "safety_rule_violation" not in rendered
        prompt = _render_outline_prompt({"session_state": session, "llm_input": prepared["llm_input"],
                                        "recovery_resources": prepared["recovery_resources"]})
        assert "xy_radius=2.0000 m, max=0.7000 m" in prompt
        assert "calculated owner motion target" in prompt and '"revision": 4' in prompt


def test_union_nominal_witness_requires_declared_product_effects():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import _exact_nominal_witness

    prepared, _, _, _, _ = _union_case()
    model = prepared["recovery_resources"][JID]["recovery_des_model"]
    model["state_variables"]["part_state"] = {"scope": "part"}
    declaration = model["events"][0]
    declaration["updates"]["part_state"] = {"set": "ready"}
    ra = {"primitive_support": {"semantics": "successor_primitive_union"},
          "geometry_evidence": {"task_contract_witness": {"task_contract": "reset_fault",
              "declaration_fingerprint": recovery_validation_fingerprint(declaration), "params": {},
              "resource_updates": {"resource_state": "idle"}, "part_updates": {}}}}
    node = {"resource_jid": JID, "function_name": "reset_fault", "params": {}}
    assert "effect unresolved" in _exact_nominal_witness(node, ra, prepared)


def _nominal_step_coverage_case():
    steps = [
        {"id": "descend", "op": "move_cartesian", "params": {"z": {"$state": "_task_ctx", "path": ["place_z"]}}},
        {"id": "release", "op": "release_part", "params": {}},
        {"id": "snap_part_to_slot", "op": "snap_part_to_slot", "params": {
            "model_name": {"$state": "_task_ctx", "path": ["model_name"]}},
         "when": [{"predicate": "execution_mode", "args": {"mode": "simulation"}}]},
        {"id": "lift", "op": "move_cartesian", "params": {"z": {"$state": "_task_ctx", "path": ["travel_z"]}}},
    ]
    event = {"event_name": "place_insert", "composition_contract": {"params": {}, "steps": steps},
             "updates": {"held_part": {"set": None}}}
    entries = []
    for index, step in enumerate(steps):
        entries.append({"index": index, "step_id": step["id"],
            "step_fingerprint": recovery_validation_fingerprint(step), "status": "supported",
            "condition": {"status": "applicable", "facts": [{"ref": "execution_mode", "value": "simulation"}]
                if step.get("when") else []},
            "params": {"z": .1} if step["op"] == "move_cartesian" else
                      {"model_name": "KET4_Square_4mm"} if step["op"] == "snap_part_to_slot" else {},
            "interval_indices": [index],
            "effect_evidence_fingerprint": recovery_validation_fingerprint({"step": index, "supported": True})})
    witness = {"task_contract": "place_insert", "declaration_fingerprint": recovery_validation_fingerprint(event),
        "params": {}, "resource_updates": {"held_part": None}, "part_updates": {}, "coverage_version": 1,
        "program_fingerprint": recovery_validation_fingerprint(event["composition_contract"]),
        "interval_count": len(steps), "steps": entries}
    prepared = {"physical_evidence": {"physical_preview_contract_version": 7},
        "recovery_resources": {JID: {"recovery_des_model": {"events": [event], "state_variables": {
            "held_part": {"scope": "resource"}}}}}}
    node = {"resource_jid": JID, "function_name": "place_insert", "params": {}}
    ra = {"primitive_support": {"semantics": "successor_primitive_union"},
          "geometry_evidence": {"task_contract_witness": witness}}
    return node, ra, prepared, event, witness


def test_nominal_witness_requires_every_declared_step_after_hidden_snap_is_added():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import _exact_nominal_witness

    node, ra, prepared, event, witness = _nominal_step_coverage_case()
    before = deepcopy((node, ra, prepared))
    assert _exact_nominal_witness(node, ra, prepared) == ""
    assert (node, ra, prepared) == before
    witness["steps"].pop(2)
    assert "every registered step" in _exact_nominal_witness(node, ra, prepared)
    witness["steps"] = before[1]["geometry_evidence"]["task_contract_witness"]["steps"]
    event["composition_contract"]["steps"][2]["params"]["slot_x"] = .3
    witness["declaration_fingerprint"] = recovery_validation_fingerprint(event)
    witness["program_fingerprint"] = recovery_validation_fingerprint(event["composition_contract"])
    assert "step identity" in _exact_nominal_witness(node, ra, prepared)


@pytest.mark.parametrize("change", [
    "missing_coverage", "wrong_program", "reordered", "index_bool", "missing_condition", "condition_unknown",
    "guard_without_facts", "wrong_step_hash", "unsupported_status", "missing_params", "missing_effect",
    "interval_out_of_bounds", "interval_bool", "interval_duplicate", "extra_interval", "unconditional_skip",
])
def test_nominal_witness_rejects_incomplete_or_conflicting_step_coverage(change):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import _exact_nominal_witness

    node, ra, prepared, _, witness = _nominal_step_coverage_case()
    row = witness["steps"][2]
    if change == "missing_coverage":
        witness.pop("coverage_version")
    elif change == "wrong_program":
        witness["program_fingerprint"] = "changed"
    elif change == "reordered":
        witness["steps"][0], witness["steps"][1] = witness["steps"][1], witness["steps"][0]
    elif change == "index_bool":
        witness["steps"][0]["index"] = False
    elif change == "missing_condition":
        row.pop("condition")
    elif change == "condition_unknown":
        row["condition"]["status"] = "unknown"
    elif change == "guard_without_facts":
        row["condition"]["facts"] = []
    elif change == "wrong_step_hash":
        row["step_fingerprint"] = "changed"
    elif change == "unsupported_status":
        row["status"] = "unresolved"
    elif change == "missing_params":
        row.pop("params")
    elif change == "missing_effect":
        row.pop("effect_evidence_fingerprint")
    elif change == "interval_out_of_bounds":
        row["interval_indices"] = [4]
    elif change == "interval_bool":
        row["interval_indices"] = [True]
    elif change == "interval_duplicate":
        row["interval_indices"] = [2, 2]
    elif change == "extra_interval":
        witness["interval_count"] = 5
    else:
        witness["steps"][0].update(status="skipped", condition={"status": "not_applicable", "facts": [{"value": False}]},
                                   interval_indices=[])
    assert _exact_nominal_witness(node, ra, prepared).startswith("NEEDS_CONTEXT:")


def test_nominal_witness_allows_owner_supported_skipped_branch_and_pure_calculation():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import _exact_nominal_witness

    node, ra, prepared, event, witness = _nominal_step_coverage_case()
    witness["steps"][2].update(status="skipped", condition={"status": "not_applicable", "facts": [
        {"ref": "execution_mode", "value": "hardware"}]}, interval_indices=[])
    witness["steps"][3]["interval_indices"] = [2]
    witness["interval_count"] = 3
    assert _exact_nominal_witness(node, ra, prepared) == ""
    calculation = {"id": "calculate", "op": "compute_place_targets", "params": {}}
    event["composition_contract"]["steps"].append(calculation)
    witness["steps"].append({"index": 4, "step_id": "calculate", "step_fingerprint": recovery_validation_fingerprint(calculation),
        "status": "supported", "condition": {"status": "applicable", "facts": []}, "params": {},
        "interval_indices": [], "effect_evidence_fingerprint": recovery_validation_fingerprint({"calculation": "recorded"})})
    witness["program_fingerprint"] = recovery_validation_fingerprint(event["composition_contract"])
    witness["declaration_fingerprint"] = recovery_validation_fingerprint(event)
    assert _exact_nominal_witness(node, ra, prepared) == ""


@pytest.mark.parametrize("version", [None, 1, 2, 3, 4, 5, 6])
def test_historical_nominal_witness_does_not_rewrite_recorded_step_coverage(version):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import _exact_nominal_witness

    node, ra, prepared, _, witness = _nominal_step_coverage_case()
    prepared["physical_evidence"].clear()
    if version is not None:
        prepared["physical_evidence"]["physical_preview_contract_version"] = version
    for key in ("coverage_version", "program_fingerprint", "interval_count", "steps"):
        witness.pop(key)
    before = deepcopy((node, ra, prepared))
    assert _exact_nominal_witness(node, ra, prepared) == ""
    assert (node, ra, prepared) == before


def test_nominal_argument_adapter_preserves_environment_requirements():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import _nominal_arguments
    from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry

    declarations = {argument.name: {} for argument in robot_task_registry()["move_home"].arguments}
    node = {"id": "saved-task", "resource_jid": JID, "function_name": "move_home",
            "params": {"resource_id": "ur5e-3", "home_available": True}}
    event = {"event_id": 12, "event_name": "move_home", "parameter_bindings": {
        "resource_id": {"equals": "ur5e-3", "type": "string"}, "home_available": {"equals": True}}}
    prepared = {"nominal_continuation_semantics": "recorded_unfinished_tasks",
        "recovery_resources": {JID: {"resource_type": "robot", "recovery_des_model": {"events": [
            {"event_name": "move_home", "composition_contract": {"params": declarations}}]},
            "resource_evidence": {"inputs": {}, "source": {"run_id": "saved", "revision": 3}}}},
        "physical_evidence": {"jids": {"ur5e-3": JID}, "product_name": "assembly_board-v1",
            "environment_models": {"ur5e-3": {"events": [event]}}, "products": {}}}
    original = deepcopy(prepared), deepcopy(node)
    arguments, evidence = _nominal_arguments(node, prepared)
    assert arguments == {"task_id": "saved-task", "product_jid": "assembly_board-v1"}
    assert evidence["environment_parameters"] == node["params"]
    assert evidence["environment_event"] == event
    assert evidence["completion_evidence_status"] == "requires_validation"
    assert (prepared, node) == original
    node["params"]["unregistered"] = True
    with pytest.raises(ValueError, match="exact recorded event binding"):
        _nominal_arguments(node, prepared)
    node["params"].pop("unregistered")
    node["params"]["home_available"] = False
    with pytest.raises(ValueError, match="exact recorded event binding"):
        _nominal_arguments(node, prepared)
    node["params"]["home_available"] = True
    binding = event["parameter_bindings"]["resource_id"]
    for key, value in (("type", "number"), ("type", ["string", "null"]), ("type", None),
                       ("reference", "resource_id"), ("reference", None)):
        original_binding = deepcopy(binding)
        binding[key] = value
        with pytest.raises(ValueError, match="unsupported nominal parameter binding semantics: resource_id"):
            _nominal_arguments(node, prepared)
        binding.clear()
        binding.update(original_binding)
    unresolved = deepcopy(event)
    unresolved["event_id"] = 13
    unresolved["parameter_bindings"]["resource_id"]["reference"] = "resource_id"
    prepared["physical_evidence"]["environment_models"]["ur5e-3"]["events"].append(unresolved)
    with pytest.raises(ValueError, match="unsupported nominal parameter binding semantics"):
        _nominal_arguments(node, prepared)
    prepared["physical_evidence"]["environment_models"]["ur5e-3"]["events"].pop()
    prepared.pop("nominal_continuation_semantics")
    with pytest.raises(ValueError, match="absent from the registered task contract"):
        _nominal_arguments(node, prepared)


@pytest.mark.parametrize("dimensions,model_name", [
    (None, "gear_small"), ([], "gear_small"), ([0.05, 0.05], "gear_small"),
    ([0.05, 0.05, 0], "gear_small"), ([0.05, True, 0.05], "gear_small"),
    ([0.05, float("nan"), 0.05], "gear_small"), ([0.05, 0.05, 0.05], ""),
])
def test_nominal_argument_adapter_rejects_missing_geometry_before_fallback(monkeypatch, dimensions, model_name):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import _nominal_arguments
    from cais_spade_llm.recovery_framework import workflow_execution
    from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry

    def forbidden(*args, **kwargs):
        raise AssertionError("Incomplete recorded geometry must not reach the live adapter's fallback")

    monkeypatch.setattr(workflow_execution, "_robot_arguments", forbidden)
    params = {"resource_id": "ur5e-3", "part_name": "gear_small",
              "destination_location": "Buffer For Machined parts"}
    node = {"id": "saved-task", "resource_jid": JID, "function_name": "place_approach", "params": params}
    declarations = {argument.name: {} for argument in robot_task_registry()["place_approach"].arguments}
    event = {"event_id": 12, "event_name": "place_approach",
             "parameter_bindings": {key: {"equals": value} for key, value in params.items()}}
    prepared = {"nominal_continuation_semantics": "recorded_unfinished_tasks",
        "recovery_resources": {JID: {"resource_type": "robot", "recovery_des_model": {"events": [
            {"event_name": "place_approach", "composition_contract": {"params": declarations}}]},
            "resource_evidence": {"inputs": {"geometry": {"parts": {
                "dimensions_m": {"gear_small": dimensions}, "model_map": {"gear_small": model_name}}}}}}},
        "physical_evidence": {"jids": {"ur5e-3": JID}, "environment_models": {"ur5e-3": {"events": [event]}}}}
    with pytest.raises(ValueError, match="recorded dimensions or model identity"):
        _nominal_arguments(node, prepared)


def test_new_nominal_continuation_replays_failed_owner_before_final_goals():
    prepared, session, planner, _ = _case()
    prepared["nominal_continuation_semantics"] = "recorded_unfinished_tasks"
    model = prepared["recovery_resources"][JID]["recovery_des_model"]
    model["events"] = [{"event_name": "reset_fault", "guards": {"resource_state": {"equals": "failed"}},
        "updates": {"resource_state": {"set": "idle"}},
        "primitive_support": [{"primitive": "reset_fault", "params": {}}],
        "composition_contract": {"params": {}, "required_params": [], "entry_state": "failed", "entry_guards": [],
            "steps": [{"id": "completion", "executor": "primitive", "op": "reset_fault", "args": {}, "guards": []}],
            "effects": []}}]
    prepared["saved_pending_nodes"] = [{"id": "interrupted", "resource_jid": JID,
        "function_name": "reset_fault", "params": {}, "status": "pending", "predecessors": []}]
    prepared["marked_reentry_context"] = {"pending_suffix_summary": [
        {"role": "recovery_replaced", "pending_task_ids": ["interrupted"]}]}
    initial = deepcopy(session)
    result = asyncio.run(validate_nominal_continuation(session=session, prepared=prepared, planner=planner))
    assert result["complete"] is True, result
    assert result["replaced_task_ids"] == []
    assert result["tasks"][0]["task_id"] == "interrupted"
    assert result["execution_authorized"] is False
    assert session["symbolic_resources"] == initial["symbolic_resources"]
    assert session["accepted_outline_prefix"] == []
    prepared["llm_input"]["goal_conditions"][0]["expected"] = "completed"
    result = asyncio.run(validate_nominal_continuation(session=session, prepared=prepared, planner=planner))
    assert result["complete"] is False


@pytest.mark.parametrize("missing", [None, "held_part", "geometry"])
def test_nominal_search_keeps_checked_state_without_copying_archived_attempts(missing):
    class ArchivedRecord(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("Archived attempts must remain on the original session")

    prepared, session, planner, _ = _case()
    prepared["nominal_continuation_semantics"] = "recorded_unfinished_tasks"
    model = prepared["recovery_resources"][JID]["recovery_des_model"]
    model["events"] = [{"event_name": "reset_fault", "guards": {
        "resource_state": {"equals": "failed"}, "held_part": {"equals": None}},
        "updates": {"resource_state": {"set": "idle"}},
        "primitive_support": [{"primitive": "reset_fault", "params": {}}],
        "composition_contract": {"params": {}, "required_params": [], "entry_state": "failed", "entry_guards": [],
            "steps": [{"id": "completion", "executor": "primitive", "op": "reset_fault", "args": {}, "guards": []}],
            "effects": []}}]
    prepared["saved_pending_nodes"] = [{"id": "interrupted", "resource_jid": JID,
        "function_name": "reset_fault", "params": {}, "status": "pending", "predecessors": []}]
    session["projected_physical_evidence"] = deepcopy(prepared["physical_evidence"])
    session["observation_count"] = 0
    if missing == "held_part":
        session["symbolic_resources"][JID].pop("held_part")
        prepared["recovery_resources"][JID]["recovery_snapshot"].pop("held_part")
    elif missing == "geometry":
        session["projected_physical_evidence"]["geometry"]["resources"].pop("ur5e-3")
        prepared["physical_evidence"]["geometry"]["resources"].pop("ur5e-3")
    original_session, original_prepared = deepcopy(session), deepcopy(prepared)
    expected = asyncio.run(validate_nominal_continuation(session=deepcopy(session), prepared=prepared, planner=planner))
    expected_ra, expected_cca = planner.product_agent.ra_requests[:], planner.product_agent.cca_requests[:]
    planner.product_agent.ra_requests.clear()
    planner.product_agent.cca_requests.clear()
    archives = {key: ArchivedRecord({"saved": {"original": [1, 2, 3]}})
                for key in ("turns", "nominal_continuation_evidence", "pending_outline_validation")}
    session.update(archives)
    result = asyncio.run(validate_nominal_continuation(session=session, prepared=prepared, planner=planner))
    assert result == expected
    assert result["complete"] is (missing is None)
    if missing:
        assert result["status"] == "NEEDS_CONTEXT"
    assert planner.product_agent.ra_requests == expected_ra
    assert planner.product_agent.cca_requests == expected_cca
    assert {key: value for key, value in session.items() if key not in archives} == original_session
    assert prepared == original_prepared
    for key, archive in archives.items():
        assert session[key] is archive
        assert archive == {"saved": {"original": [1, 2, 3]}}


def test_nominal_search_validation_mutations_do_not_cross_branches(monkeypatch):
    prepared, session, planner, _ = _case()
    prepared["nominal_continuation_semantics"] = "recorded_unfinished_tasks"
    model = prepared["recovery_resources"][JID]["recovery_des_model"]
    model["events"] = [{"event_name": "reset_fault", "guards": {"resource_state": {"equals": "failed"}},
        "updates": {"resource_state": {"set": "idle"}},
        "primitive_support": [{"primitive": "reset_fault", "params": {}}],
        "composition_contract": {"params": {}, "required_params": [], "entry_state": "failed", "entry_guards": [],
            "steps": [{"id": "completion", "executor": "primitive", "op": "reset_fault", "args": {}, "guards": []}],
            "effects": []}}]
    prepared["saved_pending_nodes"] = [{"id": task_id, "resource_jid": JID,
        "function_name": "reset_fault", "params": {}, "status": "pending", "predecessors": []}
        for task_id in ("interrupted_a", "interrupted_b")]
    session["projected_physical_evidence"] = deepcopy(prepared["physical_evidence"])
    session["validation_resource_context"] = {JID: {"resource_evidence": {"recorded": [1]}}}
    session["observation_count"] = 0
    original = deepcopy(session)
    validate_event = validated.validate_event
    visited = []

    async def validate_branch(*, task, session, **kwargs):
        visited.append(task["outline_id"])
        assert session["symbolic_resources"] == original["symbolic_resources"]
        assert session["outline_monitor_context"] == original["outline_monitor_context"]
        assert session["projected_physical_evidence"] == original["projected_physical_evidence"]
        assert session["validation_resource_context"] == original["validation_resource_context"]
        if task["outline_id"] == "interrupted_a":
            session["symbolic_resources"][JID]["resource_state"] = "idle"
            session["outline_monitor_context"]["branch"] = {"history": ["rejected"]}
            session["projected_physical_evidence"]["snapshot"]["resources"]["ur5e-4"]["held_part"] = "gear_small"
            session["validation_resource_context"][JID]["resource_evidence"]["recorded"].append(2)
            return {"allowed": False, "reason": "Rejected branch"}, {}
        return await validate_event(task=task, session=session, **kwargs)

    monkeypatch.setattr(validated, "validate_event", validate_branch)
    result = asyncio.run(validate_nominal_continuation(session=session, prepared=prepared, planner=planner))
    assert visited == ["interrupted_a", "interrupted_b"]
    assert result["complete"] is False
    assert len(planner.product_agent.ra_requests) == len(planner.product_agent.cca_requests) == 1
    assert session == original


def test_new_nominal_continuation_requires_recorded_product_completion():
    prepared, session, planner, _ = _case()
    prepared["nominal_continuation_semantics"] = "recorded_unfinished_tasks"
    prepared["outline_safety_specifications"] = []
    session["symbolic_resources"][JID]["resource_state"] = "idle"
    prepared["physical_evidence"]["requirements"] = {"gear_small": [
        {"processesToComplete": [{"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}]}]}
    prepared["physical_evidence"]["products"] = {"gear_small": {"state": "assembled", "processCompleted": []}}
    result = asyncio.run(validate_nominal_continuation(session=session, prepared=prepared, planner=planner))
    assert result["complete"] is False
    assert "requirements remain unfinished" in result["reason"]
    prepared["physical_evidence"]["products"]["gear_small"]["processCompleted"] = [
        {"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}]
    result = asyncio.run(validate_nominal_continuation(session=session, prepared=prepared, planner=planner))
    assert result["complete"] is True


def test_recorded_product_completion_retires_only_matching_part_work(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_nominal_evidence as nominal

    params = {"resource_id": "ur5e-3", "part_name": "gear_small", "destination_location": "assembly_board-v1"}
    nodes = {
        "approach": {"id": "approach", "resource_jid": JID, "function_name": "approach",
                     "params": deepcopy(params), "sequence_index": 0},
        "complete": {"id": "complete", "resource_jid": JID, "function_name": "complete",
                     "params": deepcopy(params), "sequence_index": 1},
        "cleanup": {"id": "cleanup", "resource_jid": JID, "function_name": "cleanup",
                    "params": {"resource_id": "ur5e-3"}, "sequence_index": 2},
    }
    effects = {"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}
    declarations = {name: {"event_id": index, "event_name": name,
        "product_effects": {"processCompleted": [effects] if name == "complete" else []}}
        for index, name in enumerate(nodes)}
    monkeypatch.setattr(nominal, "_nominal_arguments", lambda node, prepared: ({}, {
        "environment_event": declarations[node["id"]],
        "environment_event_fingerprint": recovery_validation_fingerprint(declarations[node["id"]])}))
    completion = {"declaration": {"event_id": 1, "event_name": "complete", "parameters": deepcopy(params)},
        "resource_id": "ur5e-3", "declaration_fingerprint": recovery_validation_fingerprint(declarations["complete"]),
        "source_kind": "resource_event_projection", "product_effects": {"gear_small": {"processCompleted": [effects]}}}
    physical = {"requirements": {"gear_small": [{"processesToComplete": [effects]}]},
        "products": {"gear_small": {"state": "assembled", "processCompleted": [effects]}},
        "product_completion_records": [completion]}
    session, prepared = {"projected_physical_evidence": physical}, {}
    original = deepcopy((nodes, session))
    result = nominal._fulfilled_nominal_tasks(nodes, session, prepared)
    assert set(result) == {"approach", "complete"}
    assert result["approach"]["product_completion"] == completion
    assert (nodes, session) == original
    physical["product_completion_records"] = []
    assert nominal._fulfilled_nominal_tasks(nodes, session, prepared) == {}
    physical["product_completion_records"] = [deepcopy(completion)]
    physical["product_completion_records"][0]["declaration"]["parameters"]["destination_location"] = "Buffer For Machined parts"
    assert nominal._fulfilled_nominal_tasks(nodes, session, prepared) == {}


def _alternate_completion_case(monkeypatch, *, owners=("ur5e-4", "ur5e-3"), process=None):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_nominal_evidence as nominal

    original_owner, owner = owners
    process = process or {"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}
    params = {"resource_id": original_owner, "part_name": "gear_small", "destination_location": "assembly_board-v1"}
    effects = {"state": "assembled", "processCompleted": [process]}
    event_name = "complete"
    declared_effects = deepcopy(effects)
    if process.get("process") == "assembly":
        event_name = "place_insert"
        params.update(handoff_acknowledged=True, assembly_acknowledged=True, target=process["target"])
        effects["target"] = process["target"]
        declared_effects = {"state": "assembled", "target": {"set_from_param": "target"},
            "processCompleted": [{"process": "assembly", "target": {"set_from_param": "target"}}]}
    events = {}
    for rid, identifier in ((original_owner, 65), (owner, 59)):
        events[rid] = {"event_id": identifier, "event_name": event_name,
            "parameter_bindings": {"resource_id": {"equals": rid},
                "part_name": {"required": True, "type": "string", "reference": "part_name"},
                "destination_location": {"equals": "assembly_board-v1"}},
            "product_effects": deepcopy(declared_effects)}
        if event_name == "place_insert":
            events[rid]["parameter_bindings"].update(handoff_acknowledged={"equals": True},
                assembly_acknowledged={"equals": True}, target={"required": True, "type": "string"})
    nodes = {
        "approach": {"id": "approach", "resource_jid": original_owner + "@localhost", "function_name": "approach",
                     "params": deepcopy(params), "sequence_index": 0, "predecessors": []},
        "complete": {"id": "complete", "resource_jid": original_owner + "@localhost", "function_name": "complete",
                     "params": deepcopy(params), "sequence_index": 1, "predecessors": ["approach"]},
        "cleanup": {"id": "cleanup", "resource_jid": original_owner + "@localhost", "function_name": "cleanup",
                    "params": {"resource_id": original_owner}, "sequence_index": 2, "predecessors": ["complete"]},
    }
    declarations = {"complete": events[original_owner],
        "approach": {"event_id": 64, "event_name": "approach", "product_effects": {"processCompleted": []}},
        "cleanup": {"event_id": 66, "event_name": "cleanup", "product_effects": {"processCompleted": []}}}
    monkeypatch.setattr(nominal, "_nominal_arguments", lambda node, prepared: ({}, {
        "environment_event": declarations[node["id"]],
        "environment_event_fingerprint": recovery_validation_fingerprint(declarations[node["id"]])}))
    accepted = {"outline_id": "independent-completion", "event_name": "new-completion-identifier",
        "resource_jid": owner + "@localhost", "part_name": "gear_small",
        "expected_start_state": {"held_part": "gear_small"},
        "expected_end_state": {"part_state": "assembled", "held_part": None}}
    record = {"declaration": {"event_id": 59, "event_name": event_name, "parameters": {**params, "resource_id": owner}},
        "resource_id": owner, "declaration_fingerprint": recovery_validation_fingerprint(events[owner]),
        "source_kind": "resource_event_projection", "kind": "predicted", "part_location": "assembly_board-v1",
        "task_id": recovery_validation_fingerprint({field: accepted[field] for field in (
            "resource_jid", "part_name", "expected_start_state", "expected_end_state")}),
        "product_effects": {"gear_small": deepcopy(effects)}}
    part = {"state": "assembled", "location": "assembly_board-v1", "processCompleted": [deepcopy(process)],
        "processCompleted_complete": True, "processCompleted_evidence": {
            "complete": True, "source_kind": "resource_event_projection", "checkpoint": "independent-completion"}}
    physical = {"requirements": {"gear_small": [{"processesToComplete": [deepcopy(process)]}]},
        "products": {"gear_small": deepcopy(part)}, "snapshot": {"parts": {"gear_small": deepcopy(part)}},
        "geometry": {"parts": {"gear_small": {"target": process.get("target")}}},
        "environment_models": {rid: {"events": [event]} for rid, event in events.items()},
        "jids": {rid: rid + "@localhost" for rid in owners}, "product_completion_records": [record]}
    session = {"projected_physical_evidence": physical, "accepted_outline_prefix": [accepted],
               "symbolic_resources": {original_owner + "@localhost": {"resource_state": "failed", "held_part": None}}}
    return nominal, nodes, session, {"completed_task_ids": []}, record


@pytest.mark.parametrize("owners,process", [(("ur5e-4", "ur5e-3"), None),
    (("M1", "M2"), {"process": "trim", "result": "square"})])
def test_supported_product_completion_from_another_owner_retires_product_work_only(monkeypatch, owners, process):
    """Synthetic supported-completion records exercise task obligations, not motion approval."""
    nominal, nodes, session, prepared, record = _alternate_completion_case(monkeypatch, owners=owners, process=process)
    original = deepcopy((nodes, session, prepared))
    result = nominal._fulfilled_nominal_tasks(nodes, session, prepared)
    assert set(result) == {"approach", "complete"}
    assert result["complete"]["product_completion"] == record
    assert result["complete"]["resource_effects_fulfilled"] is False
    assert result["complete"]["recorded_task"]["resource_jid"] == owners[0] + "@localhost"
    assert "cleanup" not in result
    assert (nodes, session, prepared) == original
    requirement = {"entity_kind": "resource", "entity": owners[0] + "@localhost",
                   "field": "resource_state", "expected": "idle"}
    assert nominal._completion_goals(session, prepared, [requirement]) == {
        "complete": False, "status": "not_ready", "unmet_conditions": [requirement]}


@pytest.mark.parametrize("defect", ["no_record", "unaccepted", "task_id", "declaration", "product_effects",
    "part_location", "part_state_only", "missing_ledger", "hard_resource_requirement", "external_dependency",
    "cycle", "missing_index", "parameter_binding"])
def test_alternate_owner_completion_requires_supported_effects_and_dependencies(monkeypatch, defect):
    nominal, nodes, session, prepared, record = _alternate_completion_case(monkeypatch)
    physical = session["projected_physical_evidence"]
    if defect == "no_record":
        physical["product_completion_records"] = []
    elif defect == "unaccepted":
        session["accepted_outline_prefix"] = []
    elif defect == "task_id":
        record["task_id"] = "unrelated accepted event"
    elif defect == "declaration":
        record["declaration_fingerprint"] = "changed"
    elif defect == "product_effects":
        record["product_effects"]["gear_small"]["state"] = "ready"
    elif defect == "part_location":
        record["part_location"] = "Buffer For Machined parts"
    elif defect == "part_state_only":
        physical["products"]["gear_small"]["processCompleted"] = []
    elif defect == "missing_ledger":
        physical["snapshot"]["parts"]["gear_small"].pop("processCompleted_evidence")
    elif defect == "hard_resource_requirement":
        physical["requirements"]["gear_small"].append({"resource_id": "ur5e-4"})
        physical["products"]["gear_small"]["resource_id"] = "ur5e-3"
    elif defect == "external_dependency":
        nodes["approach"]["predecessors"] = ["independent-owner-cleanup"]
    elif defect == "cycle":
        nodes["approach"]["predecessors"] = ["complete"]
    elif defect == "missing_index":
        nodes["complete"].pop("sequence_index")
    else:
        record["declaration"]["parameters"]["part_name"] = "KET4_Square_4mm"
    assert nominal._fulfilled_nominal_tasks(nodes, session, prepared) == {}


def test_alternate_owner_completion_preserves_external_cleanup_until_recorded_complete(monkeypatch):
    nominal, nodes, session, prepared, _ = _alternate_completion_case(monkeypatch)
    nodes["approach"]["predecessors"] = ["independent-owner-cleanup"]
    assert nominal._fulfilled_nominal_tasks(nodes, session, prepared) == {}
    prepared["completed_task_ids"] = ["independent-owner-cleanup"]
    assert set(nominal._fulfilled_nominal_tasks(nodes, session, prepared)) == {"approach", "complete"}


def test_union_nominal_retry_uses_projected_start_and_exact_registered_task():
    from cais_spade_llm.resources.resource_safety_preparation import (
        prepare_outline_event_preview_request, finalize_outline_event_preview,
    )

    prepared, session, planner, event, save_preview = _union_case()
    prepared["saved_pending_nodes"] = [{"id": "nominal-1", "resource_jid": JID,
        "function_name": "reset_fault", "params": {}, "predecessors": []}]
    decision, _ = _run(prepared, session, planner, event)
    assert decision == "need_context"
    save_preview(session["pending_outline_validation"]["task"])
    decision, turn = asyncio.run(validated.retry_pending_event(prepared=prepared, session=session, planner=planner))
    assert decision == "need_context", turn
    pending = session["pending_outline_validation"]
    assert pending["kind"] == "nominal" and pending["task"]["required_task_contract"] == "reset_fault"
    assert pending["recovery_snapshot"]["resource_state"] == "idle"
    accepted = deepcopy(session["accepted_outline_prefix"])
    request = prepare_outline_event_preview_request(task=pending["task"], recovery_snapshot=pending["recovery_snapshot"],
        recovery_des_model=prepared["recovery_resources"][JID]["recovery_des_model"],
        physical_evidence=pending["projected_physical_evidence"], resource_evidence={})
    result = finalize_outline_event_preview(prepared_preview=request, motion_results=[])
    assert result["status"] == "prepared", result
    session["projected_physical_evidence"]["resource_models"] = result["resource_models"]
    session["projected_physical_evidence"].setdefault("event_evidence", []).extend(result["event_evidence"])
    decision, turn = asyncio.run(validated.retry_pending_event(prepared=prepared, session=session, planner=planner))
    assert decision == "outline_ready", turn
    assert session["accepted_outline_prefix"] == accepted
    assert session["nominal_continuation_evidence"]["complete"] is True
    assert session["nominal_continuation_evidence"]["tasks"][0]["task_id"] == "nominal-1"


def test_marked_ra_support_does_not_turn_capability_witnesses_into_a_projection(monkeypatch):
    from cais_spade_llm.resources import recovery_feasibility, recovery_geometry, resource_safety_preparation

    prepared, _, planner, event = _case()
    owner = planner.product_agent.owner
    calls = []
    support = {"semantics": "successor_primitive_union", "primitives": [
        {"primitive": "reset_fault", "params": {}, "feasibility_status": "FEASIBLE"}]}

    def evaluate(**kwargs):
        calls.append(deepcopy(kwargs))
        return {"allowed": True, "feasibility_status": "FEASIBLE", "primitive_support": deepcopy(support)}

    def forbidden(*args, **kwargs):
        raise AssertionError("Unordered support must not enter ordered derivation or projection")

    owner.validate_recovery_primitive_support = evaluate
    monkeypatch.setattr(recovery_feasibility, "validate_recovery_transition", forbidden)
    monkeypatch.setattr(recovery_geometry, "project_outline_evidence", forbidden)
    monkeypatch.setattr(resource_safety_preparation, "prepare_recovery_outline_preview", forbidden)
    payload = {"validation_contract_version": 2, "primitive_support_semantics": "successor_primitive_union",
               "candidates": [{"event_id": event["outline_id"], "task": event,
                               "physical_input": {"physical_evidence": prepared["physical_evidence"],
                                                  "parameter_witnesses": [{"primitive": "reset_fault", "params": {}}]}}]}
    before = deepcopy(payload)
    row = validate_resource_outline_payload(owner, payload, offline=True)["results"][0]
    assert row["allowed"] is True and row["feasibility_status"] == "FEASIBLE"
    assert row["primitive_support"] == support
    assert row["geometry_evidence"]["status"] == "needs_context"
    assert row["geometry_evidence"]["interval_complete"] is False
    assert "trace" not in row["primitive_support"]
    assert "projected_snapshot" not in row["primitive_support"]
    assert row["execution_authorized"] is False
    assert calls[0]["physical_input"]["primitive_support_semantics"] == "successor_primitive_union"
    assert calls[0]["physical_input"]["parameter_witnesses"] == payload["candidates"][0]["physical_input"]["parameter_witnesses"]
    assert payload == before


@pytest.mark.parametrize("semantics", ["successor_primitive_union", "unknown"])
def test_marked_ra_missing_owner_or_unknown_semantics_cannot_fall_back(semantics, monkeypatch):
    from cais_spade_llm.resources import recovery_feasibility

    _, _, planner, event = _case()
    monkeypatch.setattr(recovery_feasibility, "validate_recovery_transition", lambda **kwargs: pytest.fail("legacy fallback"))
    row = validate_resource_outline_payload(planner.product_agent.owner, {
        "validation_contract_version": 2, "primitive_support_semantics": semantics,
        "candidates": [{"task": event}]}, offline=True)["results"][0]
    assert row["allowed"] is False and row["feasibility_status"] == "NEEDS_CONTEXT"
    if semantics == "successor_primitive_union":
        assert f"/resources/{JID}/recovery_des_model" in row["context_requests"]
        assert f"/resources/{JID}/recovery_snapshot" in row["context_requests"]


def test_marked_robot_wrapper_passes_scoped_witnesses_without_deriving_parameters(monkeypatch):
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
    from cais_spade_llm.resources import recovery_feasibility

    calls = []
    primitive_calls = []
    witness = {"primitive": "move_cartesian", "event_name": "move_home", "step_index": 0,
               "params": {"x": .1, "y": .2, "z": 1.2}, "source": "recorded", "captured_at": "time", "revision": 4}
    owner = SimpleNamespace(jid=JID, check_recovery_primitive_feasibility=lambda **kwargs: primitive_calls.append(kwargs) or {
        "allowed": True, "feasibility_status": "FEASIBLE"})

    def evaluate(**kwargs):
        calls.append(kwargs)
        kwargs["validate_primitive"](primitive="move_cartesian", params=deepcopy(witness["params"]))
        return {"allowed": True, "feasibility_status": "FEASIBLE"}

    monkeypatch.setattr(recovery_feasibility, "validate_primitive_support", evaluate)
    RobotAgent.validate_recovery_primitive_support(owner, task={}, recovery_des_model={},
        recovery_snapshot={"current_pose": {"x": 88., "y": 99., "z": 100.}},
        physical_input={"primitive_support_semantics": "successor_primitive_union", "parameter_witnesses": [witness],
                        "parameter_evidence": {"dz": 123.}, "model_complete": False,
                        "grounded_action": {"preconditions": {"held_part": "another_part"}}})
    assert calls[0]["parameter_witnesses"] == [witness]
    assert "parameter_evidence" not in calls[0]
    assert calls[0]["model_complete"] is False
    assert primitive_calls[0]["grounded_action"] == {}


def test_ra_rejection_does_not_call_cca_or_project():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import _get_response_schema

    prepared, session, planner, event = _case()
    session["outline_mode"] = "incremental_validated"
    session["recovery_des_models"][JID]["state_variables"]["held_part"].update(
        type=["string", "null"], domain_complete=False)
    event["expected_end_state"] = {"held_part": "gear_small"}
    schema = _get_response_schema("outline", session)["schema"]
    Draft202012Validator(schema).validate({"thought": "Recorded declaration fixture", "next_transition": event})
    before = deepcopy(session)
    status, _ = _run(prepared, session, planner, event)
    assert status in {"need_revision", "need_context"}
    assert not planner.product_agent.cca_requests
    assert session["accepted_outline_prefix"] == before["accepted_outline_prefix"]
    assert session["symbolic_resources"] == before["symbolic_resources"]
    assert session["outline_monitor_context"] == before["outline_monitor_context"]


def test_ra_public_conflict_survives_feedback_without_committing_state(monkeypatch):
    prepared, session, planner, event = _case()
    before = deepcopy(session)
    conflict = {'field': 'held_part', 'actual': 'gear_small', 'expected': None,
                'condition_operator': 'equals', 'source': {'descriptor_fingerprint': 'saved-model'}}

    async def reject(**kwargs):
        return {'allowed': False, 'feasibility_status': 'INFEASIBLE',
                'constraint_code': 'unsupported_successor_condition', 'reason': 'Recorded field conflict',
                'evidence': {'mismatches': [conflict]}}, {}

    monkeypatch.setattr(validated, 'validate_event', reject)
    status, turn = _run(prepared, session, planner, event)
    assert status == 'need_revision'
    finding = turn['validation_findings'][0]
    assert finding['constraint_code'] == 'unsupported_successor_condition'
    assert finding['evidence']['mismatches'] == [conflict]
    assert not planner.product_agent.cca_requests
    for key in ('accepted_outline_prefix', 'symbolic_resources', 'outline_monitor_context'):
        assert session[key] == before[key]


def test_real_mutex_rejection_does_not_commit_any_projection():
    prepared, session, planner, event = _case()
    for row in prepared["physical_evidence"]["geometry"]["resources"].values():
        row["component_bounds"][0]["bounds"] = [[.9, 1.1], [-.1, .1], [-.1, .1]]
    before = deepcopy(session)
    status, turn = _run(prepared, session, planner, event)
    assert status == "need_revision", turn
    assert turn["validation_stages"][-1]["evidence"]["validation_status"] == "rejected"
    for key in ("accepted_outline_prefix", "symbolic_resources", "outline_monitor_context"):
        assert session[key] == before[key]


def test_missing_geometry_is_not_accepted_even_without_selected_safety():
    prepared, session, planner, event = _case()
    prepared["outline_safety_specifications"] = []
    prepared["physical_evidence"]["geometry"]["resources"].pop("ur5e-4")
    status, _ = _run(prepared, session, planner, event)
    assert status == "need_context"
    assert not planner.product_agent.cca_requests
    assert session["accepted_outline_prefix"] == []


@pytest.mark.parametrize("unresolved", [False, True])
def test_cca_rejection_or_unknown_does_not_append_or_advance_history(unresolved):
    prepared, session, planner, event = _case()
    before = deepcopy(session)
    def reply(result, payload):
        row = result["results"][0]
        row.update(is_safe=False, validation_status="unresolved" if unresolved else "rejected",
                   reason="Required CCA evidence is unavailable" if unresolved else "Rule rejected", checks=[])
        row["monitor_context_after"] = {"must_not_commit": True}
        return result
    planner.product_agent.cca_reply = reply
    status, turn = _run(prepared, session, planner, event)
    assert status == ("need_context" if unresolved else "need_revision")
    assert [row["validator_role"] for row in turn["validation_stages"]] == ["ResourceAgent", "CentralControllerAgent"]
    for key in ("accepted_outline_prefix", "symbolic_resources", "outline_monitor_context"):
        assert session[key] == before[key]


@pytest.mark.parametrize("stage", ["ra", "cca"])
def test_reply_fingerprint_must_match_exact_proposed_event(stage):
    prepared, session, planner, event = _case()
    def corrupt(result, payload):
        result["results"][0]["candidate_fingerprint"] = recovery_validation_fingerprint({"other": "event"})
        return result
    setattr(planner.product_agent, stage + "_reply", corrupt)
    status, _ = _run(prepared, session, planner, event)
    assert status == "need_context"
    assert session["accepted_outline_prefix"] == []
    assert session["symbolic_resources"][JID]["resource_state"] == "failed"
    if stage == "ra":
        assert not planner.product_agent.cca_requests


def test_stale_live_revision_cannot_commit_response():
    prepared, session, planner, event = _case()
    prepared["offline_recovery"] = False
    def changed(result, payload):
        planner.product_agent.environment_runtime.context.revision = 5
        return result
    planner.product_agent.ra_reply = changed
    status, _ = _run(prepared, session, planner, event)
    assert status == "need_context"
    assert session["accepted_outline_prefix"] == []
    assert not planner.product_agent.cca_requests


def test_changed_owner_declarations_invalidate_proposal_before_cca():
    prepared, session, planner, event = _case()
    changed = deepcopy(prepared["recovery_resources"][JID]["recovery_des_model"])
    changed["primitive_catalog"][0]["effects"]["resource_state"] = {"set": "failed"}
    planner.product_agent.owner.recovery_des_model = lambda **kwargs: deepcopy(changed)
    status, turn = _run(prepared, session, planner, event)
    assert status == "need_context"
    assert not session["accepted_outline_prefix"]
    assert not planner.product_agent.cca_requests
    assert "Resource declarations changed after capture" in turn["validation_findings"][0]["reason"]


@pytest.mark.parametrize("new_information", [False, True])
def test_same_proposal_retrieval_retries_only_when_saved_evidence_changes(new_information):
    prepared, session, planner, event = _case()
    if new_information:
        session["projected_physical_evidence"] = {}
    def missing_first(result, payload):
        if len(planner.product_agent.ra_requests) == 1:
            row = result["results"][0]
            row.update(allowed=False, feasibility_status="NEEDS_CONTEXT", context_requests=["/physical_evidence"],
                       reason="Recorded physical context required")
        return result
    planner.product_agent.ra_reply = missing_first
    status, turn = _run(prepared, session, planner, event)
    assert status == ("outline_ready" if new_information else "need_context"), turn
    assert len(planner.product_agent.ra_requests) == (2 if new_information else 1)
    assert all(row["candidates"][0]["task"]["event_name"] == event["event_name"] for row in planner.product_agent.ra_requests)
    assert turn["validation_context_exchanges"][0]["context_requests"] == ["/physical_evidence"]


def test_private_processing_retrieval_counts_new_facts_and_preserves_projections():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import _retrieve

    prepared, session, planner, _event = _case()
    session['primitive_support_semantics'] = 'successor_primitive_union'
    prepared['physical_evidence']['processing_observations'] = {'M2': {'active_job': 'environment_23'}}
    ref = '/physical_evidence/processing_observations/M2'
    turn = {}
    assert asyncio.run(_retrieve(prepared, session, planner, [ref], turn)) is True
    assert turn['validation_context_exchanges'][0]['returned_facts'][ref]['value']['active_job'] == 'environment_23'
    assert asyncio.run(_retrieve(prepared, session, planner, [ref], {})) is False
    session['projected_physical_evidence'].update(provenance={'kind': 'projected'},
        processing_observations={'M2': {'active_job': None, 'job_state': 'paused'}})
    before = deepcopy(session)
    turn = {}
    assert asyncio.run(_retrieve(prepared, session, planner, [ref], turn)) is False
    assert session == before
    returned = turn['validation_context_exchanges'][0]['returned_facts'][ref]
    assert returned['source'] == 'projected' and returned['value']['active_job'] is None
    turn = {}
    assert asyncio.run(_retrieve(prepared, session, planner, [ref + '/commands'], turn)) is False
    assert turn['validation_context_exchanges'][0]['missing_facts'][0]['reason'] == 'Unsupported validator context reference'


@pytest.mark.parametrize("field", ["candidate_events", "transition_suffix", "candidate_index"])
def test_new_contract_rejects_legacy_multi_candidate_payload(field):
    prepared, session, planner, event = _case()
    status, _ = _run(prepared, session, planner, event, **{field: []})
    assert status == "need_revision"
    assert not planner.product_agent.ra_requests
    assert session["accepted_outline_prefix"] == []


@pytest.mark.parametrize("outline_mode", ["incremental", "incremental_candidates_validated", "single_pass"])
def test_versioned_schema_keeps_legacy_suffix_only_for_legacy_attempts(outline_mode):
    declarations = _case()[0]["recovery_resources"][JID]["recovery_des_model"]["state_variables"]
    options = {"outline_context_retrieval": True, "declared_state_variables_by_resource_jid": {JID: declarations}}
    legacy = multi_turn_phase_response_schema("outline", **options)
    current = multi_turn_phase_response_schema("outline", validation_contract_version=2, outline_mode=outline_mode, **options)
    Draft202012Validator.check_schema(current["schema"])
    assert "transition_suffix" in legacy["schema"]["properties"]
    assert "transition_suffix" not in current["schema"]["properties"]
    assert "next_transition" in current["schema"]["properties"]
    assert "anyOf" not in current["schema"]
    assert current["strict"] is False
    assert not uses_outline_evidence({})
    assert uses_outline_evidence({"validation_contract_version": OUTLINE_VALIDATION_CONTRACT_VERSION})


def test_missing_nominal_capture_is_not_equivalent_to_completed_tasks():
    prepared, session, planner, _ = _case()
    session["symbolic_resources"][JID]["resource_state"] = "idle"
    prepared.pop("saved_pending_nodes")
    result = asyncio.run(validate_nominal_continuation(session=session, prepared=prepared, planner=planner))
    assert result["complete"] is False
    assert result["status"] == "NEEDS_CONTEXT"
    assert not planner.product_agent.ra_requests


def test_nominal_guard_feedback_preserves_public_values_and_hides_private_conditions():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import (
        _nominal_task, nominal_continuation_findings,
    )

    prepared, session, _, _ = _case()
    model = prepared["recovery_resources"][JID]["recovery_des_model"]
    event = {"event_name": "finish_cycle", "composition_contract": {"params": {}}}
    model["events"].append(event)
    event["guards"] = {"resource_state": {"equals": "idle"}}
    node = {"id": "saved-nominal", "resource_jid": JID, "function_name": event["event_name"], "params": {}}
    public = []
    task, reason = _nominal_task(node, session, prepared, public)
    assert task is None and "not satisfied" in reason
    result = {"status": "not_ready", "failures": [{"task_id": node["id"], "resource_jid": JID,
        "public_findings": public, "projected_after_task_ids": ["prior-nominal"]}]}
    findings = nominal_continuation_findings(result, prepared)
    failure = findings[1]
    assert failure["evidence"] == {"field": "resource_state", "actual": "failed", "expected": "idle",
        "condition_operator": "equals", "available": True, "projected_after_task_ids": ["prior-nominal"]}
    assert failure["task_id"] == "saved-nominal"
    event["guards"] = {"task_ctx.private": {"equals": "private-value"}}
    model["state_variables"]["task_ctx.private"] = {"scope": "resource", "private": True}
    public.clear()
    _nominal_task(node, session, prepared, public)
    result["status"] = "NEEDS_CONTEXT"
    result["failures"][0]["public_findings"] = public
    rendered = json.dumps(nominal_continuation_findings(result, prepared))
    assert "nominal_task_condition_unavailable" in rendered
    assert "task_ctx" not in rendered and "private-value" not in rendered


@pytest.mark.parametrize("function_name,bindings", [
    ("machine_part", {"resource_id": "M2", "part_name": "RGOCG4-50_Round_4mm", "process": "trim", "result": "circle"}),
    ("move_to_resource", {"resource_id": "KMR", "source_resource": "M2", "target_resource": "Storage"}),
    ("print_part", {"resource_id": "3D Printing Station", "part_name": "KET4_Square_4mm"}),
])
def test_nominal_event_uses_exact_recorded_parameter_binding(function_name, bindings):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import (
        _nominal_event, _nominal_arguments,
    )

    event = {"event_name": function_name, "composition_contract": {"parameter_bindings": deepcopy(bindings)}}
    other = deepcopy(event)
    other["composition_contract"]["parameter_bindings"]["resource_id"] = "M1"
    node = {"id": "saved-nominal", "resource_jid": JID, "function_name": function_name, "params": deepcopy(bindings)}
    prepared = {"recovery_resources": {JID: {"recovery_des_model": {"events": [other, event]}}}}
    before = deepcopy(prepared)
    assert _nominal_event(node, prepared) == event
    args, evidence = _nominal_arguments(node, prepared)
    assert args == bindings
    assert evidence["completion_evidence_status"] == "requires_validation"
    assert prepared == before
    prepared["recovery_resources"][JID]["recovery_des_model"]["events"].append(deepcopy(event))
    with pytest.raises(ValueError, match="one exact declared transition"):
        _nominal_event(node, prepared)


@pytest.mark.parametrize("changed", [{}, {"part_name": "gear_small"}, {"part_name": "gear_small", "confirmed": 1}])
def test_nominal_parameter_binding_preserves_missing_null_and_types(changed):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import _nominal_arguments

    bindings = {"part_name": None, "confirmed": True}
    event = {"event_name": "move_to_resource", "composition_contract": {"parameter_bindings": bindings}}
    prepared = {"recovery_resources": {JID: {"recovery_des_model": {"events": [event]}}}}
    node = {"id": "saved-nominal", "resource_jid": JID, "function_name": "move_to_resource", "params": bindings}
    assert _nominal_arguments(node, prepared)[0] == bindings
    node["params"] = changed
    with pytest.raises(ValueError, match="exact declared transition"):
        _nominal_arguments(node, prepared)


@pytest.mark.parametrize("status", ["not_ready", "NEEDS_CONTEXT"])
def test_nominal_blockers_reach_next_proposal_and_validation_retry(monkeypatch, status):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_nominal_evidence as nominal
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _outline_feedback_for_prompt

    prepared, session, planner, event = _case()
    counterexample = {"rule_id": "SAFE_shared_area_mutex", "ap_values": {"a": True, "b": True},
                      "private_monitor": "private-value"}
    continuation = {"complete": False, "status": status, "missing_task_ids": ["missing-predecessor"],
        "failures": [{"task_id": "saved-nominal", "task": {"resource_jid": JID},
            "projected_after_task_ids": ["prior-nominal"], "reason": "private-value",
            "validation_stages": [{"validator_role": "CentralControllerAgent", "evidence": {
                "is_safe": False, "validation_status": "INFEASIBLE", "private_model": "private-value",
                "checks": [{"is_safe": False, "validation_status": "INFEASIBLE", "counterexample": counterexample}]}}]}]}

    async def continuation_result(**kwargs):
        return deepcopy(continuation)

    monkeypatch.setattr(nominal, "validate_nominal_continuation", continuation_result)
    decision, _ = _run(prepared, session, planner, event)
    assert decision == ("need_context" if status == "NEEDS_CONTEXT" else "need_next_task")
    findings = session["outline_validation_findings"]
    assert findings[0]["evidence"]["missing_task_ids"] == ["missing-predecessor"]
    public = [_outline_feedback_for_prompt(row, {"resource_state"}) for row in findings]
    assert public[1]["evidence"]["safety_checks"][0]["counterexample"] == {
        "rule_id": "SAFE_shared_area_mutex", "ap_values": {"a": True, "b": True}}
    assert "private-value" not in json.dumps(public)
    assert len(session["accepted_outline_prefix"]) == 1
    before = deepcopy(session["accepted_outline_prefix"])
    session["pending_outline_validation"] = {"kind": "nominal", "source_turn_index": 2}
    retry, _ = asyncio.run(validated.retry_pending_event(prepared=prepared, session=session, planner=planner))
    assert retry == decision and session["outline_validation_findings"] == findings
    assert session["accepted_outline_prefix"] == before


def test_nominal_continuation_validates_exact_registered_task_without_appending_it():
    prepared, session, planner, event = _case()
    model = prepared["recovery_resources"][JID]["recovery_des_model"]
    model["state_variables"]["resource_state"]["domain"].append("completed")
    command = {"primitive": "finish_cycle", "params": {}}
    model["primitive_catalog"].append({"name": "finish_cycle", "params": {},
        "preconditions": {"resource_state": {"equals": "idle"}},
        "effects": {"resource_state": {"set": "completed"}}})
    model["events"].append({"event_name": "finish_cycle", "guards": {"resource_state": {"equals": "idle"}},
        "updates": {"resource_state": {"set": "completed"}}, "primitive_support": [command],
        "composition_contract": {"params": {}, "required_params": [], "entry_state": "idle", "entry_guards": [],
            "steps": [{"id": "completion", "executor": "primitive", "op": "finish_cycle", "args": {}, "guards": []}],
            "effects": []}})
    prepared["physical_evidence"]["resource_models"]["ur5e-3"]["primitive_model"]["configuration"]["geometry_unchanged"].append(command)
    prepared["saved_pending_nodes"] = [{"id": "nominal_finish", "function_name": "finish_cycle", "resource_jid": JID,
                                         "params": {}, "predecessors": [], "status": "pending"}]
    status, turn = _run(prepared, session, planner, event)
    assert status == "outline_ready", turn.get("nominal_continuation_evidence", turn)
    assert len(session["accepted_outline_prefix"]) == 1
    assert session["accepted_outline_prefix"][0]["event_name"] == event["event_name"]
    assert session["symbolic_resources"][JID]["resource_state"] == "idle"
    assert session["nominal_continuation_evidence"]["tasks"][0]["task_id"] == "nominal_finish"
    nominal = planner.product_agent.ra_requests[-1]["candidates"][0]["task"]
    assert nominal["required_task_contract"] == "finish_cycle"
    assert nominal["outline_id"] == "nominal_finish"


@pytest.mark.parametrize("known", [False, True])
def test_nominal_completion_distinguishes_missing_custody_from_explicit_null(known):
    prepared, session, planner, _ = _case()
    prepared["outline_safety_specifications"] = []
    prepared["llm_input"]["goal_conditions"][0].update(field="held_part", expected=None)
    if not known:
        session["symbolic_resources"][JID].pop("held_part")
        prepared["recovery_resources"][JID]["recovery_snapshot"].pop("held_part")
    result = asyncio.run(validate_nominal_continuation(session=session, prepared=prepared, planner=planner))
    assert result["complete"] is known, result
    if not known:
        assert result["status"] == "NEEDS_CONTEXT"


def test_unknown_projection_removes_old_aliases_but_explicit_null_is_preserved():
    prepared, session, _, _ = _case()
    session["symbolic_resources"][JID].update(current_location="Buffer For Machined parts", resource_location="Buffer For Machined parts")
    session["symbolic_parts"] = {"gear_small": {"part_holder_resource_jid": JID, "current_holder_resource_jid": JID,
        "holder_resource_jid": JID, "part_location": "3D Printing Station", "location": "3D Printing Station"}}
    task = {"resource_jid": JID, "part_name": "gear_small"}
    ra = {"primitive_support": {"projected_snapshot": {"held_part": None}, "projected_part_context": {"held": None},
          "terminal_state": {"unknown": {"resource": ["resource_location"], "part": ["part_holder_resource_jid", "part_location"]}}}}
    ra["geometry_evidence"] = {"status": "projected", "successor_evidence": deepcopy(prepared["physical_evidence"])}
    validated._project(task, ra, {"monitor_context_after": {}}, session)
    assert session["symbolic_resources"][JID]["held_part"] is None
    assert not {"current_location", "resource_location"} & session["symbolic_resources"][JID].keys()
    assert not {"part_location", "location", "part_holder_resource_jid", "current_holder_resource_jid", "holder_resource_jid"} & session["symbolic_parts"]["gear_small"].keys()


def test_projected_fk_pose_and_unknown_fields_do_not_revert_to_saved_observations():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_context import (
        _resource_fact, public_resource_snapshot,
    )

    prepared, session, _, event = _case()
    observed = [0., 0., 0., 0., 0., 0., 1.]
    projected = [2., 0., 0., 0., 0., 0., 1.]
    saved = prepared["recovery_resources"][JID]["recovery_snapshot"]
    saved.update(current_pose=observed, current_pose_captured_at="capture", resource_location="3D Printing Station")
    session["symbolic_resources"][JID].update(deepcopy(saved))
    session["projected_physical_evidence"] = deepcopy(prepared["physical_evidence"])
    successor = deepcopy(prepared["physical_evidence"])
    successor["snapshot"]["resources"]["ur5e-3"]["current_pose"] = projected
    ra = {"primitive_support": {"projected_snapshot": {"resource_state": "idle", "held_part": None},
        "terminal_state": {"unknown": {"resource": ["resource_location"]}}},
        "geometry_evidence": {"status": "projected", "successor_evidence": successor}}
    validated._project(event, ra, {"monitor_context_after": {}}, session)
    session["accepted_outline_prefix"] = [deepcopy(event)]
    current = public_resource_snapshot(prepared, session, JID)
    assert current["current_pose"] == projected
    assert "resource_location" not in current
    assert "current_pose_captured_at" not in current
    value, source = _resource_fact(prepared, session, JID, "current_pose", f"/resources/{JID}/current_pose")
    assert value == projected and source == "projected_state"
    assert saved["current_pose"] == observed


def test_active_cca_history_cannot_be_reset_prospectively_without_compatible_context():
    prepared, session, planner, event = _case()
    status, _ = _run(prepared, session, planner, event)
    assert status == "outline_ready"
    payload = deepcopy(planner.product_agent.cca_requests[0])
    safety = payload["candidates"][0]["safety_input"]
    safety["monitor_history"] = {"active": True, "physical_monitor": {"history": [{"observed": True}]}}
    result = validate_safety_outline_payload(payload)["results"][0]
    assert result["is_safe"] is False
    assert result["validation_status"] == "unresolved"
    assert result["monitor_context_after"] == result["monitor_context_before"] == {}
    assert "prospective reset is forbidden" in result["reason"]


def test_validated_outline_cannot_enter_legacy_diagnostic_execution():
    from cais_spade_llm.recovery_framework.saved_recovery_execution import validate_saved_proposal

    with pytest.raises(ValueError, match="does not authorize execution"):
        asyncio.run(validate_saved_proposal(SimpleNamespace(), {"validation_contract_version": 2}, {}))


def _capability_rejection_record(task, snapshot, model, physical, facts, capabilities):
    from cais_spade_llm.resources.recovery_geometry import outline_event_binding

    reason = "pose is outside gripper_reach: xy_radius=2.0000 m, max=0.7000 m"
    return {"status": "rejected", "validation_status": "rejected", "physical_capability_rejected": True,
        "capability_result": {"feasibility_status": "INFEASIBLE", "reason": reason, "checks": [{"motion_index": 0,
            "feasibility_status": "INFEASIBLE",
            "target_index": 0, "checked_pose": {"x": 2., "y": 0., "z": 1.}, "frame": "world",
            "gripper_reach": capabilities["gripper_reach"], "reason": reason}]},
        "motion_requests": [{"mode": "cartesian", "waypoints": [{"x": 2., "y": 0., "z": 1.}]}],
        "task": deepcopy(task), "reason": reason,
        "event_record_binding": outline_event_binding(task, snapshot, model, physical),
        "resource_evidence_fingerprint": recovery_validation_fingerprint(facts),
        "static_capabilities_fingerprint": recovery_validation_fingerprint(capabilities)}


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
    assert public["reason"] == reason
    assert public["evidence"]["capability_result"]["checks"][0]["reason"] == reason
    assert "secret" not in json.dumps(public)
    assert finding == original
    finding["reason"] = "private proof suggests a particular event"
    assert "private proof" not in json.dumps(_outline_feedback_for_prompt(finding, set()))
    check["reason"] += "; private proof suggests a particular event"
    assert "capability_result" not in _outline_feedback_for_prompt(finding, set())["evidence"]


@pytest.mark.parametrize("defect", [None, "checks", "target", "task", "facts", "capabilities", "scene"])
def test_owner_capability_rejection_is_recalculated_from_bound_inputs(monkeypatch, defect):
    from cais_spade_llm.recovery_framework.outline_validation import rejected_outline_capability_projection
    from cais_spade_llm.resources import resource_safety_preparation

    prepared, session, planner, event, _ = _union_case()
    _run(prepared, session, planner, event)
    task = session["pending_outline_validation"]["task"]
    entry = prepared["recovery_resources"][JID]
    scene = prepared["physical_evidence"]
    facts, capabilities = {"source": {"capture": "recorded"}}, {"gripper_reach": {"max_xy_radius_m": .7}}
    original = _capability_rejection_record(task, entry["recovery_snapshot"], entry["recovery_des_model"], scene, facts, capabilities)
    calls = []
    def calculate(**kwargs):
        calls.append(kwargs)
        return deepcopy(original)
    monkeypatch.setattr(resource_safety_preparation, "prepare_outline_event_preview_request", calculate)
    record = deepcopy(original)
    if defect == "checks":
        record["capability_result"]["checks"][0]["reason"] = "Fabricated rejection"
    elif defect == "target":
        record["motion_requests"][0]["waypoints"][0]["x"] = 3.
    elif defect == "task":
        record["task"]["expected_end_state"]["resource_state"] = "failed"
    elif defect == "facts":
        facts["source"]["capture"] = "changed"
    elif defect == "capabilities":
        capabilities["gripper_reach"]["max_xy_radius_m"] = 3.
    elif defect == "scene":
        scene["snapshot"]["resources"]["ur5e-3"]["current_pose"][0] += .01
    kwargs = dict(record=record, task=task, recovery_snapshot=entry["recovery_snapshot"],
        recovery_des_model=entry["recovery_des_model"], physical_evidence=scene,
        resource_evidence=facts, static_capabilities=capabilities)
    if defect:
        with pytest.raises(ValueError, match="Rejected owner capability"):
            rejected_outline_capability_projection(**kwargs)
    else:
        result = rejected_outline_capability_projection(**kwargs)
        assert len(calls) == 1
        assert calls[0] == {key: value for key, value in kwargs.items() if key != "record"}
        assert result["status"] == "rejected" and result["observations"] == []
        assert result["execution_authorized"] is False
        assert record == original


def test_workflow_owner_capability_rejection_preserves_ra_and_returns_factual_feedback(tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery
    from cais_spade_llm.resources import resource_safety_preparation
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _render_outline_prompt

    prepared, session, planner, event, _ = _union_case()
    _run(prepared, session, planner, event)
    task = session["pending_outline_validation"]["task"]
    entry = prepared["recovery_resources"][JID]
    facts = {"source": {"run_id": "captured-run", "captured_at": "2026-10-10T04:14:18+00:00",
                        "revision": 4, "kind": "recorded_owner_state", "private_model": "private source details"}}
    capabilities = {"gripper_reach": {"max_xy_radius_m": .7, "frame": "world",
                    "origin_pose": {"x": 0., "y": 0., "z": 0.}, "z_min_m": .85,
                    "z_max_m": 1.6, "tolerance_m": .01}}
    owner = planner.product_agent.owner
    owner.static_capabilities = capabilities
    owner.get_recovery_resource_evidence = lambda: deepcopy(facts)
    record = _capability_rejection_record(task, entry["recovery_snapshot"], entry["recovery_des_model"],
        prepared["physical_evidence"], facts, capabilities)
    record["capability_result"]["private_evaluator_data"] = "private proof"
    monkeypatch.setattr(resource_safety_preparation, "prepare_outline_event_preview_request", lambda **_: deepcopy(record))
    prepared["multi_turn_session_state"] = session
    recovery = SimpleNamespace(_stopped=False, _generation=1, _prepared=prepared,
        offline=True, _offline_cancel=threading.Event(), _observations={},
        runtime=SimpleNamespace(source_observation_path=None),
        agent=SimpleNamespace(process_planner=planner, resources={JID: owner}),
        _state={"artifact_directory": str(tmp_path)})
    recovery._publish = lambda **values: recovery._state.update(values)
    recovery._offline_work = PartSlippageRecovery._offline_work.__get__(recovery)
    recovery._offline_preview_observations = PartSlippageRecovery._offline_preview_observations.__get__(recovery)
    before = deepcopy(session)
    asyncio.run(PartSlippageRecovery.retry_outline_validation(recovery, record))
    current = prepared["multi_turn_session_state"]
    assert recovery._state["status"] == "outline_ready"
    assert current["projected_physical_evidence"]["rejected_capability_evidence"] == [record]
    for field in ("accepted_outline_prefix", "symbolic_resources", "symbolic_parts", "outline_monitor_context"):
        assert current[field] == before[field]
    finding = current["outline_validation_findings"][0]
    assert finding["constraint_code"] == "physical_capability_rejected"
    assert finding["constraint_owner"] == "resource"
    assert finding["feasibility_status"] == "FEASIBLE"
    assert finding["validation_status"] == "rejected"
    assert finding["reason"] == record["reason"]
    prompt = _render_outline_prompt({"session_state": current, "llm_input": prepared["llm_input"],
                                    "recovery_resources": prepared["recovery_resources"]})
    assert record["reason"] in prompt
    for field in ('"x": 2.0', '"max_xy_radius_m": 0.7', '"tolerance_m": 0.01', '"frame": "world"',
                  '"revision": 4', '"run_id": "captured-run"', "2026-10-10T04:14:18+00:00",
                  f"/resources/{JID}/static_capabilities", "calculated owner motion target"):
        assert field in prompt
    for private in ("private_evaluator_data", "private proof", "private_model", "private source details", "motion_requests"):
        assert private not in prompt
    saved = json.loads(next((tmp_path / "recovery_outline/turns").glob("*validation*.json")).read_text())
    stages = {row["validator_role"]: row["evidence"] for row in saved["validation_stages"]}
    assert stages["ResourceAgent"]["geometry_evidence"]["physical_capability_rejected"] is True
    assert stages["CentralControllerAgent"]["formal_safety_status"] == "not_checked"
    assert saved["request_sent"] is False and saved["decision"] == "need_revision"


def test_owner_effect_rejection_returns_revision_without_appending_or_checking_safety():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _outline_feedback_for_prompt

    prepared, session, planner, event, _ = _union_case()
    owner = planner.product_agent.owner
    owner.project_recovery_outline_event = lambda **_: {
        "status": "rejected", "validation_status": "rejected", "owner_effect_rejected": True,
        "reason": "Every declared supporting alternative conflicts with proposed part_state",
        "owner_effect_result": {"feasibility_status": "INFEASIBLE", "conflicts": [{
            "field": "part_state", "requested": "in_gripper", "derived": "in_transit",
            "declaration_fingerprint": "private declaration", "parameter_bindings": {"private": "private binding"}}]},
        "observations": [], "interval_complete": False, "execution_authorized": False,
    }
    before = deepcopy(session)
    decision, turn = _run(prepared, session, planner, event)
    assert decision == "need_revision"
    assert session["status"] == "paused_after_outline_turn"
    assert "pending_outline_validation" not in session
    for field in ("accepted_outline_prefix", "symbolic_resources", "symbolic_parts", "outline_monitor_context"):
        assert session[field] == before[field]
    stages = {row["validator_role"]: row["evidence"] for row in turn["validation_stages"]}
    assert stages["ResourceAgent"]["feasibility_status"] == "FEASIBLE"
    cca = stages["CentralControllerAgent"]
    assert cca["formal_safety_status"] == "not_checked" and cca["checks"] == []
    assert cca["monitor_context_before"] == cca["monitor_context_after"] == before["outline_monitor_context"]
    assert cca["execution_authorized"] is False
    finding = session["outline_validation_findings"][0]
    assert finding["constraint_owner"] == "resource"
    assert finding["constraint_code"] == "owner_effect_rejected"
    public = _outline_feedback_for_prompt(finding, {"part_state"})
    assert public["evidence"]["owner_effect_result"]["conflicts"] == [
        {"field": "part_state", "requested": "in_gripper", "derived": "in_transit"}]
    assert "private" not in json.dumps(public)
    assert "not observations" in public["reason"]


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


def test_mating_counterexample_feedback_retains_public_bounds_without_claiming_sampled_collision(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _outline_feedback_for_prompt, _render_outline_prompt,
    )

    prepared, session, planner, event = _case()
    counterexample = {'segment': 1, 'time_from_start': .25, 'fraction': '1/2',
        'joint_intervals': {'private_joint': [0., 1.]}, 'part_transform_bounds': ['private_matrix'],
        'violated_conditions': [{'condition': 'target_origin_pose.z', 'relation': '<',
            'value_bounds': [-.002, -.001], 'limit_bounds': [-.0005, -.0005], 'private': 'omit'}]}

    async def reject(**kwargs):
        return {'allowed': True, 'feasibility_status': 'FEASIBLE'}, {
            'is_safe': False, 'validation_status': 'rejected', 'physical_motion_rejected': True,
            'continuous_mating_rejected': True, 'reason': 'The interpolated part leaves the declared mating corridor',
            'checks': [], 'mating_rejection': {'status': 'rejected', 'counterexample': deepcopy(counterexample)}}

    monkeypatch.setattr(validated, 'validate_event', reject)
    before = deepcopy(session)
    decision, _ = _run(prepared, session, planner, event)
    assert decision == 'need_revision'
    finding = session['outline_validation_findings'][0]
    assert finding['constraint_code'] == 'continuous_mating_rejected'
    public = _outline_feedback_for_prompt(finding, {'resource_state'})
    assert public['evidence']['mating_counterexample']['time_from_start'] == .25
    assert public['evidence']['mating_counterexample']['violated_conditions'] == [{
        'condition': 'target_origin_pose.z', 'relation': '<', 'value_bounds': [-.002, -.001], 'limit_bounds': [-.0005, -.0005]}]
    prompt = _render_outline_prompt({'session_state': session, 'llm_input': prepared['llm_input'],
        'recovery_resources': prepared['recovery_resources']})
    assert 'leaves the recorded mating corridor' in prompt
    assert 'intersects recorded geometry' not in prompt
    assert 'private_joint' not in prompt and 'private_matrix' not in prompt and '"private"' not in prompt
    for key in ('accepted_outline_prefix', 'symbolic_resources', 'symbolic_parts', 'outline_monitor_context'):
        assert session[key] == before[key]


def test_nominal_mating_counterexample_keeps_exact_public_conditions():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import nominal_continuation_findings

    prepared, _, _, task = _case()
    proof = {'counterexample': {'segment': 2, 'time_from_start': .3, 'fraction': '3/4',
        'joint_intervals': {'private_joint': [1., 2.]}, 'violated_conditions': [{
            'condition': 'yaw_tolerance_rad', 'relation': '>', 'value_bounds': [.01, .02], 'limit_bounds': [.005, .005]}]}}
    result = {'complete': False, 'status': 'not_ready', 'failures': [{'task_id': 'nominal', 'task': task,
        'validation_stages': [{'validator_role': 'CentralControllerAgent', 'evidence': {
            'is_safe': False, 'validation_status': 'rejected', 'physical_motion_rejected': True,
            'continuous_mating_rejected': True, 'mating_rejection': proof}}]}]}
    findings = nominal_continuation_findings(result, prepared)
    finding = next(row for row in findings if row['constraint_code'] == 'continuous_mating_rejected')
    assert finding['evidence']['mating_counterexample']['violated_conditions'] == proof['counterexample']['violated_conditions']
    assert 'leaves the recorded mating corridor' in finding['reason']
    assert 'private_joint' not in json.dumps(findings)


def test_endpoint_rejection_feedback_keeps_contact_facts_separate_from_motion(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _outline_feedback_for_prompt
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import nominal_continuation_findings

    prepared, session, planner, event = _case()
    proof = {'changed_object_ids': ['part/link/collision'], 'world_object_ids': ['private_unrelated_inventory'],
        'request_binding': {'private': 'model'}, 'robot_contacts': [], 'checked_pairs': [{
            'body_1': 'part/link/collision', 'body_2': 'table/link/collision', 'allowed_support_tangency': True,
            'contacts': [{'shape_1': 0, 'shape_2': 0, 'depth': .001, 'position': [0., 0., 1.]}]}],
        'support_pairs': [{'part_object_id': 'part/link/collision', 'support_object_id': 'table/link/collision',
            'minimum_signed_distance': -.001}]}
    cca = {'is_safe': False, 'validation_status': 'rejected', 'physical_motion_rejected': True,
        'physical_endpoint_rejected': True, 'reason': 'The endpoint has unsupported contact',
        'checks': [], 'endpoint_rejection': proof}

    async def reject(**kwargs):
        return {'allowed': True, 'feasibility_status': 'FEASIBLE'}, deepcopy(cca)

    monkeypatch.setattr(validated, 'validate_event', reject)
    before = deepcopy(session)
    decision, _ = _run(prepared, session, planner, event)
    assert decision == 'need_revision'
    finding = session['outline_validation_findings'][0]
    assert finding['constraint_code'] == 'physical_endpoint_rejected'
    public = _outline_feedback_for_prompt(finding, {'resource_state'})
    assert public['evidence']['endpoint_counterexample']['world_contacts'][0]['contacts'][0]['depth'] == .001
    assert _outline_feedback_for_prompt(public, {'resource_state'}) == public
    assert 'private' not in json.dumps(public)
    assert 'deposited-part endpoint' in public['reason']
    assert 'intersects recorded geometry' not in public['reason']
    for key in ('accepted_outline_prefix', 'symbolic_resources', 'symbolic_parts', 'outline_monitor_context'):
        assert session[key] == before[key]
    nominal = {'complete': False, 'status': 'not_ready', 'failures': [{'task_id': 'nominal', 'task': event,
        'validation_stages': [{'validator_role': 'CentralControllerAgent', 'evidence': cca}]}]}
    findings = nominal_continuation_findings(nominal, prepared)
    nominal_finding = next(row for row in findings if row['constraint_code'] == 'physical_endpoint_rejected')
    assert nominal_finding['evidence']['endpoint_counterexample'] == public['evidence']['endpoint_counterexample']


@pytest.mark.parametrize("missing_geometry", [False, True])
def test_event_transaction_keeps_archives_without_copying_them(missing_geometry):
    class Archive(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("Archived validation attempts must not enter the working transaction")

    prepared, session, planner, event = _case()
    if missing_geometry:
        prepared["physical_evidence"].pop("geometry")
    expected_session = deepcopy(session)
    expected = _run(prepared, expected_session, planner, deepcopy(event))
    expected_requests = deepcopy((planner.product_agent.ra_requests, planner.product_agent.cca_requests))
    planner.product_agent.ra_requests.clear()
    planner.product_agent.cca_requests.clear()
    archives = {key: Archive(recorded=[key]) for key in
        ("turns", "nominal_continuation_evidence", "pending_outline_validation")}
    session.update(archives)
    before = deepcopy(prepared)
    result = _run(prepared, session, planner, event)
    assert result == expected
    assert (planner.product_agent.ra_requests, planner.product_agent.cca_requests) == expected_requests
    assert prepared == before
    assert session["turns"] is archives["turns"]
    for key, value in expected_session.items():
        assert session[key] == value
    for key, archive in archives.items():
        assert archive == {"recorded": [key]}
    assert (result[0] == "outline_ready") is not missing_geometry


def test_event_working_mutations_leave_rejected_source_state_unchanged(monkeypatch):
    prepared, session, planner, event = _case()
    session["projected_physical_evidence"] = deepcopy(prepared["physical_evidence"])
    session["validation_resource_context"] = {JID: {"resource_evidence": {"recorded": [1]}}}
    before = deepcopy(session)

    async def reject(*, session, **kwargs):
        session["symbolic_resources"][JID]["held_part"] = "gear_small"
        session["projected_physical_evidence"]["snapshot"]["resources"]["ur5e-4"]["held_part"] = "gear_small"
        session["validation_resource_context"][JID]["resource_evidence"]["recorded"].append(2)
        return {"allowed": False, "feasibility_status": "INFEASIBLE", "reason": "Recorded conflict"}, {}

    monkeypatch.setattr(validated, "validate_event", reject)
    assert _run(prepared, session, planner, event)[0] == "need_revision"
    for key, value in before.items():
        assert session[key] == value


def test_retry_keeps_pending_evidence_and_detaches_only_authored_response(monkeypatch):
    class Archive(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("The retry dispatcher must not copy pending physical evidence")

    prepared, session, planner, event = _case()
    response = {"thought": "Recorded facts", "next_transition": deepcopy(event)}
    original_response = deepcopy(response)
    pending = {"kind": "event", "response": response, "source_turn_index": 7,
        "task": Archive(recorded="task"), "projected_physical_evidence": Archive(recorded="scene"),
        "recovery_snapshot": Archive(recorded="snapshot")}
    session["pending_outline_validation"] = pending

    async def retry(*, session, response, **kwargs):
        assert session["pending_outline_validation"] is pending
        assert response == original_response and response is not pending["response"]
        response["next_transition"]["expected_end_state"]["resource_state"] = "changed"
        session.pop("pending_outline_validation")
        return "need_revision", {"checked": True}

    monkeypatch.setattr(validated, "handle_validated_event", retry)
    result = asyncio.run(validated.retry_pending_event(prepared=prepared, session=session, planner=planner))
    assert result == ("need_revision", {"checked": True, "validation_only": True,
        "source_turn_index": 7, "request_sent": False})
    assert pending["response"] == original_response
    assert pending["projected_physical_evidence"] == {"recorded": "scene"}


@pytest.mark.parametrize("scope,field,row,expected", [
    ("part", "state", {"current_state": "misplaced", "part_state": "assembled"}, "assembled"),
    ("part", "location", {"current_location": None, "part_location": "assembly_board-v1"}, "assembly_board-v1"),
    ("part", "holder", {"current_holder_resource_jid": JID, "part_holder_resource_jid": None}, None),
    ("resource", "state", {"current_state": "failed", "resource_state": "idle"}, "idle"),
    ("resource", "location", {"current_location": "assembly_board-v1", "resource_location": "Buffer For Machined parts"}, "Buffer For Machined parts"),
])
def test_version2_completion_reads_supported_projected_fields_without_changing_capture(scope, field, row, expected):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import _continuation_condition_satisfied

    prepared, session, _, _ = _case()
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

    prepared, session, _, _ = _case()
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

    prepared, session, _, _ = _case()
    session["symbolic_parts"]["gear_small"] = {
        "current_state": "misplaced", "part_state": "assembled", "current_location": None,
        "part_location": "assembly_board-v1", "current_holder_resource_jid": None,
    }
    condition = {"entity_kind": "part", "entity": "gear_small", "field": field, "expected": expected}
    assert _continuation_condition_satisfied(condition, session_state=session, prepared_recovery_request=prepared) is accepted


def test_recorded_product_requirements_remain_completion_obligations_without_legacy_goals():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import _completion_goals

    prepared = {'nominal_continuation_semantics': 'recorded_unfinished_tasks', 'physical_evidence': {
        'requirements': {'part': [{'processesToComplete': [{'process': 'trim', 'result': 'square'}]}]},
        'products': {'part': {'processCompleted': []}}}}
    assert _completion_goals({}, prepared, [])['status'] == 'not_ready'
    session = {'projected_physical_evidence': {'products': {'part': {
        'processCompleted': [{'process': 'trim', 'result': 'square'}]}}}}
    assert _completion_goals(session, prepared, []) is None
    assert prepared['physical_evidence']['products']['part']['processCompleted'] == []
    prepared['physical_evidence']['requirements'] = {}
    assert _completion_goals(session, prepared, [])['status'] == 'NEEDS_CONTEXT'


def test_supported_gear_projection_satisfies_final_goals_without_rewriting_recorded_fields():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import _completion_goals

    prepared, session, _, event = _case()
    session["symbolic_parts"]["gear_small"] = {"current_state": "misplaced", "current_location": None}
    task = {**event, "part_name": "gear_small", "expected_end_state": {
        "part_state": "assembled", "part_location": "assembly_board-v1"}}
    ra = {"primitive_support": {"semantics": "successor_primitive_union"}, "geometry_evidence": {
        "status": "projected", "successor_evidence": deepcopy(prepared["physical_evidence"]),
        "projected_resource_snapshot": {"resource_state": "idle", "held_part": None},
        "part_updates": {"part_state": "assembled", "part_location": "assembly_board-v1"}}}
    goals = [{"entity_kind": "part", "entity": "gear_small", "field": field, "expected": expected}
             for field, expected in (("state", "assembled"), ("location", "assembly_board-v1"))]
    assert _completion_goals(session, prepared, goals)["status"] == "not_ready"
    original = deepcopy(prepared)
    validated._project(task, ra, {"monitor_context_after": {}}, session)
    assert _completion_goals(session, prepared, goals) is None
    assert session["symbolic_parts"]["gear_small"]["current_state"] == "misplaced"
    assert session["symbolic_parts"]["gear_small"]["current_location"] is None
    assert prepared == original


@pytest.mark.parametrize("scope,field,row,expected,status", [
    ("part", "part_state", {"current_state": "assembled"}, "assembled", "NEEDS_CONTEXT"),
    ("part", "current_state", {"part_state": "assembled"}, "assembled", "NEEDS_CONTEXT"),
    ("part", "part_holder_resource_jid", {"current_holder_resource_jid": None}, None, "NEEDS_CONTEXT"),
    ("part", "current_holder_resource_jid", {"part_holder_resource_jid": None}, None, "NEEDS_CONTEXT"),
    ("resource", "resource_state", {"current_state": "idle"}, "idle", "NEEDS_CONTEXT"),
    ("resource", "current_state", {"resource_state": "idle"}, "idle", "NEEDS_CONTEXT"),
    ("part", "state", {"current_state": "assembled"}, "assembled", None),
    ("part", "state", {"part_state": None, "current_state": "assembled"}, "assembled", "not_ready"),
    ("part", "location", {"part_location": None, "current_location": "assembly_board-v1"}, None, None),
    ("resource", "phase", {}, None, "NEEDS_CONTEXT"),
    ("resource", "phase", {"phase": True}, 1, "not_ready"),
])
def test_version2_completion_distinguishes_missing_explicit_facts_from_conflicts(scope, field, row, expected, status):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import _completion_goals

    prepared, session, _, _ = _case()
    entity = "gear_small" if scope == "part" else JID
    session["symbolic_parts" if scope == "part" else "symbolic_resources"][entity] = deepcopy(row)
    condition = {"entity_kind": scope, "entity": entity, "field": field, "expected": expected}
    before = deepcopy((prepared, session))
    result = _completion_goals(session, prepared, [condition])
    assert (result.get("status") if result else None) == status
    if status == "NEEDS_CONTEXT":
        assert result["missing_conditions"] == [condition]
    elif status == "not_ready":
        assert result["unmet_conditions"] == [condition]
    assert (prepared, session) == before
