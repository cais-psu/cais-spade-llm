from __future__ import annotations

"""Contract-level regression coverage for generalized ordered composition."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cais_spade_llm.resources.recovery_composition import (
    composition_program_error, composition_start_views, resolve_composition_value,
    validate_recovery_primitive_composition,
)


def feasible(**kwargs):
    return {"allowed": True, "feasibility_status": "FEASIBLE", "params": kwargs["params"]}


@pytest.fixture
def contract():
    return {"task": {"outline_id": "restore", "event_name": "restore", "resource_jid": "owner",
                "part_name": "P", "expected_start_state": {"mode": "fault", "quality": "pending"},
                "expected_end_state": {"mode": "ready", "quality": "checked"}},
        "recovery_des_model": {"state_variables": {"mode": {"scope": "resource"},
                                                    "quality": {"scope": "part"}}},
        "recovery_snapshot": {"mode": "fault"}, "part_context": {"quality": "pending"},
        "grounding_context": {"limit": 3}, "validate_primitive": feasible,
        "primitive_catalog": [
            {"name": "reset", "params": {}, "preconditions": {"mode": {"equals": "fault"}},
             "effects": {"mode": {"set": "idle"}}},
            {"name": "qualify", "params": {"count": {"type": "integer", "minimum": 1, "maximum": 3}},
             "required_params": ["count"], "preconditions": {"mode": {"equals": "idle"}},
             "effects": {"mode": {"set": "ready"}, "quality": {"set": "checked"}}}],
        "primitive_steps": [{"primitive": "reset", "params": {}},
                            {"primitive": "qualify", "params": {"count": {"context_ref": "/limit"}}}]}


@pytest.mark.parametrize("resource_type", ["machine", "Conveyor", "kmr", "unfamiliar_resource"])
def test_owner_fields_and_product_effects_without_robot_state(contract, resource_type):
    contract["recovery_snapshot"]["resource_type"] = resource_type
    before = deepcopy(contract)
    result = validate_recovery_primitive_composition(**contract)
    assert result["feasibility_status"] == "FEASIBLE"
    assert result["derived_end_state"] == {"mode": "ready", "quality": "checked"}
    assert result["trace"][1]["start_snapshot"]["mode"] == "idle"
    assert result["trace"][1]["resolved_params"] == {"count": 3}
    assert result["execution_authorized"] is False
    assert contract == before


@pytest.mark.parametrize("change, status, step", [
    ("order", "INFEASIBLE", 0), ("reference", "NEEDS_CONTEXT", 1),
    ("params", "INFEASIBLE", 1), ("extra_param", "INFEASIBLE", 1),
    ("identity", "INFEASIBLE", 0), ("effects", "INFEASIBLE", None),
    ("missing_effects", "NEEDS_CONTEXT", 1), ("missing_contract", "NEEDS_CONTEXT", None),
])
def test_rejections_do_not_substitute_requested_postconditions(contract, change, status, step):
    if change == "order":
        contract["primitive_steps"].reverse()
    elif change == "reference":
        contract["grounding_context"] = {}
    elif change == "params":
        contract["grounding_context"]["limit"] = 20
    elif change == "extra_param":
        contract["primitive_steps"][1]["params"]["other"] = 1
    elif change == "identity":
        contract["primitive_steps"][0]["primitive"] = "RESET"
    elif change == "effects":
        contract["primitive_catalog"][1]["effects"].pop("quality")
    elif change == "missing_effects":
        contract["primitive_catalog"][1].pop("effects")
    else:
        contract["recovery_des_model"] = {}
    result = validate_recovery_primitive_composition(**contract)
    assert result["feasibility_status"] == status
    assert result["findings"][0]["step_index"] == step
    assert result["allowed"] is False
    assert result["derived_end_state"] != contract["task"]["expected_end_state"]


def test_null_is_a_fact_and_missing_is_unknown(contract):
    contract["task"]["expected_start_state"]["mode"] = None
    contract["primitive_catalog"][0]["preconditions"]["mode"] = {"equals": None}
    contract["recovery_snapshot"]["mode"] = None
    assert validate_recovery_primitive_composition(**contract)["allowed"]
    contract["recovery_snapshot"].pop("mode")
    assert validate_recovery_primitive_composition(**contract)["feasibility_status"] == "NEEDS_CONTEXT"


def test_parameter_references_use_intermediate_part_effects(contract):
    contract["grounding_context"]["part"] = {"quality": "pending"}
    contract["primitive_catalog"][0]["effects"]["quality"] = {"set": "measured"}
    contract["primitive_catalog"][1]["params"]["quality"] = {"const": "measured"}
    contract["primitive_steps"][1]["params"]["quality"] = {"context_ref": "/part/quality"}
    assert validate_recovery_primitive_composition(**contract)["feasibility_status"] == "FEASIBLE"


@pytest.mark.parametrize("reference", ["/resource/measurement", "/resources/owner/measurement"])
def test_unknown_effect_clears_previously_available_reference(contract, reference):
    contract["recovery_des_model"]["state_variables"]["measurement"] = {"scope": "resource"}
    contract["recovery_snapshot"]["measurement"] = 3
    contract["grounding_context"]["resources"] = {"owner": {"measurement": 3}}
    contract["primitive_catalog"][0]["effects"]["measurement"] = {"set_unknown": True}
    contract["primitive_steps"][1]["params"]["count"] = {"context_ref": reference}
    result = validate_recovery_primitive_composition(**contract)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["findings"][0]["step_index"] == 1


def test_literal_identifiers_are_never_interpreted_as_references():
    assert resolve_composition_value("/owner/null", {}) == "/owner/null"
    assert resolve_composition_value({"context_ref": "/A B/value"}, {"A B": {"value": None}}) is None
    with pytest.raises(KeyError):
        resolve_composition_value({"context_ref": "/a b/value"}, {"A B": {"value": None}})


def test_declared_macro_effects_are_not_applied_to_incomplete_program(contract):
    contract["recovery_des_model"]["events"] = [{"event_name": "complete",
        "composition_contract": {"steps": [{"op": "extra"}]},
        "updates": {"quality": {"set": "checked"}}}]
    contract["primitive_catalog"][1]["effects"].pop("quality")
    result = validate_recovery_primitive_composition(**contract)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["projected_part_context"]["quality"] == "pending"


def _prepared(contract):
    return {"composition_validation_version": 1,
        "grounding_context": {"parts": {"P": deepcopy(contract["part_context"])}},
        "recovery_resources": {"owner": {"recovery_snapshot": deepcopy(contract["recovery_snapshot"]),
            "recovery_des_model": deepcopy(contract["recovery_des_model"]),
            "execution_primitive_catalog": deepcopy(contract["primitive_catalog"])}}}


def _row(contract):
    return {**deepcopy(contract["task"]), "primitive_steps": deepcopy(contract["primitive_steps"]),
            "composition_validation": validate_recovery_primitive_composition(**contract)}


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "wrong_owner", "wrong_program", "wrong_contract"])
def test_exact_coverage_and_current_evidence_bindings(contract, mutation):
    prepared, program = _prepared(contract), [_row(contract)]
    assert composition_program_error([contract["task"]], program, prepared=prepared, session={}) is None
    if mutation == "duplicate":
        program.append(deepcopy(program[0]))
    elif mutation == "missing":
        program = []
    elif mutation == "wrong_owner":
        program[0]["resource_jid"] = "another"
    elif mutation == "wrong_program":
        program[0]["primitive_steps"].reverse()
    else:
        prepared["recovery_resources"]["owner"]["recovery_des_model"]["revision"] = 2
    assert composition_program_error([contract["task"]], program, prepared=prepared, session={})


def test_other_resources_consume_validated_product_effects(contract):
    prepared = _prepared(contract)
    prepared["recovery_resources"]["other"] = {"recovery_snapshot": {"phase": "waiting"}}
    session = {"accepted_primitive_program": [_row(contract)]}
    resource, part = composition_start_views(session=session, prepared=prepared,
        task={"resource_jid": "other", "part_name": "P"})
    assert resource == {"phase": "waiting"}
    assert part["quality"] == "checked"
    session["accepted_primitive_program"][0]["composition_validation"]["feasibility_status"] = "INFEASIBLE"
    _, part = composition_start_views(session=session, prepared=prepared, task={"resource_jid": "other", "part_name": "P"})
    assert part["quality"] == "pending"


def test_generation_routes_one_event_at_a_time_and_preserves_source(contract, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_composition import generate_next_composition_event

    prepared = _prepared(contract)
    second = deepcopy(contract)
    second["task"].update(outline_id="next", resource_jid="other",
        expected_start_state={"mode": "fault", "quality": "checked"})
    second["part_context"]["quality"] = "checked"
    prepared["recovery_resources"]["other"] = deepcopy(prepared["recovery_resources"]["owner"])
    events = [contract["task"], second["task"]]
    session = {"accepted_outline_prefix": deepcopy(events)}
    calls = []
    async def generate(**kwargs):
        calls.append(kwargs["resource_jid"])
        assert len(kwargs["assigned_outline_events"]) == 1
        if len(calls) == 2:
            assert kwargs["session_state"]["accepted_primitive_program"][0]["composition_validation"]["derived_end_state"]["quality"] == "checked"
        return {"decision": "draft_ready", "primitive_events": [_row(contract if len(calls) == 1 else second)], "turns": []}
    monkeypatch.setattr(multi_turn, "_run_resource_primitive_batch", generate)
    async def run():
        assert (await generate_next_composition_event(planner=SimpleNamespace(), session_state=session,
            prepared_recovery_request=prepared))[0] == "primitive_steps_ready"
        assert (await generate_next_composition_event(planner=SimpleNamespace(), session_state=session,
            prepared_recovery_request=prepared))[0] == "draft_ready"
    asyncio.run(run())
    assert calls == ["owner", "other"]
    assert session["accepted_outline_prefix"] == events


@pytest.mark.parametrize("scope", ["product", "unknown", None])
def test_unknown_scopes_require_context(contract, scope):
    contract["recovery_des_model"]["state_variables"]["quality"]["scope"] = scope
    assert validate_recovery_primitive_composition(**contract)["feasibility_status"] == "NEEDS_CONTEXT"


def test_evidence_content_tampering_is_rejected(contract):
    row = _row(contract)
    row["composition_validation"]["projected_snapshot"]["mode"] = "fault"
    assert "fingerprint" in composition_program_error([contract["task"]], [row], prepared=_prepared(contract), session={})


def test_peer_resource_and_multiple_part_effects_reach_next_owner(contract):
    from cais_spade_llm.resources.recovery_composition import composition_grounding_context

    prepared = _prepared(contract)
    prepared["recovery_resources"]["receiver"] = {"recovery_snapshot": {"zone_A_part": None}}
    row = _row(contract)
    row["composition_validation"].update(projected_resources={"receiver": {"zone_A_part": "P"}},
        projected_parts={"Q": {"location": "receiver", "quality": "checked"}})
    session = {"accepted_primitive_program": [row]}
    context = composition_grounding_context(session=session, prepared=prepared,
        task={"resource_jid": "receiver", "part_name": "Q"})
    assert context["resource"]["zone_A_part"] == "P"
    assert context["part"] == {"location": "receiver", "quality": "checked"}


def test_rejected_event_does_not_advance_accepted_program(contract, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_composition import generate_next_composition_event

    prepared = _prepared(contract)
    session = {"accepted_outline_prefix": [deepcopy(contract["task"])], "accepted_primitive_program": []}
    row = _row(contract)
    row["composition_validation"]["feasibility_status"] = "INFEASIBLE"
    async def reject(**kwargs):
        return {"primitive_events": [row], "turns": [], "feedback": [{"reason": "wrong order"}]}
    monkeypatch.setattr(multi_turn, "_run_resource_primitive_batch", reject)
    decision, _, _ = asyncio.run(generate_next_composition_event(planner=SimpleNamespace(), session_state=session,
        prepared_recovery_request=prepared))
    assert decision == "primitive_blocked"
    assert session["accepted_primitive_program"] == []
    assert session["primitive_rejection_feedback"][0]["reason"] == "wrong order"


def test_composition_carries_history_references_without_copying_full_transcripts(contract, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    class ArchivedTranscript:
        def __deepcopy__(self, memo):
            raise AssertionError("Full archived transcripts must not be copied into the next resource batch")

    async def generate(**kwargs):
        carried = kwargs["carried_session_state"]
        assert carried["turns"][0]["response_artifact_path"] == "recorded_response.txt"
        assert "prompt_input" not in carried["turns"][0]
        assert carried["accepted_primitive_program"][0]["outline_id"] == "restore"
        return {"decision": "draft_ready", "primitive_events": []}

    monkeypatch.setattr(multi_turn, "_resource_agent_map", lambda planner: {
        "owner": SimpleNamespace(generate_recovery_primitives_batch=generate)})
    session = {"accepted_primitive_program": [_row(contract)], "turns": [{
        "phase": "primitive_generation", "response_artifact_path": "recorded_response.txt",
        "prompt_input": ArchivedTranscript()}]}
    asyncio.run(multi_turn._run_resource_primitive_batch(planner=SimpleNamespace(),
        prepared_recovery_request=_prepared(contract), session_state=session,
        resource_jid="owner", assigned_outline_events=[contract["task"]]))


def test_composition_parent_retains_substream_artifacts_without_nesting_batch_session(contract, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_composition import generate_next_composition_event

    full_turn = {"phase": "primitive_generation", "outline_id": "restore",
        "prompt_artifact_path": "recorded_prompt.txt", "response_artifact_path": "recorded_response.txt",
        "prompt_input": {"earlier_session": {"turns": ["large retained transcript"]}}}
    result = {"decision": "draft_ready", "primitive_events": [_row(contract)],
              "turns": [full_turn], "session_state": {"turns": [full_turn]}}

    async def generate(**kwargs):
        return result

    monkeypatch.setattr(multi_turn, "_run_resource_primitive_batch", generate)
    session = {"accepted_outline_prefix": [contract["task"]]}
    decision, turn, response = asyncio.run(generate_next_composition_event(planner=SimpleNamespace(),
        session_state=session, prepared_recovery_request=_prepared(contract)))
    assert decision == "draft_ready"
    child = turn["primitive_substream_turns"][0]
    assert child["prompt_artifact_path"] == "recorded_prompt.txt"
    assert child["response_artifact_path"] == "recorded_response.txt"
    assert "prompt_input" not in child
    assert "session_state" not in response["resource_batches"][0]
    assert "turns" not in response["resource_batches"][0]
    assert result["session_state"]["turns"][0] == full_turn


def test_composition_stops_after_three_same_cause_rejections(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_primitive_generation as generation, multi_turn_prompts as prompts,
    )

    calls = []

    async def ask(prompt, **kwargs):
        calls.append(prompt)
        return {"decision": "primitive_steps_ready", "revision": len(calls)}

    async def recorded(fn, **kwargs):
        return await fn(kwargs["prompt"])

    async def reject(**kwargs):
        feedback = [{"outline_id": "restore", "resource_jid": "owner",
            "constraint_code": "composition_infeasible", "reason": "Retreat is missing",
            "evidence_fingerprint": str(len(calls))}]
        kwargs["session_state"]["primitive_rejection_feedback"] = feedback
        return "need_primitive_revision", {"primitive_rejection_feedback": feedback}

    monkeypatch.setattr(prompts, "build_multi_turn_phase_prompt_input", lambda **kwargs: {})
    monkeypatch.setattr(prompts, "render_multi_turn_phase_prompt", lambda value: "author")
    monkeypatch.setattr(generation, "build_primitive_generation_prompt_context", lambda **kwargs: {})
    monkeypatch.setattr(generation, "recorded_structured_call", recorded)
    monkeypatch.setattr(generation, "_handle_primitive_generation_phase", reject)
    monkeypatch.setattr(generation, "_write_primitive_subturn_artifact", lambda **kwargs: {})
    result = asyncio.run(generation.generate_primitive_batch_with_llm_agent(
        llm_agent=SimpleNamespace(ask_llm_structured=ask),
        prepared_recovery_request={"composition_validation_version": 1},
        assigned_outline_events=[{"outline_id": "restore", "resource_jid": "owner"}]))

    assert len(calls) == 3
    assert result["decision"] == "primitive_blocked"
    assert result["primitive_events"] == []
    assert result["feedback"][0]["reason"] == "Retreat is missing"
    assert result["turns"][-1]["composition_same_cause_failures"] == 3


def test_move_home_completion_cannot_override_a_later_motion():
    from cais_spade_llm.resources.robot.recovery_composition import project_composition_completion

    model = {"events": [{"event_name": "move_home", "composition_contract": {
        "steps": [{"op": "move_to_named_pose", "params": {"pose_name": "home"}}]},
        "updates": {"resource_location": {"set": "home"}}}]}
    trace = [{"primitive": "move_to_named_pose", "params": {"pose_name": "home"},
              "resolved_params": {"pose_name": "home"}, "step_index": 0},
             {"primitive": "move_cartesian", "params": {"x": 2., "y": 0., "z": 1.},
              "resolved_params": {"x": 2., "y": 0., "z": 1.}, "step_index": 1}]
    result = project_composition_completion(task={"resource_jid": "robot"}, trace=trace,
        recovery_des_model=model, recovery_snapshot={}, projected_snapshot={"resource_location": "elsewhere"},
        part_context={}, grounding_context={}, primitive_catalog=[], resource_evidence={}, physical_evidence={})
    assert result["projected_snapshot"]["resource_location"] == "elsewhere"
    assert result["completed_contracts"] == []


@pytest.mark.parametrize("changed", [False, True])
def test_place_input_binding_uses_recorded_context_before_owner_output_enrichment(monkeypatch, changed):
    from cais_spade_llm.resources.robot import recovery_capability
    from cais_spade_llm.resources.robot.recovery_composition import preview_composition_output

    recorded = {"state": {"_task_ctx": {"part_name": "P", "pick_tcp_z": 1.2, "travel_z": 1.5}}}
    calculated = deepcopy(recorded)
    calculated["state"]["_task_ctx"].update(surface_role="support", insert_pose={"z": 1.0})
    monkeypatch.setattr(recovery_capability, "calculate_recovery_place_target", lambda **kwargs: {
        "status": "calculated", "destination_location": "buffer", "target_pose": {"x": 0., "y": 0., "z": 1.0}})
    params = {"part_name": "P", "destination_location": "buffer",
              "pick_ctx": deepcopy(recorded["state"]["_task_ctx"])}
    if changed:
        params["pick_ctx"]["pick_tcp_z"] = 2.0
    result = preview_composition_output(primitive="compute_place_targets", params=params,
        recovery_snapshot={}, part_context={}, task={"part_name": "P", "resource_jid": "robot"},
        grounding_context={"resources": {"robot": {"resource_evidence": recorded}}},
        resource_evidence=calculated, physical_evidence={})
    assert result["status"] == ("NEEDS_CONTEXT" if changed else "calculated")
    assert "surface_role" not in recorded["state"]["_task_ctx"]


def test_pick_override_feedback_identifies_all_unsupported_inputs_without_accepting_them(monkeypatch):
    from cais_spade_llm.resources.robot import recovery_capability
    from cais_spade_llm.resources.robot.recovery_composition import preview_composition_output

    monkeypatch.setattr(recovery_capability, "calculate_recovery_pick_target", lambda **kwargs: {
        "status": "calculated", "target_pose": {"x": 0., "y": 0., "z": 1.}, "travel_z": 1.5})
    values = dict(primitive="compute_pick_targets", recovery_snapshot={}, part_context={},
        grounding_context={}, task={"part_name": "P", "resource_jid": "robot"},
        resource_evidence={}, physical_evidence={})
    result = preview_composition_output(params={"part_name": "P", "product_geometry": {},
        "target_pose": {"x": 0., "y": 0., "z": 1.}, "target_pose_source": "observed_pose"}, **values)
    assert result["status"] == "NEEDS_CONTEXT"
    assert "product_geometry, target_pose, target_pose_source" in result["reason"]
    assert "Omit these parameters" in result["reason"]
    assert preview_composition_output(params={"part_name": "P"}, **values)["status"] == "calculated"
