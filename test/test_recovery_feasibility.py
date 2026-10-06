"""Recovery task synthesis feasibility and evidence-handoff regression tests."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cais_spade_llm.resources.recovery_feasibility import validate_primitive_support


def _task() -> dict:
    return {
        "outline_id": "recovery_home",
        "event_name": "evt_q7",
        "resource_jid": "xarm6@localhost",
        "expected_start_state": {"resource_state": "failed", "held_part": None},
        "expected_end_state": {"resource_state": "idle"},
    }


def _model() -> dict:
    return {
        "descriptor_fingerprint": "test-model",
        "state_variables": {
            "resource_state": {"scope": "resource", "domain": ["failed", "idle"]},
            "held_part": {"scope": "resource", "domain": [None]},
        },
        "events": [
            {
                "event_name": "move_home",
                "guards": {"resource_state": {"equals": "idle"}},
                "updates": {"resource_state": {"set": "idle"}},
                "primitive_support": [
                    {"primitive": "move_to_named_pose", "params": {"pose_name": "home"}}
                ],
            },
            {
                "event_name": "place_insert",
                "guards": {"resource_state": {"equals": "positioned"}},
                "updates": {"resource_state": {"set": "idle"}},
                "primitive_support": [{"primitive": "release_part", "params": {}}],
            },
        ],
        "primitive_catalog": [
            {
                "name": "move_to_named_pose",
                "params": {"pose_name": {"type": "string", "enum": ["home", "prusa-mk4-1"]}},
                "required_params": ["pose_name"],
                "capability_constraints": {},
            },
            {"name": "release_part", "params": {}, "required_params": []},
        ],
    }


def _check(model: dict | None = None, task: dict | None = None, evaluator=None) -> dict:
    return validate_primitive_support(
        task=task or _task(),
        recovery_des_model=model or _model(),
        recovery_snapshot={"current_state": "failed", "held_part": None},
        validate_primitive=evaluator
        or (
            lambda **_: {
                "allowed": True,
                "feasibility_status": "FEASIBLE",
                "evidence": {"mocked": True},
            }
        ),
    )


def test_one_feasible_alternative_does_not_satisfy_the_union() -> None:
    def evaluate(*, primitive: str, params: dict) -> dict:
        allowed = primitive == "release_part"
        return {"allowed": allowed, "feasibility_status": "FEASIBLE" if allowed else "INFEASIBLE"}

    result = _check(evaluator=evaluate)
    assert result["allowed"] is False
    assert result["feasibility_status"] == "INFEASIBLE"
    assert [row["primitive"] for row in result["primitive_support"]["primitives"]] == [
        "move_to_named_pose", "release_part",
    ]
    assert [row["feasibility_status"] for row in result["primitive_support"]["primitives"]] == [
        "INFEASIBLE", "FEASIBLE",
    ]


def test_new_transition_uses_existing_successor_without_nominal_predecessor() -> None:
    result = _check()
    assert _task()["event_name"] not in [row["event_name"] for row in _model()["events"]]
    assert result["allowed"] is True
    assert result["primitive_support"]["expected_start_state"]["resource_state"] == "failed"
    assert result["primitive_support"]["expected_end_state"]["resource_state"] == "idle"
    assert result["primitive_support"]["primitives"][0]["params"] == {"pose_name": "home"}


@pytest.mark.parametrize("value", ["xarm6_clear_state", "Idle", True])
def test_unsupported_successor_is_rejected_before_primitive_validation(value) -> None:
    task = _task()
    task["expected_end_state"]["resource_state"] = value

    def unexpected(**_):
        pytest.fail("An unsupported successor must not reach forward validation")

    result = _check(task=task, evaluator=unexpected)
    assert result["allowed"] is False
    assert result["constraint_code"] == "unsupported_successor_condition"


def test_all_primitives_of_one_alternative_require_witnesses() -> None:
    model = _model()
    model["events"] = model["events"][:1]
    model["events"][0]["primitive_support"].append({"primitive": "release_part", "params": {}})

    def evaluate(*, primitive: str, params: dict) -> dict:
        return {
            "allowed": primitive != "release_part",
            "feasibility_status": "FEASIBLE" if primitive != "release_part" else "NEEDS_CONTEXT",
        }

    result = _check(model=model, evaluator=evaluate)
    assert result["allowed"] is False
    assert result["feasibility_status"] == "NEEDS_CONTEXT"


def test_finite_parameter_domain_search_returns_satisfying_assignment() -> None:
    model = _model()
    model["events"] = model["events"][:1]
    model["events"][0]["primitive_support"][0]["params"] = {}

    def evaluate(*, primitive: str, params: dict) -> dict:
        allowed = params["pose_name"] == "prusa-mk4-1"
        return {"allowed": allowed, "feasibility_status": "FEASIBLE" if allowed else "INFEASIBLE"}

    result = _check(model=model, evaluator=evaluate)
    assert result["primitive_support"]["primitives"][0]["params"]["pose_name"] == "prusa-mk4-1"


def test_parameter_domain_failure_is_not_physical_feasibility() -> None:
    model = _model()
    model["events"] = model["events"][:1]
    model["events"][0]["primitive_support"][0]["params"] = {"pose_name": 2}
    assert _check(model=model)["feasibility_status"] == "INFEASIBLE"


def test_missing_continuous_parameter_evidence_remains_needs_context() -> None:
    model = _model()
    model["events"] = model["events"][:1]
    model["events"][0]["primitive_support"][0]["params"] = {"pose_name": {"$step": "target"}}
    model["primitive_catalog"][0]["params"]["pose_name"] = {"type": "string"}
    assert _check(model=model)["feasibility_status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize(
    ("condition", "status"),
    [
        ({"current_state": {"equals": "idle"}}, "INFEASIBLE"),
        ({"controller_ready": {"equals": True}}, "NEEDS_CONTEXT"),
        ({"current_state": {"equals": "failed"}}, "FEASIBLE"),
    ],
)
def test_resource_capability_constraints_are_authoritative(condition: dict, status: str) -> None:
    model = _model()
    model["events"] = model["events"][:1]
    model["primitive_catalog"][0]["capability_constraints"] = condition
    assert _check(model=model)["feasibility_status"] == status


def test_sequence_preconditions_are_deferred_to_primitive_composition() -> None:
    model = _model()
    model["primitive_catalog"][0]["preconditions"] = {"current_state": {"equals": "idle"}}
    assert _check(model=model)["allowed"] is True


def test_missing_catalog_and_missing_evaluator_evidence_do_not_pass() -> None:
    model = _model()
    model["primitive_catalog"] = []
    assert _check(model=model)["feasibility_status"] == "NEEDS_CONTEXT"
    result = _check(evaluator=lambda **_: {"allowed": True})
    assert result["allowed"] is False
    assert result["feasibility_status"] == "NEEDS_CONTEXT"


def test_bound_successor_parameter_reaches_primitive_witness() -> None:
    model = _model()
    model["events"] = model["events"][:1]
    model["state_variables"]["resource_location"] = {
        "scope": "resource",
        "domain": ["home", "prusa-mk4-1"],
    }
    model["events"][0]["updates"]["resource_location"] = {"set_from_param": "pose_name"}
    model["events"][0]["primitive_support"][0]["params"] = {"pose_name": {"$arg": "pose_name"}}
    task = _task()
    task["expected_end_state"]["resource_location"] = "home"
    assert _check(model=model, task=task)["primitive_support"]["primitives"][0]["params"] == {
        "pose_name": "home",
    }


def test_ra_catalog_exposes_printer_support_and_constraints() -> None:
    from cais_spade_llm.agents.resource_agent.printing_agent import PrintingAgent
    from cais_spade_llm.resources.machine.printer_profile import PRINTER_PROFILE
    from cais_spade_llm.resources.resource_primitives import build_execution_primitive_catalog

    resource = SimpleNamespace(
        jid="prusa-mk4-1@localhost",
        static_capabilities={"resource_type": "printer"},
        _RESOURCE_PROFILE=PRINTER_PROFILE,
        _RECOVERY_PRIMITIVES=PrintingAgent._RECOVERY_PRIMITIVES,
        pause_job=PrintingAgent.pause_job,
        resume_job=PrintingAgent.resume_job,
        cancel_job=PrintingAgent.cancel_job,
    )
    catalog = build_execution_primitive_catalog(resource)
    pause = next(row for row in catalog if row["name"] == "pause_job")
    assert pause["capability_constraints"] == {"current_state": {"equals": "printing"}}
    model = PrintingAgent.recovery_des_model(resource, snapshot={"current_state": "printing"})
    assert model["events"][0]["primitive_support"][0]["primitive"] == "pause_job"
    assert model["primitive_catalog"] == catalog


def test_pa_commits_support_and_composition_receives_it(tmp_path) -> None:
    from test_case3_recovery_dryrun import (
        _fixture_outline_responses,
        _load_case3_runtime_context,
        _prepare_recovery_dryrun_harness,
        multi_turn_outline_generation,
    )

    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _primitive_authoring_event_context,
        _primitive_batch_session_state,
    )

    # Exercise primitive composition after explicit destination occupancy.
    context = _load_case3_runtime_context()
    for snapshot in context["resource_snapshots"]:
        if snapshot["resource_id"] == "ur5e-3":
            snapshot.update(current_location="assembly_board-v1",
                            occupancy={"location": "assembly_board-v1"})
        else:
            snapshot.pop("current_location")
    path = tmp_path / "occupied_context.json"
    path.write_text(json.dumps(context))
    _, _, planner, request = asyncio.run(
        _prepare_recovery_dryrun_harness(runtime_context_path=path, scripted_responses=_fixture_outline_responses())
    )
    session = deepcopy(request["multi_turn_session_seed"])
    candidate = {
        "outline_id": "recovery_home",
        "event_name": "evt_q7",
        "resource_jid": "recovery-resource-3@localhost",
        "expected_end_state": {"resource_state": "idle", "resource_location": "home"},
        "rationale": "Clear the occupied destination using supported resource conditions.",
    }
    decision, turn = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session,
            parsed_response={"thought": "clear occupancy", "candidate_events": [candidate]},
            prepared_recovery_request=request,
            planner=planner,
        )
    )
    assert decision == "need_next_task", turn
    accepted = session["accepted_outline_prefix"][0]
    assert accepted["primitive_support"]["event_name"] == "move_home"
    assert accepted["primitive_support"]["primitives"][0]["params"]["pose_name"] == "home"
    assert (
        _primitive_authoring_event_context(accepted)["primitive_support"]
        == accepted["primitive_support"]
    )
    assert session["accepted_primitive_program"] == []
    assert "primitive_steps" not in accepted
    batch = _primitive_batch_session_state(assigned_outline_events=[accepted])
    assert batch["accepted_outline_prefix"][0]["primitive_support"] == accepted["primitive_support"]
    assert batch["accepted_primitive_program"] == []
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _render_primitive_generation_prompt,
    )

    prompt = _render_primitive_generation_prompt({"session_state": batch})
    assert '"primitive_support"' in prompt
    assert "compatible parameter bindings" in prompt
    assert "An accepted outline is not an executable program" in prompt

    before = deepcopy(session)
    candidate["expected_end_state"]["resource_state"] = "xarm6_clear_state"
    decision, turn = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session,
            parsed_response={"thought": "unsupported successor", "candidate_events": [candidate]},
            prepared_recovery_request=request,
            planner=planner,
        )
    )
    assert decision == "need_revision"
    for key in (
        "accepted_outline_prefix",
        "symbolic_resources",
        "symbolic_parts",
        "projected_safety_dfa_states",
    ):
        assert session[key] == before[key]


def test_malformed_parameter_domain_remains_needs_context() -> None:
    model = _model()
    model["events"] = model["events"][:1]
    model["primitive_catalog"][0]["params"]["pose_name"] = {"type": "missing_type"}
    assert _check(model=model)["feasibility_status"] == "NEEDS_CONTEXT"


def test_robot_support_retains_private_primitives_and_unresolved_parameters() -> None:
    from cais_spade_llm.resources.robot.robot_task_recovery import robot_recovery_des_descriptor

    model = robot_recovery_des_descriptor(
        resource_jid="ur5e@localhost",
        snapshot={"current_state": "positioned", "held_part": "LG"},
        task_names=["place_insert"],
    )
    support = model["events"][0]["primitive_support"]
    assert any(step["primitive"] == "move_insert" for step in support)
    release = next(step for step in support if step["primitive"] == "release_part")
    assert isinstance(release["params"]["model_name"], dict)
    assert "<MODEL_NAME_FROM_PART_TARGET>" not in str(support)


def test_printer_missing_physical_context_remains_needs_context() -> None:
    from cais_spade_llm.agents.resource_agent.printing_agent import PrintingAgent

    resource = SimpleNamespace(_RECOVERY_PRIMITIVES=PrintingAgent._RECOVERY_PRIMITIVES)
    result = PrintingAgent.check_recovery_primitive_feasibility(
        resource,
        primitive="resume_job",
        params={},
        task={},
        recovery_snapshot={"current_state": "paused"},
        part_context={},
        grounded_action={},
    )
    assert result["allowed"] is False
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["evidence"]["missing_fields"] == ["material_state", "bed_state"]


@pytest.mark.parametrize("tamper", ["missing", "stale", "NEEDS_CONTEXT"])
def test_pa_rejects_missing_or_stale_primitive_support(monkeypatch, tamper: str) -> None:
    from test_case3_recovery_dryrun import (
        _fixture_outline_responses,
        _prepare_recovery_dryrun_harness,
        multi_turn_outline_generation,
    )

    _, product_agent, planner, request = asyncio.run(
        _prepare_recovery_dryrun_harness(
            scripted_responses=_fixture_outline_responses(),
        )
    )
    session = deepcopy(request["multi_turn_session_seed"])
    before = deepcopy(session)
    transport = product_agent.request_recovery_outline_physical_validation

    async def altered(**kwargs):
        reply = await transport(**kwargs)
        for row in reply["results"]:
            physical = row["physical_feasibility"]
            if physical.get("allowed") is not True:
                continue
            if tamper == "missing":
                physical.pop("primitive_support", None)
            elif tamper == "stale":
                physical["primitive_support"]["descriptor_fingerprint"] = "stale"
            else:
                physical["primitive_support"]["primitives"][0]["feasibility_status"] = tamper
        return reply

    monkeypatch.setattr(product_agent, "request_recovery_outline_physical_validation", altered)
    candidate = deepcopy(_fixture_outline_responses()[2]["candidate_events"][0])
    evaluation = asyncio.run(
        multi_turn_outline_generation._validate_candidate_sequence(
            candidate={"candidate_index": 0, "surface_events": [candidate]},
            sequence_index=1,
            action_horizon="1",
            action_horizon_k=1,
            session_state=session,
            prepared_recovery_request=request,
            planner=planner,
        )
    )
    assert evaluation["valid"] is False
    assert [stage["status"] for stage in evaluation["validation_stages"]] == [
        "passed",
        "passed",
        "rejected",
        "skipped",
    ]
    assert (
        evaluation["validation_findings"][0]["constraint_code"] == "resource_validation_unavailable"
    )
    assert session == before


def test_optional_finite_parameter_can_supply_the_feasible_witness() -> None:
    model = _model()
    model["events"] = model["events"][:1]
    model["primitive_catalog"][0]["params"]["speed"] = {"type": "number", "enum": [0.1, 0.2]}

    def evaluate(*, primitive: str, params: dict) -> dict:
        allowed = params.get("speed") == 0.2
        return {"allowed": allowed, "feasibility_status": "FEASIBLE" if allowed else "INFEASIBLE"}

    result = _check(model=model, evaluator=evaluate)
    assert result["allowed"] is True
    assert result["primitive_support"]["primitives"][0]["params"]["speed"] == 0.2


def test_unsearched_optional_continuous_domain_cannot_prove_infeasible() -> None:
    model = _model()
    model["events"] = model["events"][:1]
    model["primitive_catalog"][0]["params"]["speed"] = {"type": "number"}
    result = _check(
        model=model,
        evaluator=lambda **_: {
            "allowed": False,
            "feasibility_status": "INFEASIBLE",
        },
    )
    assert result["feasibility_status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize(
    ("named_poses", "pose_name", "status"),
    [
        ([], "home", "NEEDS_CONTEXT"),
        (["home"], "", "INFEASIBLE"),
        (["home"], "Home", "INFEASIBLE"),
        (["home"], "home", "FEASIBLE"),
    ],
)
def test_robot_named_pose_witness_uses_exact_primitive_parameters(
    named_poses: list,
    pose_name: str,
    status: str,
) -> None:
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    actions = []

    def check(**kwargs):
        actions.append(kwargs["grounded_action"])
        return {"allowed": True, "evidence": {"mocked": True}}

    resource = SimpleNamespace(
        static_capabilities={"named_poses": named_poses},
        check_recovery_physical_feasibility=check,
    )
    result = RobotAgent.check_recovery_primitive_feasibility(
        resource,
        primitive="move_to_named_pose",
        params={"pose_name": pose_name},
        task={},
        recovery_snapshot={},
        part_context={},
        grounded_action={"function_name": "place_approach"},
    )
    assert result["feasibility_status"] == status
    if status == "FEASIBLE":
        assert actions[0]["function_name"] == "move_to_named_pose"
        assert actions[0]["target"]["named_pose"] == "home"
    else:
        assert actions == []


def test_cca_rejection_after_ra_success_preserves_the_pa_prefix() -> None:
    from test_case3_recovery_dryrun import (
        _fixture_outline_responses,
        _prepare_recovery_dryrun_harness,
        multi_turn_outline_generation,
    )

    _, _, planner, request = asyncio.run(
        _prepare_recovery_dryrun_harness(
            scripted_responses=_fixture_outline_responses(),
        )
    )
    request["llm_input"]["loaded_safety_rules"] = [
        {
            "id": "SAFE_1",
            "ap_scope": "both",
            "recovery_aps": [],
            "text": "No accepting continuation after this projected step.",
            "dfa_dot": "digraph DFA { node [shape = doublecircle]; 1; "
            "node [shape = circle]; 2; init -> 1; "
            '1 -> 2 [label="true"]; 2 -> 2 [label="true"]; }',
        }
    ]
    session = deepcopy(request["multi_turn_session_seed"])
    before = deepcopy(session)
    candidate = deepcopy(_fixture_outline_responses()[2]["candidate_events"][0])
    decision, turn = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session,
            parsed_response={"thought": "test CCA rejection", "candidate_events": [candidate]},
            prepared_recovery_request=request,
            planner=planner,
        )
    )
    assert decision == "need_revision"
    evaluation = turn["candidate_evaluations"][0]
    assert [stage["status"] for stage in evaluation["validation_stages"]] == [
        "passed",
        "passed",
        "passed",
        "rejected",
    ]
    assert evaluation["validation_findings"][0]["rule_id"] == "SAFE_1"
    for key in (
        "accepted_outline_prefix",
        "symbolic_resources",
        "symbolic_parts",
        "projected_safety_dfa_states",
    ):
        assert session.get(key) == before.get(key)


def test_union_deduplicates_identity_and_retains_all_bindings() -> None:
    model = _model()
    second = deepcopy(model["events"][0])
    second["event_name"] = "another_home_transition"
    second["primitive_support"][0]["params"]["pose_name"] = "prusa-mk4-1"
    model["events"] = [model["events"][0], second]
    calls = []

    def evaluate(*, primitive, params):
        calls.append(deepcopy(params))
        allowed = params["pose_name"] == "prusa-mk4-1"
        return {"allowed": allowed, "feasibility_status": "FEASIBLE" if allowed else "INFEASIBLE"}

    result = _check(model=model, evaluator=evaluate)
    assert result["allowed"] is True
    support = result["primitive_support"]
    assert support["semantics"] == "successor_primitive_union"
    assert len(support["primitives"]) == 1
    witness = support["primitives"][0]
    assert len(witness["contributors"]) == 2
    assert len(witness["parameter_attempts"]) == 2
    assert witness["params"] == {"pose_name": "prusa-mk4-1"}
    assert calls == [{"pose_name": "home"}, {"pose_name": "prusa-mk4-1"}]


@pytest.mark.parametrize("missing, status", [(False, "INFEASIBLE"), (True, "NEEDS_CONTEXT")])
def test_known_empty_union_differs_from_missing_support(missing, status) -> None:
    model = _model()
    for event in model["events"]:
        if missing:
            event.pop("primitive_support")
        else:
            event["primitive_support"] = []
    assert _check(model=model)["feasibility_status"] == status


def test_union_does_not_claim_product_effect_coverage() -> None:
    model = _model()
    model["state_variables"]["part_location"] = {"scope": "part", "domain": ["M1 staging tray"]}
    task = _task()
    task["expected_end_state"]["part_location"] = "M1 staging tray"
    support = _check(model=model, task=task)["primitive_support"]
    assert support["covered_valuation_fields"] == ["resource_state"]
    assert support["deferred_valuation_fields"] == ["part_location"]
    assert all("part_location" not in row["valuation_coverage"] for row in support["matching_transitions"])


def test_one_impossible_union_primitive_overrides_other_missing_evidence() -> None:
    result = _check(evaluator=lambda primitive, params: {
        "allowed": False,
        "feasibility_status": "INFEASIBLE" if primitive == "release_part" else "NEEDS_CONTEXT",
    })
    assert result["feasibility_status"] == "INFEASIBLE"


def _staging_validation(steps):
    from cais_spade_llm.resources.robot.robot_primitives import robot_primitive_sequence_validator

    return robot_primitive_sequence_validator(
        outline_event={
            "resource_jid": "ur5e-4@localhost", "part_name": "gear_large",
            "expected_start_state": {"held_part": "gear_large", "part_location": "ur5e-4@localhost"},
            "expected_end_state": {"held_part": None, "part_location": "M1 staging tray"},
        },
        primitive_steps=[], trace_metadata={"step_results": steps},
        start_snapshot={"held_part": "gear_large"}, projected_snapshot={"held_part": None},
    )


def _placement_trace():
    return [
        {"primitive": "compute_place_targets", "event_fact_path": "event_facts.place_targets.gear_large",
         "resolved_params": {"part_name": "gear_large", "destination_location": "M1 staging tray"}},
        {"primitive": "move_cartesian", "resolved_params": {"x": 1.0, "y": 0.2, "z": 0.5},
         "params": {axis: {"context_ref": f"event_facts.place_targets.gear_large.target_pose.{axis}"} for axis in ("x", "y", "z")},
         "context_refs": [f"event_facts.place_targets.gear_large.target_pose.{axis}" for axis in ("x", "y", "z")]},
        {"primitive": "release_part", "resolved_params": {"part_name": "gear_large"}},
    ]


def test_airborne_release_does_not_establish_staging() -> None:
    findings = _staging_validation(_placement_trace()[-1:])
    assert {row["constraint_code"] for row in findings} >= {"release_target_grounding_required", "trace_fact_required"}
    assert not _staging_validation(_placement_trace())


def test_move_away_invalidates_release_motion_witness() -> None:
    steps = _placement_trace()
    steps.insert(2, {"primitive": "move_relative", "resolved_params": {"dx": 0, "dy": 0, "dz": 0.4}})
    findings = _staging_validation(steps)
    assert any(row["evidence"].get("required_trace_fact") == "motion_landed_on_target" for row in findings)


def test_one_target_coordinate_cannot_ground_a_complete_release_pose() -> None:
    steps = _placement_trace()
    steps[1]["context_refs"] = steps[1]["context_refs"][:1]
    steps[1]["params"].update(y=5.0, z=5.0)
    assert _staging_validation(steps)


def test_swapped_target_coordinates_cannot_establish_a_landing() -> None:
    steps = _placement_trace()
    params = steps[1]["params"]
    params["y"], params["z"] = params["z"], params["y"]
    assert any(row["evidence"].get("required_trace_fact") == "motion_landed_on_target"
               for row in _staging_validation(steps))


def test_pick_target_motion_cannot_witness_placement_of_the_same_part() -> None:
    steps = _placement_trace()
    steps.insert(1, {
        "primitive": "compute_pick_targets", "event_fact_path": "event_facts.pick_targets.gear_large",
        "resolved_params": {"part_name": "gear_large"},
    })
    motion = steps[2]
    motion["params"] = {axis: {"context_ref": f"event_facts.pick_targets.gear_large.target_pose.{axis}"} for axis in ("x", "y", "z")}
    motion["context_refs"] = [value["context_ref"] for value in motion["params"].values()]
    assert any(row["evidence"].get("required_trace_fact") == "motion_landed_on_target"
               for row in _staging_validation(steps))


def test_resource_composition_preparation_requires_owner_provider() -> None:
    from unittest.mock import AsyncMock

    from cais_spade_llm.recovery_framework.kmr_agent import KMRResourceAgent

    async def scenario():
        worker = SimpleNamespace(run=AsyncMock())
        actor = KMRResourceAgent("recovery-resource-8@localhost", "none", worker=worker)
        actor._primitive_state = {"held_part": "KET8_Square_8mm", "current_pose": [1, 2, 3, 1, 0, 0, 0]}
        actor.workflow_custody = {"grasp_transform": [0, 0, .02, 1, 0, 0, 0], "attached": True}
        actor._primitive_evidence = {"source": "observed_worker_result"}
        request = {"resource_jid": str(actor.jid), "program_hash": "program",
                   "synthetic": True, "allowed": True}
        unavailable = await actor.prepare_recovery_composition_evidence(request)
        assert unavailable["status"] == "NEEDS_CONTEXT"
        exported = unavailable["physical_snapshot"]
        assert "base_pose" not in exported["snapshot"]
        assert exported["snapshot"]["held_part"] == "KET8_Square_8mm"
        assert exported["snapshot"]["grasp_transform"] == actor.workflow_custody["grasp_transform"]
        assert exported["evidence"]["primitive_evidence"] == actor._primitive_evidence
        exported["snapshot"]["current_pose"][0] = 99
        assert actor._primitive_state["current_pose"][0] == 1
        prepare = AsyncMock(return_value={
            "status": "prepared", "resource_jid": str(actor.jid), "program_hash": "program",
            "execution_mode": "mock", "mock_executor": True, "preparation_id": "owner-preparation",
        })
        actor.recovery_composition_evidence_provider = SimpleNamespace(prepare=prepare)
        prepared = await actor.prepare_recovery_composition_evidence(request)
        assert prepared["status"] == "prepared"
        assert prepare.await_args.kwargs["request"] == request
        prepare.return_value["program_hash"] = "another-program"
        assert (await actor.prepare_recovery_composition_evidence(request))["status"] == "NEEDS_CONTEXT"
        worker.run.assert_not_awaited()

    asyncio.run(scenario())


def _registered_resource_request(actor):
    reference = {"recovery_id": "sequence", "task_id": "task", "outline_id": "outline",
                 "program_hash": "program", "problem_id": "problem"}
    target = [1., 2., 3., 1., 0., 0., 0.]
    steps = [{"primitive": "move_cartesian", "params": {"target": target}}]
    grant = {"recovery_composition_ref": reference, "preparation_id": "prepared",
             "schedule_id": "schedule", "primitive_steps": steps,
             "resolved_primitive_steps": deepcopy(steps)}
    params = {"primitive_steps": steps, "recovery_composition_ref": reference,
              "task_id": "task", "out_state": "predicted_only", "start_safety_mode": "fast_path"}
    actor._kmr_execution_request = {}
    return params, grant


@pytest.mark.parametrize("grant_change", [None, "missing", "resolved_params"])
def test_registered_macro_rejects_ungranted_calls_and_preserves_observed_state(grant_change) -> None:
    from unittest.mock import AsyncMock

    from cais_spade_llm.recovery_framework.kmr_agent import KMRResourceAgent

    async def scenario():
        target = [1., 2., 3., 1., 0., 0., 0.]
        worker = SimpleNamespace(run=AsyncMock(return_value={
            "status": "completed", "result": {"tcp_pose": target},
            "primitive_results": [{"primitive": "move_cartesian", "status": "completed",
                                   "result": {"tcp_pose": target}}],
        }))
        actor = KMRResourceAgent("recovery-resource-8@localhost", "none", worker=worker)
        params, grant = _registered_resource_request(actor)
        actor._primitive_state = {"current_state": "idle", "held_part": None}
        observed = AsyncMock()
        actor._recovery_composition_observation_senders["task"] = observed
        if grant_change != "missing":
            actor._recovery_composition_grants["task"] = deepcopy(grant)
        if grant_change == "resolved_params":
            actor._recovery_composition_grants["task"]["resolved_primitive_steps"][0]["params"]["target"][0] = 9
        result = await actor.execute_recovery_macro(**params)
        if grant_change:
            assert result["status"].startswith("failed:recovery_composition")
            worker.run.assert_not_awaited()
            observed.assert_not_awaited()
        else:
            assert result["status"] == "completed"
            worker.run.assert_awaited_once()
            observation = observed.await_args.args[0]
            assert observation["params"] == params["primitive_steps"][0]["params"]
            assert observation["result"]["observations"]["primitive_results"][0]["status"] == "completed"
            assert observation["physical_snapshot"]["snapshot"]["current_pose"] == target
            assert actor._primitive_state["current_state"] == "idle"
            assert actor.current_state == "idle"
            actor._recovery_composition_grants["task"] = deepcopy(grant)
            duplicate = await actor.execute_recovery_macro(**params)
            assert duplicate["status"] == "failed:recovery_composition_grant_consumed"
            worker.run.assert_awaited_once()

    asyncio.run(scenario())


def test_registered_macro_waits_for_authenticated_cca_grant(monkeypatch) -> None:
    from unittest.mock import AsyncMock

    from spade.message import Message

    from cais_spade_llm.agents.resource_agent import resource_agent
    from cais_spade_llm.recovery_framework.kmr_agent import KMRResourceAgent

    async def scenario():
        worker = SimpleNamespace(run=AsyncMock(return_value={"status": "completed", "result": {}}))
        actor = KMRResourceAgent("recovery-resource-8@localhost", "none", worker=worker, cca_jid="cca@localhost")
        params, grant = _registered_resource_request(actor)
        task = Message(sender="product@localhost", body=json.dumps({
            "task_id": "task", "instruction": {"function_name": "execute_recovery_macro", "params": params},
        }))
        sent = []
        requested = asyncio.Event()

        async def send(_inbox, msg, **_kwargs):
            payload = json.loads(msg.body)
            sent.append(payload)
            if payload.get("status") == "safety_check":
                requested.set()

        monkeypatch.setattr(resource_agent, "send_agent_message", send)
        inbox = SimpleNamespace(agent=actor, receive=AsyncMock(return_value=task), _ack=AsyncMock())
        execution = asyncio.create_task(resource_agent.ResourceAgent._TaskInbox.run(inbox))
        await asyncio.wait_for(requested.wait(), 2)
        worker.run.assert_not_awaited()
        assert sent[0]["params"]["start_safety_mode"] == "cca_check"
        first_request_id = sent[0]["recovery_composition_request_id"]
        assert "recovery_composition_request_id" not in sent[0]["params"]
        reply = Message(sender="other@localhost", body=json.dumps({
            "task_id": "task", "decision": "allow", "recovery_composition_grant": grant,
            "recovery_composition_request_id": first_request_id,
        }))
        decisions = SimpleNamespace(agent=actor, receive=AsyncMock(return_value=reply))
        await resource_agent.ResourceAgent._SafetyDecisionInbox.run(decisions)
        assert "task" not in actor._safety_decisions
        worker.run.assert_not_awaited()
        reply.sender = "cca@localhost"
        reply.body = json.dumps({"task_id": "task", "decision": "block",
                                 "recovery_composition_request_id": first_request_id})
        await resource_agent.ResourceAgent._SafetyDecisionInbox.run(decisions)
        await asyncio.wait_for(execution, 2)
        worker.run.assert_not_awaited()
        assert inbox._ack.await_args.kwargs["status"] == "blocked"

        requested.clear()
        execution = asyncio.create_task(resource_agent.ResourceAgent._TaskInbox.run(inbox))
        await asyncio.wait_for(requested.wait(), 2)
        second_request = [row for row in sent if row.get("status") == "safety_check"][-1]
        second_request_id = second_request["recovery_composition_request_id"]
        assert second_request_id != first_request_id
        for decision in ("allow", "block"):
            reply.body = json.dumps({"task_id": "task", "decision": decision,
                                     "recovery_composition_grant": grant,
                                     "recovery_composition_request_id": first_request_id})
            await resource_agent.ResourceAgent._SafetyDecisionInbox.run(decisions)
            assert "task" not in actor._safety_decisions
            assert "task" not in actor._recovery_composition_grants
            worker.run.assert_not_awaited()
        reply.body = json.dumps({"task_id": "task", "decision": "allow",
                                 "recovery_composition_grant": grant,
                                 "recovery_composition_request_id": second_request_id})
        await resource_agent.ResourceAgent._SafetyDecisionInbox.run(decisions)
        await asyncio.wait_for(execution, 2)
        await asyncio.sleep(0)
        worker.run.assert_awaited_once()
        assert any(row["status"] == "recovery_primitive_observed" for row in sent)
        assert not actor._recovery_composition_grants
        assert not actor._recovery_composition_observation_senders
        assert not actor._pending_recovery_composition_request_ids
        await resource_agent.ResourceAgent._SafetyDecisionInbox.run(decisions)
        assert "task" not in actor._safety_decisions
        request_count = sum(row["status"] == "safety_check" for row in sent)
        await resource_agent.ResourceAgent._TaskInbox.run(inbox)
        assert inbox._ack.await_args.kwargs["status"] == "blocked"
        assert sum(row["status"] == "safety_check" for row in sent) == request_count
        worker.run.assert_awaited_once()

    asyncio.run(scenario())


def test_pa_registers_complete_sequence_before_requiring_dispatch_reference() -> None:
    from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
    from cais_spade_llm.agents.intelligent_product.product_recovery_controller import (
        ProductRecoveryController,
    )

    sequence = {"recovery_sequence_id": "sequence", "recovery_task_ids": ["task1", "task2"],
                "validation_policy": "validated", "recovery_safety_scope_id": "scope",
                "pending_nominal_task_ids": ["KMR_STORAGE_KET8_MOVE_TO_M1"]}
    nodes = [{"id": task_id, "function_name": "execute_recovery_macro", "resource_jid": "recovery-resource-8@localhost",
              "recovery_outline_id": task_id + "_outline", "event_name": task_id + "_event",
              "params": {"primitive_steps": [{"primitive": "release_part", "params": {}}]}}
             for task_id in sequence["recovery_task_ids"]]
    agent = SimpleNamespace(jid="product@localhost", runtime_recovery={"validation_policy": "validated", "active_recovery_sequence": sequence},
                            _runtime_recovery_context={}, process_planner=SimpleNamespace(nodes=nodes, global_fsa={}),
                            _active_recovery_sequence=lambda: sequence,
                            _enrich_observed_pose_recovery_params=deepcopy,
                            _runtime_recovery_session_validation_policy=lambda: "validated",
                            _build_runtime_plan_context=lambda: {})
    controller = ProductRecoveryController(agent)
    agent._dispatch_params_for_task_node = controller._dispatch_params_for_task_node
    agent._build_recovery_composition_request = controller._build_recovery_composition_request
    original = deepcopy(nodes)
    payload = ProductAgent._build_plan_validation_payload(agent)
    request = payload["recovery_composition_request"]
    assert [row["task_id"] for row in request["tasks"]] == ["task1", "task2"]
    assert request["pending_nominal_task_ids"] == ["KMR_STORAGE_KET8_MOVE_TO_M1"]
    assert "allowed" not in request and "snapshot" not in request
    with pytest.raises(RuntimeError, match="no CCA recovery composition reference"):
        controller._dispatch_params_for_task_node(nodes[0])
    reference = {"task_id": "task1", "recovery_id": "sequence", "outline_id": "task1_outline",
                 "program_hash": "program", "problem_id": "problem"}
    sequence["recovery_composition"] = {"task_refs": {"task1": reference}}
    dispatch = controller._dispatch_params_for_task_node(nodes[0])
    assert dispatch.pop("recovery_composition_ref") == reference
    assert dispatch == request["tasks"][0]["params"]
    assert nodes == original


def test_active_recovery_proof_prevents_nominal_fast_path(monkeypatch) -> None:
    from unittest.mock import AsyncMock

    from spade.message import Message

    from cais_spade_llm.agents.resource_agent import resource_agent

    async def scenario():
        actor = resource_agent.ResourceAgent("ur5e-3@localhost", "none", name="ur5e-3", cca_jid="cca@localhost")
        execute = AsyncMock(return_value={"status": "completed"})
        actor.executables["move_home"] = execute
        actor.recovery_composition_start_guard = lambda: True
        task = Message(sender="product@localhost", body=json.dumps({
            "task_id": "nominal", "instruction": {"function_name": "move_home",
                                                  "params": {"start_safety_mode": "fast_path"}},
        }))
        sent = []

        async def send(_inbox, msg, **_kwargs):
            payload = json.loads(msg.body)
            sent.append(payload)
            reply = Message(sender="cca@localhost", body=json.dumps({
                "task_id": "nominal", "decision": "block",
            }))
            decisions = SimpleNamespace(agent=actor, receive=AsyncMock(return_value=reply))
            await resource_agent.ResourceAgent._SafetyDecisionInbox.run(decisions)

        monkeypatch.setattr(resource_agent, "send_agent_message", send)
        inbox = SimpleNamespace(agent=actor, receive=AsyncMock(return_value=task), _ack=AsyncMock())
        await resource_agent.ResourceAgent._TaskInbox.run(inbox)
        assert sent[0]["status"] == "safety_check"
        assert sent[0]["params"]["start_safety_mode"] == "cca_check"
        assert inbox._ack.await_args.kwargs["status"] == "blocked"
        execute.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("raw_result", [None, [], {}, {"success": 1}, {"success": "true"}])
def test_registered_macro_requires_explicit_primitive_success(raw_result) -> None:
    from unittest.mock import AsyncMock

    from cais_spade_llm.recovery_framework.kmr_agent import KMRResourceAgent

    async def scenario():
        actor = KMRResourceAgent("recovery-resource-8@localhost", "none", worker=SimpleNamespace(run=AsyncMock()))
        params, grant = _registered_resource_request(actor)
        actor.recovery_execution_primitive_catalog()
        actor.kmr_primitives.move_cartesian = AsyncMock(return_value=raw_result)
        observed = AsyncMock()
        actor._recovery_composition_observation_senders["task"] = observed
        actor._recovery_composition_grants["task"] = grant
        result = await actor.execute_recovery_macro(**params)
        assert result["status"] == "failed"
        assert result["observations"]["completed_steps"] == 0
        assert observed.await_args.args[0]["result"] == raw_result
        assert "task" in actor._consumed_recovery_composition_refs

    asyncio.run(scenario())
