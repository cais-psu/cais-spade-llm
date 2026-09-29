"""Recovery task synthesis feasibility and evidence-handoff regression tests."""

from __future__ import annotations

import asyncio
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


def test_one_feasible_alternative_suffices() -> None:
    def evaluate(*, primitive: str, params: dict) -> dict:
        allowed = primitive == "release_part"
        return {"allowed": allowed, "feasibility_status": "FEASIBLE" if allowed else "INFEASIBLE"}

    result = _check(evaluator=evaluate)
    assert result["allowed"] is True
    assert result["primitive_support"]["event_name"] == "place_insert"
    assert [row["feasibility_status"] for row in result["evidence"]["alternatives"]] == [
        "INFEASIBLE",
        "FEASIBLE",
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


def test_pa_commits_support_and_composition_receives_it() -> None:
    from test_case3_recovery_dryrun import (
        _fixture_outline_responses,
        _prepare_recovery_dryrun_harness,
        multi_turn_outline_generation,
    )

    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _primitive_authoring_event_context,
        _primitive_batch_session_state,
    )

    _, _, planner, request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session = deepcopy(request["multi_turn_session_seed"])
    candidate = {
        "outline_id": "recovery_home",
        "event_name": "evt_q7",
        "resource_jid": "xarm6@localhost",
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
