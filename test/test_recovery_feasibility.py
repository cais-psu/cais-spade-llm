from __future__ import annotations

"""Recovery task synthesis feasibility and evidence-handoff regression tests."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cais_spade_llm.resources.recovery_feasibility import (
    derive_primitive_union,
    validate_primitive_support,
    validate_recovery_transition,
)


def test_resource_validation_inbox_reports_controller_error_and_remains_available(monkeypatch):
    from cais_spade_llm.agents.resource_agent import resource_agent

    validator = Mock(side_effect=[KeyError("target_pose"), {"results": [{"candidate_index": 7, "allowed": True}]}])
    owner = SimpleNamespace(jid="recovery-resource-4@localhost", logger=Mock(),
                            validate_recovery_outline_physical_candidates=validator)
    message = SimpleNamespace(sender="assembly_board-v1@localhost", body=json.dumps({
        "request_id": "validation-request", "candidates": [{"candidate_index": 7}],
    }))
    inbox = SimpleNamespace(agent=owner, receive=AsyncMock(return_value=message))
    send = AsyncMock()
    monkeypatch.setattr(resource_agent, "send_agent_message", send)

    async def run():
        await resource_agent.ResourceAgent._RecoveryOutlinePhysicalValidationInbox.run(inbox)
        failed = json.loads(send.call_args.args[1].body)
        assert failed["request_id"] == "validation-request"
        assert failed["results"][0]["allowed"] is False
        assert failed["results"][0]["findings"][0]["constraint_code"] == "resource_validation_unavailable"
        await resource_agent.ResourceAgent._RecoveryOutlinePhysicalValidationInbox.run(inbox)
        assert json.loads(send.call_args.args[1].body)["results"][0]["allowed"] is True
    asyncio.run(run())


@pytest.fixture
def preserved_part_physical_input():
    pose = {"x": 0.0, "y": -0.2, "z": 1.04}
    owner = SimpleNamespace(jid="recovery-resource-4@localhost", static_capabilities={})
    snapshot = {"held_part": None, "gripper_state": "open", "resource_state": "idle"}
    part = {"part_name": "KET4_Square_4mm", "part_state": "misplaced", "part_location": None,
            "part_holder_resource_jid": None, "observed_pose": pose}
    action = {"task_kind": "part_handling", "effect_scope": "resource_and_part",
              "target": {"pose": deepcopy(pose)},
              "preconditions": {"part": {"requires_acquisition": False}},
              "expected_effect": {"resource": {"current_state": "at_pick"},
                                  "part": {"state": "misplaced", "location": None,
                                           "holder": None, "pose": deepcopy(pose)}}}
    return owner, snapshot, part, action


@pytest.mark.parametrize("function_name", ["recorded_event", "detect_parts", "print_part"])
def test_preserved_part_fields_do_not_require_robot_custody(preserved_part_physical_input, function_name):
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    owner, snapshot, part, action = preserved_part_physical_input
    action["function_name"] = function_name
    original = deepcopy((snapshot, part, action))
    result = RobotAgent.check_recovery_physical_feasibility(owner, operation_kind="part_transfer",
        part_name=part["part_name"], part_context=part, recovery_snapshot=snapshot, grounded_action=action)
    assert result["allowed"] is False
    assert result["constraint_code"] == "resource_validation_unavailable"
    assert result["guard"]["kind"] == "physical_validation_required"
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert (snapshot, part, action) == original


@pytest.mark.parametrize(("field", "value"), [
    ("state", "assembled"), ("location", "assembly_board-v1"),
    ("pose", {"x": 0.1, "y": -0.2, "z": 1.04}),
    ("holder", "recovery-resource-4@localhost"), ("missing_state", None),
])
def test_real_or_unknown_part_changes_still_require_robot_custody(preserved_part_physical_input, field, value):
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    owner, snapshot, part, action = preserved_part_physical_input
    if field == "missing_state":
        del part["part_state"]
    else:
        action["expected_effect"]["part"][field] = value
    result = RobotAgent.check_recovery_physical_feasibility(owner, operation_kind="part_transfer",
        part_name=part["part_name"], part_context=part, recovery_snapshot=snapshot, grounded_action=action)
    assert result["allowed"] is False
    assert result["constraint_code"] == "required_part_not_held"


def test_explicit_acquisition_keeps_conflicting_custody_rejection(preserved_part_physical_input):
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    owner, snapshot, part, action = preserved_part_physical_input
    snapshot["held_part"] = "gear_small"
    action["preconditions"]["part"]["requires_acquisition"] = True
    result = RobotAgent.check_recovery_physical_feasibility(owner, operation_kind="part_transfer",
        part_name=part["part_name"], part_context=part, recovery_snapshot=snapshot, grounded_action=action)
    assert result["allowed"] is False
    assert result["constraint_code"] == "holder_conflict"


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


@pytest.mark.parametrize('combination', [False, True])
def test_unsupported_public_conditions_are_distinguished_from_combinations(combination):
    model = _model()
    model['state_variables']['held_part'].update(domain=[None, 'part'], domain_complete=True)
    model['events'] = [{
        'event_name': 'a', 'guards': {}, 'updates': {'resource_state': {'set': 'idle'}, 'held_part': {'set': None}},
        'primitive_support': [{'primitive': 'release_part', 'params': {}}]}, {
        'event_name': 'b', 'guards': {}, 'updates': {'resource_state': {'set': 'failed'}, 'held_part': {'set': 'part'}},
        'primitive_support': [{'primitive': 'release_part', 'params': {}}]}]
    task = {**_task(), 'expected_end_state': {'resource_state': 'idle', 'held_part': 'part' if combination else 'unknown'}}
    result = _check(model=model, task=task)
    assert result['feasibility_status'] == 'INFEASIBLE'
    evidence = result['evidence']['condition_support']
    assert evidence['combination_supported'] is False
    assert evidence['individual_conditions_supported'] is combination
    assert next(row for row in evidence['conditions'] if row['field'] == 'held_part')['supported'] is combination
    if combination:
        assert evidence['unsupported_combinations'] == [[
            {'field': 'held_part', 'actual': 'part'}, {'field': 'resource_state', 'actual': 'idle'}]]
    else:
        assert 'unsupported_combinations' not in evidence
    model['model_complete'] = False
    assert _check(model=model, task=task)['feasibility_status'] == 'NEEDS_CONTEXT'


def test_unsupported_pair_excludes_unrelated_conditions_and_preserves_input_order_independence():
    model = _model()
    model['state_variables']['held_part'].update(domain=[None, 'part'], domain_complete=True)
    model['state_variables']['resource_location'] = {'scope': 'resource', 'domain': ['M1', 'M2']}
    model['events'] = [{
        'event_name': 'private_first', 'guards': {},
        'updates': {'resource_state': {'set': 'idle'}, 'held_part': {'set': None}},
        'primitive_support': [{'primitive': 'release_part', 'params': {}}]}, {
        'event_name': 'private_second', 'guards': {},
        'updates': {'resource_state': {'set': 'failed'}, 'held_part': {'set': 'part'}},
        'primitive_support': [{'primitive': 'release_part', 'params': {}}]}]
    task = {**_task(), 'expected_start_state': {'resource_state': 'failed', 'held_part': None, 'resource_location': 'M2'},
        'expected_end_state': {'resource_state': 'idle', 'held_part': 'part', 'resource_location': 'M2'}}
    before = deepcopy((model, task))
    first = _check(model=model, task=task)['evidence']['condition_support']
    assert first['unsupported_combinations'] == [[
        {'field': 'held_part', 'actual': 'part'}, {'field': 'resource_state', 'actual': 'idle'}]]
    assert (model, task) == before
    model['events'].reverse()
    task['expected_end_state'] = dict(reversed(list(task['expected_end_state'].items())))
    assert _check(model=model, task=task)['evidence']['condition_support'] == first
    assert 'private_first' not in str(first) and 'private_second' not in str(first)


def test_preserving_a_value_does_not_establish_support_for_changing_it():
    model = _model()
    model['state_variables']['held_part']['domain'] = [None, 'part']
    model['events'][0]['guards']['held_part'] = {'equals': 'part'}
    task = {**_task(), 'expected_end_state': {'held_part': 'part'}}
    result = _check(model=model, task=task)
    assert result['evidence']['condition_support']['conditions'] == [
        {'field': 'held_part', 'actual': 'part', 'supported': False}]


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


@pytest.fixture
def preserved_custody_successor():
    task = {
        "part_name": "gear_small",
        "expected_start_state": {"resource_state": "failed", "held_part": None},
        "expected_end_state": {
            "resource_state": "positioned", "held_part": "gear_small",
            "resource_location": "assembly_board-v1",
        },
    }
    model = {
        "state_variables": {
            "resource_state": {"scope": "resource", "domain": ["failed", "picked", "positioned"]},
            "held_part": {"scope": "resource", "domain": [None]},
            "resource_location": {"scope": "resource", "domain": ["assembly_board-v1"]},
        },
        "events": [{
            "event_name": "place_approach",
            "guards": {"resource_state": {"equals": "picked"},
                       "held_part": {"equals_from_param": "part_name"}},
            "updates": {"resource_state": {"set": "positioned"}},
            "primitive_support": [{"primitive": "compute_place_targets", "params": {
                "part_name": {"$arg": "part_name"}, "destination_location": "assembly_board-v1",
            }}],
        }],
    }
    return task, model


def test_preserved_custody_uses_grounded_part_guard_beyond_captured_domain(preserved_custody_successor):
    task, model = preserved_custody_successor
    before = deepcopy((task, model))
    support = derive_primitive_union(task=task, recovery_des_model=model)
    assert [row["event_name"] for row in support["matching_transitions"]] == ["place_approach"]
    assert support["parameter_bindings"] == {"part_name": "gear_small"}
    assert support["matching_transitions"][0]["valuation_coverage"]["held_part"] == {
        "value": "gear_small", "basis": "preserved",
    }
    assert (task, model) == before


@pytest.mark.parametrize("case", [
    "different_part", "missing_part_name", "different_guard", "no_guard",
    "parameter_conflict", "declaration_conflict", "negative_guard", "closed_location", "closed_field",
])
def test_preserved_custody_does_not_weaken_other_successor_requirements(preserved_custody_successor, case):
    task, model = preserved_custody_successor
    event = model["events"][0]
    if case == "different_part":
        task["part_name"] = "KET4_Square_4mm"
    elif case == "missing_part_name":
        task.pop("part_name")
    elif case == "different_guard":
        event["guards"]["held_part"] = {"equals_from_param": "other_part"}
    elif case == "no_guard":
        event["guards"].pop("held_part")
    elif case == "parameter_conflict":
        task["params"] = {"part_name": "KET4_Square_4mm"}
    elif case == "declaration_conflict":
        event["parameter_bindings"] = {"part_name": {"equals": "KET4_Square_4mm"}}
    elif case == "negative_guard":
        event["guards"]["held_part"]["not_equals_from_param"] = "part_name"
    elif case == "closed_location":
        model["state_variables"]["resource_location"]["domain_complete"] = True
        task["expected_end_state"]["resource_location"] = "3D Printing Station"
        task["params"] = {"destination_location": "3D Printing Station"}
        event["guards"]["resource_location"] = {"equals_from_param": "destination_location"}
    elif case == "closed_field":
        model["state_variables"]["gripper_state"] = {"scope": "resource", "domain": ["open"], "domain_complete": True}
        task["expected_end_state"]["gripper_state"] = "closed"
        task["params"] = {"gripper_state": "closed"}
        event["guards"]["gripper_state"] = {"equals_from_param": "gripper_state"}
    assert derive_primitive_union(task=task, recovery_des_model=model)["matching_transitions"] == []


@pytest.mark.parametrize("value", ["xarm6_clear_state", "Idle", True])
def test_unsupported_successor_is_rejected_before_primitive_validation(value) -> None:
    task = _task()
    task["expected_end_state"]["resource_state"] = value

    def unexpected(**_):
        pytest.fail("An unsupported successor must not reach forward validation")

    result = _check(task=task, evaluator=unexpected)
    assert result["allowed"] is False
    assert result["constraint_code"] == "unsupported_successor_condition"


@pytest.mark.parametrize("missing", [None, "declaration", "complete_model"])
def test_part_only_successor_is_a_proposal_error_when_declarations_are_complete(missing):
    model = _model()
    model["state_variables"]["part_location"] = {
        "scope": "part", "domain": ["M2", "M2 staging tray"]}
    task = {**_task(), "expected_start_state": {"part_location": "M2"},
            "expected_end_state": {"part_location": "M2 staging tray"}}
    if missing == "declaration":
        model["state_variables"].pop("part_location")
    elif missing == "complete_model":
        model["model_complete"] = False
    original = deepcopy((model, task))
    evaluator = Mock(side_effect=AssertionError("A part-only proposal cannot establish resource primitive support"))
    result = _check(model=model, task=task, evaluator=evaluator)
    assert result["allowed"] is False
    assert result["feasibility_status"] == ("NEEDS_CONTEXT" if missing else "INFEASIBLE")
    if not missing:
        assert result["constraint_code"] == "unsupported_successor_condition"
        assert "public resource" in result["reason"]
    evaluator.assert_not_called()
    assert (model, task) == original


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


def test_optional_authored_argument_can_use_the_controller_default() -> None:
    model = _model()
    model["events"] = model["events"][:1]
    model["events"][0]["primitive_support"][0]["params"]["speed"] = {"$arg": "speed"}
    model["primitive_catalog"][0]["params"]["speed"] = {"type": "number"}
    result = _check(model=model)
    assert result["allowed"] is True
    assert "speed" not in result["primitive_support"]["primitives"][0]["params"]


def test_observed_parameter_witness_is_checked_and_retained() -> None:
    model = _model()
    model["events"] = model["events"][:1]
    model["events"][0]["primitive_support"][0] = {"primitive": "move_cartesian", "params": {
        axis: {"$step": "target", "path": [axis]} for axis in ("x", "y", "z")
    }}
    model["primitive_catalog"] = [{"name": "move_cartesian", "required_params": ["x", "y", "z"],
                                  "params": {axis: {"type": "number"} for axis in ("x", "y", "z")}}]
    evidence = {"x": .1, "y": .2, "z": 1.3}
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs)
        return {"allowed": False, "feasibility_status": "INFEASIBLE"}
    result = validate_primitive_support(task=_task(), recovery_des_model=model,
                                        recovery_snapshot={}, validate_primitive=evaluate,
                                        parameter_evidence=evidence)
    assert calls == [{"primitive": "move_cartesian", "params": evidence}]
    assert result["allowed"] is False
    assert result["primitive_support"]["parameter_evidence"] == evidence


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


@pytest.mark.parametrize("value,valid", [(None, True), ([], True), ([[1., 2., 3.]], True),
                                       ([["1", 2., 3.]], False), ([1., 2., 3.], False), ("[]", False)])
def test_primitive_schema_resolves_optional_nested_numeric_lists(value, valid):
    from jsonschema import Draft202012Validator
    from cais_spade_llm.function_analyzer import FunctionAnalyzer

    schema = FunctionAnalyzer._json_schema(list[list[float]] | None)
    assert schema == {"type": ["array", "null"], "items": {"type": "array", "items": {"type": "number"}}}
    assert Draft202012Validator(schema).is_valid(value) is valid


def test_primitive_schema_distinguishes_nullable_from_optional_and_preserves_explicit_schema():
    from cais_spade_llm.function_analyzer import FunctionAnalyzer

    def operation(nullable: list[float] | None, optional: float = 1., declared: str = ""):
        pass
    operation.__doc__ = '''---
params:
  declared:
    type: array
    items: {type: number}
    minItems: 3
---
Declared operation.'''
    schema = FunctionAnalyzer().analyze_function(operation)['parameters']
    assert schema['required'] == ['nullable']
    assert schema['properties']['nullable']['type'] == ['array', 'null']
    assert schema['properties']['declared'] == {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3}


def test_successor_rejection_reports_public_field_without_selecting_event():
    model = {'descriptor_fingerprint': 'recorded-model', 'state_variables': {
        'active_job': {'scope': 'resource', 'domain': ['job-1', None]},
        'job_state': {'scope': 'resource', 'domain': ['running', 'paused']}},
        'events': [{'event_name': 'PRIVATE_OWNER_EVENT', 'guards': {},
                    'updates': {'active_job': {'set': None}, 'job_state': {'set': 'paused'}},
                    'primitive_support': [{'primitive': 'pause_job', 'params': {}}]}],
        'primitive_catalog': [{'name': 'pause_job', 'params': {}, 'required_params': []}]}
    task = {'event_name': 'LLM_PROPOSED_EVENT',
            'expected_start_state': {'active_job': 'job-1', 'job_state': 'running'},
            'expected_end_state': {'active_job': 'job-1', 'job_state': 'paused'}}
    original = deepcopy((model, task))
    result = _check(model=model, task=task, evaluator=lambda **_: pytest.fail('Rejected proposal reached capabilities'))
    assert result['feasibility_status'] == 'INFEASIBLE'
    mismatch = result['evidence']['mismatches'][0]
    assert (mismatch['field'], mismatch['actual'], mismatch['expected']) == ('active_job', 'job-1', None)
    assert mismatch['source']['descriptor_fingerprint'] == 'recorded-model'
    assert 'PRIVATE_OWNER_EVENT' not in result['reason']
    assert (model, task) == original
    supported_fixture = deepcopy(task)
    supported_fixture['expected_end_state']['active_job'] = None
    assert _check(model=model, task=supported_fixture)['feasibility_status'] == 'FEASIBLE'


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


def test_outline_inbox_uses_resource_owned_primitive_support(monkeypatch) -> None:
    from unittest.mock import Mock

    from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent

    support = Mock(return_value={"allowed": False, "feasibility_status": "NEEDS_CONTEXT",
                                "reason": "resource-owned evidence is unavailable"})
    owner = SimpleNamespace(
        jid="xarm6@localhost", get_recovery_snapshot=lambda: {"current_state": "failed"},
        recovery_des_model=lambda **_: _model(),
        recovery_validation_resource_matches=lambda jid: jid == "xarm6@localhost",
        validate_recovery_primitive_support=support,
    )
    monkeypatch.setattr(ResourceAgent, "check_recovery_transition_feasibility",
                        lambda *args, **kwargs: {"allowed": True})
    result = ResourceAgent.validate_recovery_outline_physical_candidates(owner, {
        "candidates": [{"candidate_index": 0, "task": _task(), "physical_input": {}}],
    })
    support.assert_called_once()
    assert "resource-owned evidence is unavailable" in str(result)


def test_robot_relative_witness_uses_observed_pose() -> None:
    from unittest.mock import Mock

    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    check = Mock(return_value={"allowed": True})
    owner = SimpleNamespace(check_recovery_physical_feasibility=check)
    args = dict(primitive="move_relative", params={"dx": 0., "dy": 0., "dz": .08},
                task={}, part_context={}, grounded_action={})
    result = RobotAgent.check_recovery_primitive_feasibility(
        owner, recovery_snapshot={"current_pose": {"x": .2, "y": .3, "z": 1.}}, **args)
    assert result["allowed"] is True
    assert check.call_args.kwargs["grounded_action"]["target"]["pose"] == {"x": .2, "y": .3, "z": 1.08}
    check.reset_mock()
    result = RobotAgent.check_recovery_primitive_feasibility(owner, recovery_snapshot={}, **args)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    check.assert_not_called()


def test_controller_catalog_accepts_declared_none_default() -> None:
    from jsonschema import validate

    from cais_spade_llm.resources.resource_primitives import _schema_properties_for_function

    def compute_place_targets(assembly_board_v1_aruco: dict | None = None):
        """Use configured geometry when no marker observation is supplied."""

    properties, required, _ = _schema_properties_for_function(compute_place_targets)
    validate(None, properties["assembly_board_v1_aruco"])
    assert required == []


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


@pytest.mark.parametrize("field", ["active_job", "current_location", "held_part"])
def test_union_positive_parameter_binding_is_independent_of_field_name(field):
    model = {"state_variables": {field: {"scope": "resource", "domain": [None]},
                                "resource_state": {"scope": "resource", "domain": ["failed", "idle"]}},
             "events": [{"event_name": "retain", "updates": {"resource_state": {"set": "idle"}},
                         "guards": {field: {"equals_from_param": "item"}},
                         "primitive_support": [{"primitive": "inspect", "params": {}}]}]}
    task = {"params": {"item": "recorded_item"}, "expected_start_state": {"resource_state": "failed"},
            "expected_end_state": {field: "recorded_item", "resource_state": "idle"}}
    before = deepcopy((task, model))
    assert derive_primitive_union(task=task, recovery_des_model=model)["matching_transitions"]
    assert (task, model) == before
    model["state_variables"][field]["domain_complete"] = True
    assert not derive_primitive_union(task=task, recovery_des_model=model)["matching_transitions"]
    model["state_variables"][field].pop("domain_complete")
    task.pop("params")
    assert not derive_primitive_union(task=task, recovery_des_model=model)["matching_transitions"]


def test_union_preserved_nonrobot_field_does_not_introduce_primitives():
    model = {"state_variables": {"active_job": {"scope": "resource", "domain": [None, "job"]}},
             "events": [{"event_name": "inspect", "guards": {"active_job": {"equals": "job"}},
                         "updates": {}, "primitive_support": [{"primitive": "inspect", "params": {}}]}],
             "primitive_catalog": [{"name": "inspect", "params": {}, "required_params": []}]}
    task = {"event_name": "new_event", "expected_start_state": {"active_job": "job"},
            "expected_end_state": {"active_job": "job"}}
    result = _check(task=task, model=model)
    assert not result["allowed"]
    assert result["primitive_support"]["covered_valuation_fields"] == ["active_job"]
    assert result["primitive_support"]["matching_transitions"] == []
    assert result["primitive_support"]["primitives"] == []
    assert result["primitive_support"]["preserved_valuation_fields"] == ["active_job"]


@pytest.fixture
def machine_successors():
    model = {"state_variables": {
        "active_job": {"scope": "resource", "domain": ["environment_23", None]},
        "resource_state": {"scope": "resource", "domain": ["loaded", "completed"]},
        "job_state": {"scope": "resource", "domain": ["running", "paused"]},
        "selected_program": {"scope": "resource", "domain": ["circle", "square"]},
    }, "events": [
        {"event_name": "machine_part", "guards": {}, "updates": {"resource_state": {"set": "completed"}},
         "primitive_support": [{"primitive": "start_job", "params": {}}]},
        {"event_name": "select_program", "guards": {"active_job": {"equals": None}},
         "updates": {"selected_program": {"set_from_param": "program"}},
         "primitive_support": [{"primitive": "select_program", "params": {}}]},
        {"event_name": "pause_job", "guards": {},
         "updates": {"job_state": {"set": "paused"}, "active_job": {"set": None}},
         "primitive_support": [{"primitive": "pause_job", "params": {}}]},
    ], "primitive_catalog": [{"name": name, "params": {}, "required_params": []}
        for name in ("start_job", "select_program", "pause_job")]}
    task = {"resource_jid": "M2@localhost", "event_name": "LLM_PROPOSED_EVENT",
            "expected_start_state": {"active_job": "environment_23", "resource_state": "loaded",
                                     "job_state": "running", "selected_program": "circle"},
            "expected_end_state": {"job_state": "paused"}}
    return task, model


def test_minimal_goal_derives_changes_and_keeps_additional_effects_as_obligations(machine_successors):
    task, model = machine_successors
    calls = []
    result = _check(task=task, model=model, evaluator=lambda **kw: calls.append(kw) or {
        "allowed": True, "feasibility_status": "FEASIBLE"})
    assert result["allowed"]
    support = result["primitive_support"]
    assert calls == [{"primitive": "pause_job", "params": {}}]
    assert support["requested_changes"] == ["job_state"]
    assert support["matching_transitions"][0]["additional_effects"] == {"active_job": {"set": None}}
    assert "derived_end_state" not in support and "trace" not in support
    assert task["expected_end_state"] == {"job_state": "paused"}


def test_conflicting_proposal_feedback_is_independent_of_json_key_order(machine_successors):
    from itertools import permutations
    task, model = machine_successors
    fields = {**task["expected_start_state"], "job_state": "paused"}
    results = []
    for order in permutations(fields):
        task["expected_end_state"] = {key: fields[key] for key in order}
        result = _check(task=task, model=model, evaluator=lambda **_: pytest.fail("Unsupported proposal reached capabilities"))
        assert result["feasibility_status"] == "INFEASIBLE"
        results.append((result["reason"], result["evidence"]["mismatches"]))
    assert all(row == results[0] for row in results)
    assert [(row["field"], row["actual"], row["expected"]) for row in results[0][1]] == [
        ("active_job", "environment_23", None)]
    assert "completed" not in results[0][0]


def test_deferred_part_effect_does_not_introduce_unrelated_resource_feedback(machine_successors):
    task, model = machine_successors
    model["state_variables"]["part_state"] = {"scope": "part", "domain": ["loaded", "ready"]}
    model["events"][0]["updates"]["part_state"] = {"set": "ready"}
    task["expected_start_state"]["part_state"] = "loaded"
    task["expected_end_state"] = {**task["expected_start_state"], "job_state": "paused", "part_state": "ready"}
    result = _check(task=task, model=model)
    assert [row["field"] for row in result["evidence"]["mismatches"]] == ["active_job"]
    assert result["primitive_support"]["deferred_valuation_fields"] == ["part_state"]


@pytest.mark.parametrize("starting", [{}, {"job_state": None}, {"job_state": "paused"}])
def test_missing_null_and_unchanged_start_are_distinct(machine_successors, starting):
    task, model = machine_successors
    task["expected_start_state"] = starting
    result = _check(task=task, model=model)
    if not starting:
        assert result["feasibility_status"] == "NEEDS_CONTEXT"
        assert result["primitive_support"]["unknown_start_fields"] == ["job_state"]
    elif starting["job_state"] is None:
        assert result["feasibility_status"] == "FEASIBLE"
    else:
        assert result["feasibility_status"] == "INFEASIBLE"
        assert not result["primitive_support"]["primitives"]


def test_assignment_limit_is_configurable_without_changing_support_semantics():
    result = validate_primitive_support(task=_task(), recovery_des_model=_model(), recovery_snapshot={},
        validate_primitive=lambda **_: {"allowed": True, "feasibility_status": "FEASIBLE"},
        max_parameter_trials=5000)
    assert result["allowed"] and result["primitive_support"]["search"]["limit"] == 5000
    assert result["primitive_support"]["semantics"] == "successor_primitive_union"


def test_owner_snapshot_supplies_missing_starting_facts_without_rewriting_proposal(machine_successors):
    task, model = machine_successors
    snapshot = task.pop("expected_start_state")
    before = deepcopy((task, snapshot))
    support = derive_primitive_union(task=task, recovery_des_model=model, recovery_snapshot=snapshot)
    assert support["requested_changes"] == ["job_state"]
    assert [row["primitive"] for row in support["primitives"]] == ["pause_job"]
    assert (task, snapshot) == before
    missing = validate_primitive_support(task=task, recovery_des_model=model, recovery_snapshot={},
        validate_primitive=lambda **_: pytest.fail("Missing starting facts reached capability checks"))
    assert missing["feasibility_status"] == "NEEDS_CONTEXT"
    assert missing["context_requests"] == ["/resources/M2@localhost/recovery_snapshot"]


def test_union_binds_successor_fields_before_preserved_fields():
    model = {"state_variables": {name: {"scope": "resource", "domain": [None]}
                                  for name in ("active_job", "selected_job")},
             "events": [{"event_name": "select", "updates": {"selected_job": {"set_from_param": "job_id"}},
                         "guards": {"active_job": {"equals_from_param": "job_id"}}, "primitive_support": []}]}
    task = {"expected_start_state": {"active_job": "job", "selected_job": None},
            "expected_end_state": {"active_job": "job", "selected_job": "job"}}
    assert derive_primitive_union(task=task, recovery_des_model=model)["parameter_bindings"] == {"job_id": "job"}
    model["state_variables"]["selected_job"]["domain_complete"] = True
    assert not derive_primitive_union(task=task, recovery_des_model=model)["matching_transitions"]


@pytest.mark.parametrize("change", ["missing_model", "missing_events", "missing_declaration", "unsupported_update",
                                  "unsupported_guard", "malformed_event", "model_incomplete"])
def test_union_incomplete_semantics_remain_needs_context(change):
    task, model = _task(), _model()
    if change == "missing_model":
        model = {}
    elif change == "missing_events":
        model.pop("events")
    elif change == "missing_declaration":
        task["expected_end_state"]["missing"] = None
    elif change == "unsupported_update":
        model["events"][0]["updates"]["resource_state"] = {"expression": "idle"}
    elif change == "unsupported_guard":
        task["expected_end_state"]["held_part"] = None
        model["events"][0]["guards"]["held_part"] = {"predicate": "unknown"}
    elif change == "malformed_event":
        model["events"].append(None)
    else:
        model["model_complete"] = False
    result = validate_primitive_support(task=task, recovery_des_model=model, recovery_snapshot={},
        validate_primitive=lambda **_: {"allowed": True, "feasibility_status": "FEASIBLE"})
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["primitive_support"]["missing_context"]
    assert f"/resources/{task['resource_jid']}/recovery_des_model" in result["context_requests"]


def test_union_complete_empty_events_differs_from_incomplete_events():
    model = _model()
    model["events"] = []
    assert _check(model=model)["feasibility_status"] == "INFEASIBLE"
    result = validate_primitive_support(task=_task(), recovery_des_model=model, recovery_snapshot={},
        validate_primitive=Mock(), model_complete=False)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("field", ["resource_state", "held_part"])
def test_union_unsupported_public_guards_remain_unresolved_when_updated_or_unrequested(field):
    model = _model()
    model["events"][0]["guards"][field] = {"predicate": "unavailable"}
    result = _check(model=model)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert any("unsupported public resource guard" in row for row in result["primitive_support"]["missing_context"])


def test_union_unrequested_public_update_needs_supported_semantics():
    model = _model()
    model["events"][0]["updates"]["held_part"] = {"expression": "owner_value"}
    result = _check(model=model)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert any("unsupported public resource update" in row for row in result["primitive_support"]["missing_context"])


def test_union_private_guards_remain_explicit_composition_prerequisites():
    model = _model()
    model["state_variables"]["task_ctx.part_name"] = {"scope": "resource", "domain": [None], "private": True}
    guard = {"predicate": "owner_private_predicate"}
    model["events"][0]["guards"]["task_ctx.part_name"] = guard
    result = _check(model=model)
    assert result["allowed"]
    assert result["primitive_support"]["matching_transitions"][0]["composition_prerequisites"]["task_ctx.part_name"] == guard


@pytest.mark.parametrize("location_type", ["current_location", "reachable_location", "part_location", "unknown"])
def test_union_existing_location_bindings_use_declared_domains_without_current_predecessor(location_type):
    model, task = _model(), _task()
    model["events"] = model["events"][:1]
    model["state_variables"].update(resource_location={"scope": "resource", "domain": ["Buffer For Machined parts"]},
                                     part_location={"scope": "part", "domain": ["Buffer For Machined parts"]})
    event = model["events"][0]
    event["updates"]["resource_location"] = {"set_from_param": "destination_location"}
    event["parameter_bindings"] = {"destination_location": {"location_type": location_type}}
    task["expected_end_state"]["resource_location"] = "Buffer For Machined parts"
    result = _check(task=task, model=model)
    if location_type == "unknown":
        assert result["feasibility_status"] == "NEEDS_CONTEXT"
    else:
        assert result["allowed"]
        task["expected_end_state"]["resource_location"] = "undeclared"
        assert _check(task=task, model=model)["feasibility_status"] == "INFEASIBLE"


def test_union_replays_saved_buffer_proposal_without_chaining_or_source_changes(monkeypatch):
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
    from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryResource

    records = Path(__file__).parents[1] / "cais_spade_llm/monitor/environment_runs/recovery"
    capture_path = records / "1c3bec8faca6495cb1b0e22046277d5c/observations.json"
    turn_path = records / "offline_30a404b09fc54a858590659ded262b8b/generation_0001/recovery_outline/turns/turn_00004_outline.json"
    if not capture_path.is_file() or not turn_path.is_file():
        pytest.skip("The user-owned capture and recorded proposal are not available")
    source = capture_path.read_bytes()
    saved = json.loads(source)
    turn = json.loads(turn_path.read_bytes())
    task = deepcopy(turn["next_transition"])
    task["expected_start_state"] = deepcopy(turn["transition_validation"]["findings"][0]["expected_start_state"])
    assert task["event_name"] == "place"
    assert task["expected_end_state"]["part_state"] == "ready"
    resources = saved["prepared_recovery_request"]["recovery_resources"]
    assert len(resources) == 12
    entry = resources[task["resource_jid"]]
    model = entry["recovery_des_model"]
    support = derive_primitive_union(task=task, recovery_des_model=model)
    assert support["missing_context"] == []
    assert [row["event_name"] for row in support["matching_transitions"]] == ["place_insert"]
    assert [row["primitive"] for row in support["primitives"]] == ["move_cartesian", "release_part", "move_relative"]
    assert set(support["deferred_valuation_fields"]) == {"part_state", "part_location"}
    live = Mock(side_effect=AssertionError("Recorded capability validation must not construct a live robot"))
    monkeypatch.setattr(RobotAgent, "__init__", live)
    owner = SavedRecoveryResource(task["resource_jid"], entry, saved)
    result = owner.validate_recovery_primitive_support(
        task=task, recovery_des_model=model, recovery_snapshot=entry["recovery_snapshot"],
        physical_input={"primitive_support_semantics": "successor_primitive_union"})
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["primitive_support"]["semantics"] == "successor_primitive_union"
    assert "trace" not in result["primitive_support"]
    assert any(row["unresolved_parameters"] for row in result["missing_parameters"])
    root = f"/resources/{task['resource_jid']}/resource_evidence"
    assert root + "/tcp_observation/tcp_offset_z" in result["context_requests"]
    optional = next(row for row in result["primitive_support"]["primitives"] if row["primitive"] == "move_cartesian")
    assert optional["feasibility_status"] == "NOT_APPLICABLE"
    assert optional["allowed"] is False
    release = next(row for row in result["primitive_support"]["primitives"] if row["primitive"] == "release_part")
    assert release["feasibility_status"] == "FEASIBLE"
    assert "assembly_slot" not in release["params"]
    assert release["parameter_witness"]["omitted_params"] == ["assembly_slot"]
    assert capture_path.read_bytes() == source
    assert saved == json.loads(source)
    live.assert_not_called()

    # Explicit synthetic supplementary facts exercise the repaired interface;
    # they are not written into, or reported as observations from, this capture.
    supplemented = deepcopy(entry)
    supplemented["resource_evidence"] = {
        "state": {"_task_ctx": {"destination_location": "Buffer For Machined parts",
                                 "part_name": "KET4_Square_4mm", "model_name": "KET4_Square_4mm",
                                 "insert_pose": {"x": -.32, "y": .5, "z": 1.27, "qx": 0., "qy": 1., "qz": 0., "qw": 0.},
                                 "travel_z": 1.51}},
        "controller_config": {"move_group": {"frame_id": "world"}}, "frame": "world",
        "source": {"kind": "synthetic regression fixture"},
    }
    complete_owner = SavedRecoveryResource(task["resource_jid"], supplemented, saved)
    complete = complete_owner.validate_recovery_primitive_support(task=task, recovery_des_model=model,
        recovery_snapshot=entry["recovery_snapshot"], physical_input={"primitive_support_semantics": "successor_primitive_union"})
    assert complete["feasibility_status"] == "FEASIBLE", complete
    assert set(complete["primitive_support"]["deferred_valuation_fields"]) == {"part_state", "part_location"}
    assert "trace" not in complete["primitive_support"]
    assert capture_path.read_bytes() == source


def _scoped_parameter_case():
    model = _model()
    model["events"] = model["events"][:1]
    model["events"][0]["primitive_support"] = [{"primitive": "move_cartesian", "params": {
        axis: {"$step": "target", "path": [axis]} for axis in ("x", "y", "z")}}]
    model["primitive_catalog"] = [{"name": "move_cartesian", "required_params": ["x", "y", "z"],
                                  "params": {axis: {"type": "number"} for axis in ("x", "y", "z")}}]
    witness = {"primitive": "move_cartesian", "event_name": "move_home", "step_index": 0,
               "params": {"x": .1, "y": .2, "z": 1.3}, "source": "saved_observations",
               "captured_at": "2026-10-08", "revision": 7}
    return model, witness


def test_owner_parameter_resolution_supports_nonrobot_declarations_and_exact_sources():
    from cais_spade_llm.resources.recovery_parameter_evidence import prepare_resource_primitive_parameters

    task = {"resource_jid": "printer@localhost", "event_name": "new_name",
            "expected_start_state": {"temperature": 20}, "expected_end_state": {"temperature": 50}}
    model = {"state_variables": {"temperature": {"domain": [20, 50]}}, "events": [
        {"event_name": "warm", "updates": {"temperature": {"set": 50}},
         "primitive_support": [{"primitive": "heat", "params": {"target": {"$state": "limits", "path": ["target"]}}}]}],
        "primitive_catalog": [{"name": "heat", "required_params": ["target"], "params": {"target": {"type": "number"}}}]}
    snapshot = {"temperature": 20}
    evidence = {"state": {"limits": {"target": 50}}, "source": {"revision": 3}}
    before = deepcopy(evidence)
    prepared = prepare_resource_primitive_parameters(task=task, recovery_des_model=model, recovery_snapshot=snapshot,
        resource_evidence=evidence, resource_jid=task["resource_jid"])
    calls = []
    row = validate_primitive_support(task=task, recovery_des_model=model, recovery_snapshot=snapshot,
        parameter_witnesses=prepared["parameter_witnesses"], validate_primitive=lambda **kwargs: calls.append(kwargs) or {
            "allowed": True, "feasibility_status": "FEASIBLE"})
    assert row["feasibility_status"] == "FEASIBLE"
    assert calls == [{"primitive": "heat", "params": {"target": 50}}]
    assert evidence == before
    missing = prepare_resource_primitive_parameters(task=task, recovery_des_model=model, recovery_snapshot=snapshot,
        resource_evidence={"state": {}}, resource_jid=task["resource_jid"])
    assert missing["context_requests"] == ["/resources/printer@localhost/resource_evidence/state/limits/target"]


@pytest.mark.parametrize("declaration_matches", [True, False])
def test_recorded_task_defaults_resolve_optional_capability_parameters(declaration_matches):
    from cais_spade_llm.resources.recovery_parameter_evidence import prepare_resource_primitive_parameters
    from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry

    registered = robot_task_registry()["pick_approach"]
    declarations = {row.name: row.param_schema() for row in registered.arguments}
    if not declaration_matches:
        declarations["speed"]["description"] = "Different recorded declaration"
    task = {"resource_jid": "robot@localhost", "event_name": "new_identifier",
            "expected_end_state": {"resource_state": "at_pick"}}
    snapshot = {"resource_state": "idle"}
    model = {"state_variables": {"resource_state": {"domain": ["idle", "at_pick"]}},
        "events": [{"event_name": "pick_approach", "updates": {"resource_state": {"set": "at_pick"}},
            "composition_contract": {"params": declarations},
            "primitive_support": [{"primitive": "move_cartesian", "params": {
                "x": 10., "y": 0., "z": 1., "speed": {"$arg": "speed"}}}]}],
        "primitive_catalog": [{"name": "move_cartesian", "required_params": ["x", "y", "z"],
            "params": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
                       "speed": {"type": ["number", "null"]}}}]}
    before = deepcopy(model)
    prepared = prepare_resource_primitive_parameters(task=task, recovery_des_model=model,
        recovery_snapshot=snapshot, resource_evidence={"resource_type": "robot"},
        resource_jid=task["resource_jid"])
    witness = prepared["parameter_witnesses"][0]
    if declaration_matches:
        assert witness["params"]["speed"] is None
        assert witness["parameter_default_evidence"]["speed"]["source"] == "matching RobotTaskArgument callable default"
    else:
        assert "speed" not in witness["params"]
    checked = validate_primitive_support(task=task, recovery_des_model=model, recovery_snapshot=snapshot,
        parameter_witnesses=prepared["parameter_witnesses"],
        validate_primitive=lambda **_: {"allowed": False, "feasibility_status": "INFEASIBLE", "reason": "Outside declared reach"})
    assert checked["feasibility_status"] == ("INFEASIBLE" if declaration_matches else "NEEDS_CONTEXT")
    if declaration_matches:
        assert checked["reason"] == "INFEASIBLE: Outside declared reach"
    assert model == before


def _place_output_parameter_case():
    task, model = _task(), _model()
    task["part_name"] = "KET4_Square_4mm"
    producer = {"primitive": "compute_place_targets", "params": {
        "part_name": {"$arg": "part_name"}, "pick_ctx": {"$state": "_task_ctx", "path": []}}}
    consumer = {"primitive": "move_cartesian", "params": {
        axis: {"$step": "placement", "path": ["approach_pose", axis]} for axis in ("x", "y", "z")}}
    event = model["events"][0]
    event["primitive_support"] = [producer, consumer]
    event["composition_contract"] = {"steps": [{"id": "targets", "op": "compute_place_targets",
        "params": deepcopy(producer["params"]), "store_as": "placement"}]}
    model["events"] = [event]
    model["primitive_catalog"] = [
        {"name": "compute_place_targets", "required_params": ["part_name", "pick_ctx"],
         "params": {"part_name": {"type": "string"}, "pick_ctx": {"type": "object"}}},
        {"name": "move_cartesian", "required_params": ["x", "y", "z"],
         "params": {axis: {"type": "number"} for axis in ("x", "y", "z")}}]
    evidence = {"execution_mode": "simulation", "resource_type": "robot",
        "state": {"_task_ctx": {"part_name": task["part_name"], "pick_tcp_z": 1.1}},
        "source": {"revision": 9}}
    snapshot = {"held_part": task["part_name"], "current_pose": [0., .2, 1.3, 0., 0., 0., 1.]}
    return task, model, snapshot, evidence


@pytest.mark.parametrize("case", ["supported", "missing_fact", "null_output", "conflicting_output", "wrong_producer"])
def test_robot_calculation_outputs_ground_parameters_without_advancing_state(monkeypatch, case):
    from cais_spade_llm.resources.robot.recovery_capability import prepare_robot_primitive_parameters
    from cais_spade_llm.resources.robot import recovery_capability

    task, model, snapshot, evidence = _place_output_parameter_case()
    output = {"approach_pose": {"x": .1, "y": .2, "z": 1.5}}
    if case == "null_output":
        output["approach_pose"]["z"] = None
    elif case == "conflicting_output":
        evidence["step_outputs"] = {"placement": {"approach_pose": {"x": 9., "y": .2, "z": 1.5}}}
    elif case == "wrong_producer":
        model["events"][0]["composition_contract"]["steps"][0]["params"]["part_name"] = "gear_small"
    before = deepcopy((task, model, snapshot, evidence))
    calls = []

    def calculate(**kwargs):
        calls.append(deepcopy(kwargs))
        if case == "missing_fact":
            raise ValueError("Recorded insertion depth is unavailable")
        return {"output": deepcopy(output), "source": {"kind": "calculated", "calculation": "recorded owner calculation"}}

    monkeypatch.setattr(recovery_capability, "_recovery_place_step_output", calculate)
    prepared = prepare_robot_primitive_parameters(task=task, recovery_des_model=model, recovery_snapshot=snapshot,
        resource_evidence=evidence, physical_evidence={}, resource_jid=task["resource_jid"])
    assert (task, model, snapshot, evidence) == before
    assert all(row["recovery_snapshot"] == snapshot for row in calls)
    if case == "supported":
        assert prepared["context_requests"] == []
        motion = next(row for row in prepared["parameter_witnesses"] if row["primitive"] == "move_cartesian")
        assert motion["params"] == output["approach_pose"]
        assert motion["step_output_calculations"][0]["store_as"] == "placement"
        checked = validate_primitive_support(task=task, recovery_des_model=model, recovery_snapshot=snapshot,
            parameter_witnesses=prepared["parameter_witnesses"],
            validate_primitive=lambda **_: {"allowed": True, "feasibility_status": "FEASIBLE"})
        assert checked["feasibility_status"] == "FEASIBLE"
        assert "trace" not in checked["primitive_support"]
        assert "step_outputs" not in prepared["calculated_resource_evidence"]
    elif case == "null_output":
        motion = next(row for row in prepared["parameter_witnesses"] if row["primitive"] == "move_cartesian")
        assert "z" in motion["params"] and motion["params"]["z"] is None
    else:
        assert prepared["context_requests"]
        assert bool(prepared["conflicts"]) is (case == "conflicting_output")
        if case == "wrong_producer":
            assert calls == []


def test_calculated_step_outputs_keep_same_event_distinct_bindings_separate(monkeypatch):
    from cais_spade_llm.resources import recovery_parameter_evidence as parameters

    task, model, snapshot, evidence = _place_output_parameter_case()
    support = derive_primitive_union(task=task, recovery_des_model=model)
    first = support["primitives"][1]["contributors"][0]
    second = deepcopy(first)
    first["parameter_bindings"]["speed"] = .1
    second["parameter_bindings"]["speed"] = .2
    support["primitives"] = [{"primitive": "move_cartesian", "contributors": [first, second]}]
    monkeypatch.setattr(parameters, "derive_primitive_union", lambda **_: deepcopy(support))
    calculations = [{"event_name": row["event_name"], "parameter_bindings": deepcopy(row["parameter_bindings"]),
        "store_as": "placement", "output": {"approach_pose": {"x": speed, "y": .2, "z": 1.5}}}
        for row, speed in ((first, .1), (second, .2))]
    prepared = parameters.prepare_resource_primitive_parameters(task=task, recovery_des_model=model,
        recovery_snapshot=snapshot, resource_evidence=evidence, resource_jid=task["resource_jid"],
        calculated_step_outputs=calculations)
    assert prepared["context_requests"] == []
    assert prepared["conflicts"] == []
    assert [row["params"]["x"] for row in prepared["parameter_witnesses"]] == [.1, .2]
    assert [row["parameter_bindings"]["speed"] for row in prepared["parameter_witnesses"]] == [.1, .2]


def _recovery_buffer_calculation_case():
    part, destination = "KET4_Square_4mm", "Buffer For Machined parts"
    task = {"part_name": part, "resource_jid": "recovery-resource-3@localhost",
            "expected_end_state": {"resource_location": destination}}
    context = {"part_name": part, "pick_tcp_z": 1.07, "tz": 1.017, "tcp_offset_z": -.218}
    snapshot = {"current_pose": [-.32, .5, 1.5, 0., 1., 0., 0.]}
    evidence = {"execution_mode": "simulation", "state": {"_task_ctx": context},
        "source": {"revision": 11}, "controller_config": {"parts_tuning": {"insertion_depth_m": .02}},
        "inputs": {"geometry": {"parts": {"dimensions_m": {part: [.004, .004, .05]},
            "heights_m": {part: .05}, "model_map": {part: part}}},
            "scene": {destination: {"pickup_pose": [-.32, .5, 1.017]}}}}
    params = {"part_name": part, "destination_location": destination, "pick_ctx": deepcopy(context),
              "z_adjustment_m": 0., "assembly_board_v1_aruco": None}
    return {"task": task, "recovery_snapshot": snapshot, "resource_evidence": evidence,
            "physical_evidence": {}, "params": params}


@pytest.mark.parametrize("geometry", ["omitted", "null", "recorded"])
def test_recovery_buffer_producer_calculates_recorded_pose_fields(geometry):
    from cais_spade_llm.resources.robot.recovery_capability import _recovery_place_step_output
    from cais_spade_llm.recovery_framework.workflow_execution import _destination_geometry

    args = _recovery_buffer_calculation_case()
    if geometry == "null":
        args["params"]["product_geometry"] = None
    elif geometry == "recorded":
        part = args["task"]["part_name"]
        args["physical_evidence"] = {"jids": {"ur5e-3": args["task"]["resource_jid"]}}
        args["params"]["product_geometry"] = _destination_geometry(SimpleNamespace(
            inputs=args["resource_evidence"]["inputs"], product_name="assembly_board-v1",
            geometry={part: {"dimensions_m": [.004, .004, .05], "model_name": part}}),
            part, args["params"]["destination_location"], "ur5e-3")
    before = deepcopy(args)
    output = _recovery_place_step_output(**args)
    assert args == before
    assert output["output"]["target_pose"] == {"x": -.32, "y": .5, "z": pytest.approx(1.288),
                                                 "qx": 0., "qy": 1., "qz": 0., "qw": 0.}
    assert output["output"]["approach_pose"]["z"] == pytest.approx(1.338)
    assert output["source"]["kind"] == "calculated"
    assert output["target_calculation"]["command_sent"] is False
    assert "slot_x" not in output["output"]
    assert "speed" not in output["output"]
    assert ("product_geometry" in args["params"]) is (geometry != "omitted")


@pytest.mark.parametrize("defect", ["wrong_part", "wrong_destination", "wrong_context", "adjustment",
    "marker", "unknown_parameter", "conflicting_geometry", "missing_dimensions", "null_depth", "missing_pose"])
def test_recovery_buffer_producer_does_not_invent_or_override_source_facts(defect):
    from cais_spade_llm.resources.robot.recovery_capability import _recovery_place_step_output

    args = _recovery_buffer_calculation_case()
    if defect == "wrong_part":
        args["params"]["part_name"] = "gear_small"
    elif defect == "wrong_destination":
        args["params"]["destination_location"] = "assembly_board-v1"
    elif defect == "wrong_context":
        args["params"]["pick_ctx"] = {}
    elif defect == "adjustment":
        args["params"]["z_adjustment_m"] = .01
    elif defect == "marker":
        args["params"]["assembly_board_v1_aruco"] = {}
    elif defect == "unknown_parameter":
        args["params"]["unrecorded"] = 1
    elif defect == "conflicting_geometry":
        args["physical_evidence"] = {"jids": {"ur5e-3": args["task"]["resource_jid"]}}
        args["params"]["product_geometry"] = {"target_origin_pose": {"x": 9., "y": 9., "z": 9.}}
    elif defect == "missing_dimensions":
        args["resource_evidence"]["inputs"]["geometry"]["parts"].pop("dimensions_m")
    elif defect == "null_depth":
        args["resource_evidence"]["controller_config"]["parts_tuning"]["insertion_depth_m"] = None
    else:
        args["recovery_snapshot"].pop("current_pose")
    before = deepcopy(args)
    with pytest.raises(ValueError):
        _recovery_place_step_output(**args)
    assert args == before


def test_recovery_producer_keeps_declared_optional_speed_omitted(monkeypatch):
    from cais_spade_llm.resources.robot import recovery_capability

    task, model, snapshot, evidence = _place_output_parameter_case()
    model["events"][0]["primitive_support"][1]["params"]["speed"] = {"$arg": "speed"}
    model["primitive_catalog"][1]["params"]["speed"] = {"type": ["number", "null"], "default": None}
    monkeypatch.setattr(recovery_capability, "_recovery_place_step_output", lambda **_: {
        "output": {"approach_pose": {"x": .1, "y": .2, "z": 1.5}}, "source": {"kind": "calculated"}})
    prepared = recovery_capability.prepare_robot_primitive_parameters(task=task, recovery_des_model=model,
        recovery_snapshot=snapshot, resource_evidence=evidence, resource_jid=task["resource_jid"])
    checked = validate_primitive_support(task=task, recovery_des_model=model, recovery_snapshot=snapshot,
        parameter_witnesses=prepared["parameter_witnesses"],
        validate_primitive=lambda **_: {"allowed": True, "feasibility_status": "FEASIBLE"})
    assert checked["feasibility_status"] == "FEASIBLE"
    assert all("speed" not in row["params"] for row in checked["primitive_support"]["primitives"])


def test_nominal_placement_producer_retains_exact_nominal_calculator(monkeypatch):
    from cais_spade_llm.resources import resource_safety_preparation
    from cais_spade_llm.resources.robot import recovery_capability

    task, model, snapshot, evidence = _place_output_parameter_case()
    task["required_task_contract"] = model["events"][0]["event_name"]
    calculate = Mock(return_value={"output": {"approach_pose": {"x": .1, "y": .2, "z": 1.5}},
                                   "source": {"kind": "calculated"}})
    monkeypatch.setattr(resource_safety_preparation, "_outline_nominal_place_targets", calculate)
    monkeypatch.setattr(recovery_capability, "_recovery_place_step_output", Mock(side_effect=AssertionError))
    prepared = recovery_capability.prepare_robot_primitive_parameters(task=task, recovery_des_model=model,
        recovery_snapshot=snapshot, resource_evidence=evidence, resource_jid=task["resource_jid"])
    assert prepared["context_requests"] == []
    calculate.assert_called_once()


def test_robot_producer_outputs_keep_same_event_distinct_bindings_separate(monkeypatch):
    from cais_spade_llm.resources import recovery_parameter_evidence
    from cais_spade_llm.resources.robot import recovery_capability

    task, model, snapshot, evidence = _place_output_parameter_case()
    event = model["events"][0]
    event["primitive_support"][0]["params"]["z_adjustment_m"] = {"$arg": "speed"}
    event["composition_contract"]["steps"][0]["params"]["z_adjustment_m"] = {"$arg": "speed"}
    support = derive_primitive_union(task=task, recovery_des_model=model)
    transition = support["matching_transitions"][0]
    support["matching_transitions"] = [{**deepcopy(transition), "parameter_bindings": {
        **transition["parameter_bindings"], "speed": value}} for value in (.1, .2)]
    for primitive in support["primitives"]:
        contributor = primitive["contributors"][0]
        primitive["contributors"] = [{**deepcopy(contributor), "parameter_bindings": {
            **contributor["parameter_bindings"], "speed": value}} for value in (.1, .2)]
    for module in (recovery_parameter_evidence, recovery_capability):
        monkeypatch.setattr(module, "derive_primitive_union", lambda **_: deepcopy(support))
    calls = []

    def calculate(**kwargs):
        calls.append(deepcopy(kwargs))
        return {"output": {"approach_pose": {"x": kwargs["params"]["z_adjustment_m"], "y": .2, "z": 1.5}},
                "source": {"calculation": "independent recorded binding fixture"}}

    monkeypatch.setattr(recovery_capability, "_recovery_place_step_output", calculate)
    result = recovery_capability.prepare_robot_primitive_parameters(task=task, recovery_des_model=model,
        recovery_snapshot=snapshot, resource_evidence=evidence, physical_evidence={}, resource_jid=task["resource_jid"])
    assert len(calls) == 2
    assert all(row["recovery_snapshot"] == snapshot for row in calls)
    assert result["context_requests"] == []
    motions = [row for row in result["parameter_witnesses"] if row["primitive"] == "move_cartesian"]
    assert [(row["parameter_bindings"]["speed"], row["params"]["x"]) for row in motions] == [(.1, .1), (.2, .2)]


def test_union_does_not_consume_another_assignment_parameter_witness():
    model, witness = _scoped_parameter_case()
    witness["parameter_bindings"] = {"speed": .2}
    checked = Mock(return_value={"allowed": True, "feasibility_status": "FEASIBLE"})
    result = validate_primitive_support(task=_task(), recovery_des_model=model, recovery_snapshot={},
        validate_primitive=checked, parameter_witnesses=[witness])
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    checked.assert_not_called()


def test_calculated_parameter_witness_cannot_survive_changed_snapshot():
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    model, witness = _scoped_parameter_case()
    witness["snapshot_fingerprint"] = recovery_validation_fingerprint({"current_state": "different"})
    checked = Mock(return_value={"allowed": True, "feasibility_status": "FEASIBLE"})
    row = validate_primitive_support(task=_task(), recovery_des_model=model, recovery_snapshot={},
        parameter_witnesses=[witness], validate_primitive=checked)
    assert row["feasibility_status"] == "NEEDS_CONTEXT"
    assert "different observations" in row["reason"]
    checked.assert_not_called()


def test_union_scoped_witness_retains_provenance_without_source_mutation():
    model, witness = _scoped_parameter_case()
    before = deepcopy((model, witness))
    check = Mock(return_value={"allowed": True, "feasibility_status": "FEASIBLE"})
    result = validate_primitive_support(task=_task(), recovery_des_model=model, recovery_snapshot={},
        validate_primitive=check, parameter_witnesses=[witness])
    assert result["allowed"]
    check.assert_called_once_with(primitive="move_cartesian", params=witness["params"])
    assert result["primitive_support"]["primitives"][0]["parameter_witness"] == witness
    assert (model, witness) == before


@pytest.mark.parametrize("source", [None, "", " ", 1])
def test_union_scoped_witness_requires_recorded_source(source):
    model, witness = _scoped_parameter_case()
    witness["source"] = source
    check = Mock()
    result = validate_primitive_support(task=_task(), recovery_des_model=model, recovery_snapshot={},
        validate_primitive=check, parameter_witnesses=[witness])
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert "source" in result["reason"]
    check.assert_not_called()


@pytest.mark.parametrize("scope,value", [("event_name", "another_event"), ("step_index", 1), ("primitive", "move_relative")])
def test_union_parameter_witness_does_not_cross_its_scope(scope, value):
    model, witness = _scoped_parameter_case()
    witness[scope] = value
    check = Mock()
    result = validate_primitive_support(task=_task(), recovery_des_model=model, recovery_snapshot={},
        validate_primitive=check, parameter_witnesses=[witness])
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    check.assert_not_called()
    assert result["context_requests"] == [f"/resources/{_task()['resource_jid']}/parameter_witnesses"]
    assert result["missing_parameters"][0]["unresolved_parameters"]["x"] == {"$step": "target", "path": ["x"]}


@pytest.mark.parametrize("conflict", ["contributor", "authored", "flat"])
def test_union_conflicting_parameter_witness_is_never_overwritten(conflict):
    model, witness = _scoped_parameter_case()
    task = _task()
    extra = {}
    if conflict == "contributor":
        task["params"] = {"x": .5}
    elif conflict == "authored":
        model["events"][0]["primitive_support"][0]["params"]["x"] = .5
    else:
        extra["parameter_evidence"] = {"x": .5}
    check = Mock()
    result = validate_primitive_support(task=task, recovery_des_model=model, recovery_snapshot={},
        validate_primitive=check, parameter_witnesses=[witness], **extra)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    check.assert_not_called()
    assert "conflict" in result["missing_parameters"][0]["reason"]


def test_union_one_scoped_alternative_witness_is_sufficient_for_same_primitive():
    model, witness = _scoped_parameter_case()
    second = deepcopy(witness)
    second["params"]["x"] = .5
    check = Mock(side_effect=lambda **kw: {"allowed": kw["params"]["x"] == .5,
        "feasibility_status": "FEASIBLE" if kw["params"]["x"] == .5 else "INFEASIBLE"})
    result = validate_primitive_support(task=_task(), recovery_des_model=model, recovery_snapshot={},
        validate_primitive=check, parameter_witnesses=[witness, second])
    assert result["allowed"]
    assert check.call_count == 2
    assert result["primitive_support"]["primitives"][0]["params"] == second["params"]


def test_union_checks_fixed_snapshot_without_chaining_or_product_completion():
    model, task = _model(), _task()
    model["state_variables"]["part_state"] = {"scope": "part", "domain": ["ready", "assembled"]}
    task["expected_end_state"]["part_state"] = "assembled"
    for entry in model["primitive_catalog"]:
        entry["capability_constraints"] = {"current_state": {"equals": "failed"}}
        entry["preconditions"] = {"current_state": {"equals": "idle"}}
        entry["effects"] = {"current_state": {"set": "idle"}}
        entry["required_evidence"] = {"requires": ["unestablished_sequence_fact"]}
    before = deepcopy((model, task))
    result = _check(model=model, task=task)
    assert result["allowed"]
    assert result["primitive_support"]["semantics"] == "successor_primitive_union"
    assert result["primitive_support"]["deferred_valuation_fields"] == ["part_state"]
    assert "trace" not in result["primitive_support"]
    assert "derived_end_state" not in result["primitive_support"]
    assert (model, task) == before


def test_union_parameter_limit_is_shared_across_contributors_and_counts_invalid_schema():
    model = _model()
    first = model["events"][0]
    first["primitive_support"][0]["params"] = {}
    other = deepcopy(first)
    other["event_name"] = "other_transition"
    model["events"] = [first, other]
    model["primitive_catalog"][0]["params"]["pose_name"] = {"type": "string", "enum": [1, 2]}
    check = Mock()
    result = validate_primitive_support(task=_task(), recovery_des_model=model, recovery_snapshot={},
        validate_primitive=check, max_parameter_trials=3)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["primitive_support"]["search"] == {"limit": 3, "parameter_trials": 3, "limit_reached": True}
    check.assert_not_called()


def test_union_parameter_limit_is_shared_across_primitive_identities():
    check = Mock(return_value={"allowed": True, "feasibility_status": "FEASIBLE"})
    result = validate_primitive_support(task=_task(), recovery_des_model=_model(), recovery_snapshot={},
        validate_primitive=check, max_parameter_trials=1)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert check.call_count == 1
    assert result["primitive_support"]["search"]["limit_reached"]


@pytest.mark.parametrize("bound", [0, 4097, True, None])
def test_union_rejects_invalid_parameter_bounds(bound):
    result = validate_primitive_support(task=_task(), recovery_des_model=_model(), recovery_snapshot={},
        validate_primitive=Mock(), max_parameter_trials=bound)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"


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


def _transition_derivation_case():
    model = {
        "descriptor_fingerprint": "declared-primitives",
        "state_variables": {
            "resource_state": {"scope": "resource", "domain": ["faulted", "prepared", "idle"]},
            "part_state": {"scope": "part", "domain": ["raw", "done"]},
        },
        "events": [],
        "primitive_catalog": [
            {"name": "prepare", "params": {}, "required_params": [],
             "preconditions": {"current_state": {"equals": "faulted"}},
             "effects": {"current_state": {"set": "prepared"}}},
            {"name": "finish", "params": {}, "required_params": [],
             "preconditions": {"current_state": {"equals": "prepared"}},
             "effects": {"current_state": {"set": "idle"}, "part_state": {"set": "done"}}},
        ],
    }
    task = {"event_name": "unexpected_disruption_recovery", "resource_jid": "M1@localhost",
            "part_name": "gear_small", "expected_start_state": {"resource_state": "faulted", "part_state": "raw"},
            "expected_end_state": {"resource_state": "idle", "part_state": "done"}}
    return dict(task=task, recovery_des_model=model, recovery_snapshot={"current_state": "faulted"},
                part_context={"part_state": "raw"})


@pytest.mark.parametrize("event_name", ["unexpected_disruption_recovery", "evt_q7", "Conveyor breakdown"])
def test_transition_derivation_accepts_novel_event_without_registered_successor(event_name):
    inputs = _transition_derivation_case()
    inputs["task"]["event_name"] = event_name
    original = deepcopy(inputs)
    result = validate_recovery_transition(**inputs)
    assert result["feasibility_status"] == "FEASIBLE"
    support = result["primitive_support"]
    assert [row["primitive"] for row in support["primitives"]] == ["prepare", "finish"]
    assert support["trace"][1]["start_state"]["resource"]["resource_state"] == "prepared"
    assert support["derived_end_state"] == inputs["task"]["expected_end_state"]
    assert support["projected_part_context"]["part_state"] == "done"
    assert result["execution_authorized"] is False
    assert inputs == original


def test_transition_derivation_checks_intermediate_capability_state():
    inputs = _transition_derivation_case()
    calls = []

    def evaluate(**kwargs):
        calls.append(deepcopy(kwargs))
        expected = "faulted" if kwargs["primitive"] == "prepare" else "prepared"
        assert kwargs["recovery_snapshot"]["current_state"] == expected
        return {"allowed": True, "feasibility_status": "FEASIBLE"}

    assert validate_recovery_transition(**inputs, validate_primitive=evaluate)["allowed"]
    assert [call["primitive"] for call in calls] == ["prepare", "finish"]


@pytest.mark.parametrize("field,status", [("part_state", "NEEDS_CONTEXT"), ("resource_state", "INFEASIBLE")])
def test_transition_derivation_requires_every_end_field(field, status):
    inputs = _transition_derivation_case()
    effect = "current_state" if field == "resource_state" else field
    inputs["recovery_des_model"]["primitive_catalog"][-1]["effects"].pop(effect)
    result = validate_recovery_transition(**inputs)
    assert result["feasibility_status"] == status
    assert not result["derived_end_state"]


@pytest.mark.parametrize("actual,status", [(None, "FEASIBLE"), ("missing", "NEEDS_CONTEXT"), ("another_part", "INFEASIBLE")])
def test_transition_derivation_distinguishes_missing_from_explicit_null(actual, status):
    inputs = _transition_derivation_case()
    first = inputs["recovery_des_model"]["primitive_catalog"][0]
    first["preconditions"]["held_part"] = {"equals": None}
    if actual != "missing":
        inputs["recovery_snapshot"]["held_part"] = actual
    assert validate_recovery_transition(**inputs)["feasibility_status"] == status


@pytest.mark.parametrize("condition", [{"unknown": True}, {"exists": "false"}, {"in": "faulted"}])
def test_transition_derivation_rejects_unsupported_or_malformed_conditions(condition):
    inputs = _transition_derivation_case()
    inputs["recovery_des_model"]["primitive_catalog"][0]["preconditions"]["current_state"] = condition
    assert validate_recovery_transition(**inputs)["feasibility_status"] == "NEEDS_CONTEXT"


def test_transition_derivation_rejects_stale_start():
    inputs = _transition_derivation_case()
    inputs["recovery_snapshot"]["current_state"] = "idle"
    result = validate_recovery_transition(**inputs)
    assert result["feasibility_status"] == "INFEASIBLE"
    assert "starting fact" in result["reason"]


@pytest.mark.parametrize("field", ["_task_ctx", "task_ctx.part_name", "undeclared"])
def test_transition_derivation_rejects_private_or_unknown_candidate_fields(field):
    inputs = _transition_derivation_case()
    inputs["task"]["expected_end_state"][field] = "gear_small"
    assert validate_recovery_transition(**inputs)["feasibility_status"] == "INFEASIBLE"


def test_transition_derivation_does_not_mix_incompatible_parameter_witnesses():
    inputs = _transition_derivation_case()
    model = inputs["recovery_des_model"]
    model["primitive_catalog"] = [{
        "name": "set_fields", "params": {"value": {"enum": ["idle", "done"]}}, "required_params": ["value"],
        "effects": {"current_state": {"set_from_param": "value"}, "part_state": {"set_from_param": "value"}},
    }]
    for declaration in model["state_variables"].values():
        declaration["domain"].extend(["idle", "done"])
    result = validate_recovery_transition(**inputs)
    assert result["feasibility_status"] == "INFEASIBLE"


def test_transition_derivation_explicit_bindings_conflict_with_recorded_evidence():
    inputs = _transition_derivation_case()
    result = validate_recovery_transition(**inputs, parameter_evidence={"part_name": "KET4_Square_4mm"})
    assert result["feasibility_status"] == "INFEASIBLE"


def test_transition_derivation_custody_follows_declared_part_and_resource_effects():
    inputs = _transition_derivation_case()
    model, task = inputs["recovery_des_model"], inputs["task"]
    model["state_variables"].update({"held_part": {"scope": "resource", "domain": [None]},
        "part_holder_resource_jid": {"scope": "part", "domain": [None, "M1@localhost"]}})
    task["expected_start_state"] = {"held_part": None, "part_holder_resource_jid": None}
    task["expected_end_state"] = {"held_part": "gear_small", "part_holder_resource_jid": "M1@localhost"}
    inputs["recovery_snapshot"]["held_part"] = None
    inputs["part_context"]["part_holder_resource_jid"] = None
    model["primitive_catalog"] = [{"name": "declared_transfer", "params": {"part_name": {"type": "string"}},
        "required_params": ["part_name"],
        "preconditions": {"held_part": {"equals": None}, "part_holder_resource_jid": {"equals": None}},
        "effects": {"held_part": {"set_from_param": "part_name"}, "part_holder_resource_jid": {"set": "M1@localhost"}}}]
    result = validate_recovery_transition(**inputs)
    assert result["allowed"]
    assert result["derived_end_state"] == task["expected_end_state"]
    inputs["part_context"]["part_holder_resource_jid"] = "another_resource"
    task["expected_start_state"].pop("part_holder_resource_jid")
    assert validate_recovery_transition(**inputs)["feasibility_status"] == "INFEASIBLE"


def test_transition_derivation_required_evidence_has_ordered_producer():
    inputs = _transition_derivation_case()
    catalog = inputs["recovery_des_model"]["primitive_catalog"]
    catalog[1]["required_evidence"] = {"requires": [{"fact": "part_observed", "part_param": "part_name"}]}
    catalog[1]["params"] = {"part_name": {"type": "string"}}
    catalog.append({"name": "observe", "params": {"part_name": {"type": "string"}}, "required_params": ["part_name"],
                    "effects": {}, "required_evidence": {"establishes": [{"fact": "part_observed", "part_param": "part_name"}]}})
    result = validate_recovery_transition(**inputs)
    assert result["allowed"]
    names = [row["primitive"] for row in result["primitive_support"]["primitives"]]
    assert names.index("observe") < names.index("finish")


def test_transition_derivation_conditional_evidence_remains_unavailable():
    inputs = _transition_derivation_case()
    inputs["recovery_des_model"]["primitive_catalog"][0]["required_evidence"] = {
        "requires": [{"fact": "part_observed", "when": "unrecognized_condition"}]}
    assert validate_recovery_transition(**inputs)["feasibility_status"] == "NEEDS_CONTEXT"


def test_transition_derivation_limits_do_not_claim_infeasible():
    result = validate_recovery_transition(**_transition_derivation_case(), max_expansions=1)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["primitive_support"]["search"]["limit_reached"]


@pytest.mark.parametrize("resource_type", ["resource", "printer", "robot"])
def test_transition_derivation_uses_declared_fields_across_resource_types(resource_type):
    inputs = _transition_derivation_case()
    inputs["recovery_snapshot"]["resource_type"] = resource_type
    assert validate_recovery_transition(**inputs)["allowed"]


def test_transition_derivation_unknown_effect_does_not_restore_old_pose():
    inputs = _transition_derivation_case()
    old = {"x": 1.0, "y": 2.0, "z": 3.0}
    inputs["recovery_snapshot"]["current_pose"] = old
    inputs["recovery_snapshot"]["resource_facets"] = {"manipulator": {"current_pose": deepcopy(old)}}
    inputs["recovery_des_model"]["primitive_catalog"][0]["effects"]["current_pose"] = {"set_unknown": True}
    result = validate_recovery_transition(**inputs)
    assert result["allowed"]
    snapshot = result["primitive_support"]["projected_snapshot"]
    assert "current_pose" not in snapshot
    assert "current_pose" not in snapshot["resource_facets"]["manipulator"]


def test_transition_derivation_missing_relative_pose_is_not_zero():
    inputs = _transition_derivation_case()
    first = inputs["recovery_des_model"]["primitive_catalog"][0]
    first["params"] = {name: {"type": "number", "const": 0} for name in ("dx", "dy", "dz")}
    first["required_params"] = ["dx", "dy", "dz"]
    first["effects"]["current_pose"] = {"pose_relative_from_params": ["dx", "dy", "dz"]}
    result = validate_recovery_transition(**inputs)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert "Missing starting pose" in result["reason"]


def test_transition_derivation_missing_effect_catalog_is_not_capability_failure():
    inputs = _transition_derivation_case()
    del inputs["recovery_des_model"]["primitive_catalog"][0]["effects"]
    assert validate_recovery_transition(**inputs)["feasibility_status"] == "NEEDS_CONTEXT"


def test_transition_derivation_model_accepts_no_predefined_events(monkeypatch):
    from cais_spade_llm.resources import resource_primitives

    catalog = _transition_derivation_case()["recovery_des_model"]["primitive_catalog"]
    monkeypatch.setattr(resource_primitives, "_execution_catalog_for_snapshot", lambda owner: deepcopy(catalog))
    descriptor = {"state_variables": {"resource_state": {"scope": "resource", "domain": ["faulted", "idle"]}}}
    owner = SimpleNamespace(jid="M1@localhost", static_capabilities={})
    model = resource_primitives.build_recovery_des_model(owner, descriptor=descriptor, snapshot={"current_state": "faulted"})
    assert model["events"] == []
    assert model["primitive_catalog"] == catalog
    assert model["descriptor_fingerprint"]


def test_transition_derivation_updates_every_existing_snapshot_mirror():
    inputs = _transition_derivation_case()
    inputs["recovery_snapshot"].update({
        "resource_core": {"current_state": "faulted", "held_part": None},
        "resource_facets": {"manipulator": {"held_part": None}},
    })
    first = inputs["recovery_des_model"]["primitive_catalog"][0]
    first["effects"]["held_part"] = {"set": "gear_small"}

    def evaluate(**kwargs):
        if kwargs["primitive"] == "finish":
            snapshot = kwargs["recovery_snapshot"]
            assert snapshot["resource_core"]["current_state"] == "prepared"
            assert snapshot["resource_core"]["held_part"] == "gear_small"
            assert snapshot["resource_facets"]["manipulator"]["held_part"] == "gear_small"
        return {"allowed": True, "feasibility_status": "FEASIBLE"}

    assert validate_recovery_transition(**inputs, validate_primitive=evaluate)["allowed"]


def test_transition_derivation_removes_unknown_nested_snapshot_facts():
    inputs = _transition_derivation_case()
    inputs["recovery_snapshot"].update({
        "joint_positions": {"joint_a": 1.0},
        "resource_facets": {"manipulator": {"joint_positions": {"joint_a": 1.0}}},
    })
    inputs["recovery_des_model"]["primitive_catalog"][0]["effects"]["joint_positions.joint_a"] = {"set_unknown": True}
    result = validate_recovery_transition(**inputs)
    assert result["allowed"]
    snapshot = result["primitive_support"]["projected_snapshot"]
    assert "joint_a" not in snapshot["joint_positions"]
    assert "joint_a" not in snapshot["resource_facets"]["manipulator"]["joint_positions"]


@pytest.mark.parametrize("field,value", [
    ("preconditions", {None: {"equals": None}}),
    ("effects", {None: {"set": "idle"}}),
    ("params", {"value": True}),
    ("required_evidence", {"requires": [{"fact": "observed", "part_param": []}]}),
    ("required_evidence", {"establishes": [{"fact": []}]}),
])
def test_transition_derivation_malformed_contracts_report_context(field, value):
    inputs = _transition_derivation_case()
    inputs["recovery_des_model"]["primitive_catalog"][0][field] = value
    result = validate_recovery_transition(**inputs)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"


def test_transition_derivation_counts_rejected_parameter_combinations_in_bound():
    inputs = _transition_derivation_case()
    first = inputs["recovery_des_model"]["primitive_catalog"][0]
    first["params"] = {"value": {"type": "integer", "enum": ["invalid", "also_invalid", 3]}}
    first["required_params"] = ["value"]
    result = validate_recovery_transition(**inputs, max_expansions=2)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["primitive_support"]["search"]["parameter_trials"] == 2


def test_transition_derivation_terminal_evidence_preserves_exact_contract():
    inputs = _transition_derivation_case()
    result = validate_recovery_transition(**inputs)
    trace = result["primitive_support"]["trace"]
    assert trace[-1]["declared_effects"] == inputs["recovery_des_model"]["primitive_catalog"][-1]["effects"]
    assert trace[-1]["declaration_fingerprint"]


def _task_completion_derivation_case():
    inputs = _transition_derivation_case()
    inputs["recovery_des_model"]["primitive_catalog"][-1]["effects"].pop("part_state")
    event = {
        "event_name": "registered_processing",
        "guards": {"resource_state": {"equals": "faulted"}},
        "updates": {"resource_state": {"set": "idle"}, "part_state": {"set": "done"}},
        "primitive_support": [{"primitive": "prepare", "params": {}}, {"primitive": "finish", "params": {}}],
        "composition_contract": {
            "entry_state": "faulted", "part_in_state": "raw", "params": {}, "required_params": [],
            "entry_guards": [{"condition": {"field": "resource_state", "operator": "equals", "value": "faulted"}}],
            "required_context_keys": [], "part_transition": {"completed": {"state": "done"}},
            "steps": [{"op": "prepare", "executor": "primitive", "params": {}},
                      {"op": "finish", "executor": "primitive", "params": {}}],
        },
    }
    inputs["recovery_des_model"]["events"] = [event]
    return inputs, event


def test_transition_derivation_complete_task_contract_can_establish_part_effect():
    inputs, event = _task_completion_derivation_case()
    result = validate_recovery_transition(**inputs)
    assert result["allowed"]
    support = result["primitive_support"]
    assert support["part_name"] == "gear_small"
    assert [row["primitive"] for row in support["primitives"]] == ["prepare", "finish"]
    completion = support["trace"][-1]
    assert completion["primitive"] is None
    assert completion["task_contract"] == event["event_name"]
    assert completion["declared_effects"] == event["updates"]
    assert completion["primitive_start_index"] == 0
    assert completion["primitive_end_index"] == 2
    assert completion["start_state"]["part"]["part_state"] == "raw"
    assert completion["end_state"]["part"]["part_state"] == "done"


@pytest.mark.parametrize("change", ["missing_steps", "unsupported_step", "missing_primitive", "missing_context", "verification", "unknown_guard"])
def test_transition_derivation_task_contract_requires_complete_evidence(change):
    inputs, event = _task_completion_derivation_case()
    contract = event["composition_contract"]
    if change == "missing_steps":
        del contract["steps"]
    elif change == "unsupported_step":
        contract["steps"].append({"op": "observe", "executor": "runtime"})
    elif change == "missing_primitive":
        inputs["recovery_des_model"]["primitive_catalog"].pop(0)
    elif change == "missing_context":
        contract["required_context_keys"] = ["fixture"]
    elif change == "verification":
        contract["part_transition"]["completed"]["verify_camera"] = True
    else:
        contract["entry_guards"] = [{"predicate": "unknown_guard"}]
    result = validate_recovery_transition(**inputs)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert not result["derived_end_state"]


def test_transition_derivation_task_contract_does_not_bypass_declared_custody_guard():
    inputs, event = _task_completion_derivation_case()
    inputs["recovery_snapshot"]["held_part"] = "KET4_Square_4mm"
    event["guards"]["held_part"] = {"equals": None}
    result = validate_recovery_transition(**inputs)
    assert result["feasibility_status"] == "INFEASIBLE"


def test_transition_derivation_task_summary_cannot_invent_part_completion():
    inputs, event = _task_completion_derivation_case()
    event["composition_contract"]["part_transition"] = {}
    result = validate_recovery_transition(**inputs)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert "no declared task completion effect" in result["reason"]


def test_transition_derivation_task_step_cannot_substitute_another_part():
    inputs, event = _task_completion_derivation_case()
    primitive = inputs["recovery_des_model"]["primitive_catalog"][1]
    primitive["params"] = {"part_name": {"type": "string"}}
    primitive["required_params"] = ["part_name"]
    event["composition_contract"]["steps"][1]["params"] = {"part_name": "KET4_Square_4mm"}
    result = validate_recovery_transition(**inputs)
    assert result["feasibility_status"] == "INFEASIBLE"


def test_transition_derivation_task_contract_checks_step_bindings_and_intermediate_state():
    inputs, event = _task_completion_derivation_case()
    primitive = inputs["recovery_des_model"]["primitive_catalog"][1]
    primitive["params"] = {"part_name": {"type": "string"}}
    primitive["required_params"] = ["part_name"]
    contract = event["composition_contract"]
    contract["params"] = {"part_name": {"type": "string"}}
    contract["required_params"] = ["part_name"]
    contract["steps"][1]["params"] = {"part_name": {"$arg": "part_name"}}
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        if kwargs["primitive"] == "finish":
            assert kwargs["params"]["part_name"] == "gear_small"
            assert kwargs["recovery_snapshot"]["current_state"] == "prepared"
        return {"allowed": True, "feasibility_status": "FEASIBLE"}

    assert validate_recovery_transition(**inputs, validate_primitive=evaluate)["allowed"]
    assert any(row["primitive"] == "finish" for row in calls)


def test_transition_derivation_builder_captures_complete_native_task_program(monkeypatch):
    from cais_spade_llm.resources import resource_primitives
    from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry

    definition = robot_task_registry()["place_insert"]
    owner = SimpleNamespace(jid="M1@localhost", static_capabilities={}, place_insert=definition.handler)
    monkeypatch.setattr(resource_primitives, "_execution_catalog_for_snapshot", lambda _: [])
    descriptor = {"state_variables": {"resource_state": {"scope": "resource", "domain": ["positioned", "placed"]}},
                  "events": [{"event_name": "place_insert", "updates": {"resource_state": {"set": "placed"}}}]}
    model = resource_primitives.build_recovery_des_model(owner, descriptor=descriptor, snapshot={"current_state": "positioned"})
    contract = model["events"][0]["composition_contract"]
    assert [row["op"] for row in contract["steps"]] == [step.op for step in definition.program.steps]
    assert contract["part_transition"]["completed"]["verify_camera"] is True
    assert contract["required_params"] == definition.required_argument_names()


def test_transition_derivation_required_task_excludes_equivalent_shorter_primitive():
    inputs, event = _task_completion_derivation_case()
    inputs["recovery_des_model"]["primitive_catalog"].insert(0, {
        "name": "alternate", "params": {}, "required_params": [],
        "effects": {"current_state": {"set": "idle"}, "part_state": {"set": "done"}},
    })
    ordinary = validate_recovery_transition(**inputs)
    assert ordinary["primitive_support"]["primitives"][0]["primitive"] == "alternate"
    exact = validate_recovery_transition(**inputs, required_task_contract=event["event_name"])
    assert exact["allowed"]
    assert [row["primitive"] for row in exact["primitive_support"]["primitives"]] == ["prepare", "finish"]
    assert exact["primitive_support"]["trace"][-1]["primitive_start_index"] == 0


def test_transition_derivation_required_task_cannot_prepend_recovery():
    inputs, event = _task_completion_derivation_case()
    event["guards"] = {"resource_state": {"equals": "prepared"}}
    contract = event["composition_contract"]
    contract["entry_state"] = "prepared"
    contract["entry_guards"] = [{"condition": {"field": "resource_state", "operator": "equals", "value": "prepared"}}]
    contract["steps"].pop(0)
    event["primitive_support"].pop(0)
    assert validate_recovery_transition(**inputs)["allowed"]
    exact = validate_recovery_transition(**inputs, required_task_contract=event["event_name"])
    assert exact["feasibility_status"] == "INFEASIBLE"
    assert not exact["primitive_support"]["trace"]


@pytest.mark.parametrize("facts,status,called", [
    ({"value": True}, "FEASIBLE", True), ({"value": False}, "NOT_APPLICABLE", False),
    ({}, "NEEDS_CONTEXT", False), ({"value": None}, "NOT_APPLICABLE", False),
])
def test_union_optional_conditions_use_recorded_facts_without_primitive_effects(facts, status, called):
    from cais_spade_llm.resources.recovery_parameter_evidence import evaluate_resource_primitive_condition

    model = _model()
    model["events"] = model["events"][:1]
    model["events"][0]["primitive_support"][0]["when"] = [
        {"predicate": "always", "condition": {"field": "value", "operator": "equals", "value": True}}]
    snapshot = deepcopy(facts)
    owner = Mock(return_value={"allowed": True, "feasibility_status": "FEASIBLE"})
    condition = lambda **kwargs: evaluate_resource_primitive_condition(**kwargs, resource_evidence={}, resource_jid="xarm6@localhost")
    result = validate_primitive_support(task=_task(), recovery_des_model=model, recovery_snapshot=snapshot,
        validate_primitive=owner, evaluate_condition=condition)
    row = result["primitive_support"]["primitives"][0]
    assert row["feasibility_status"] == status
    assert owner.called is called
    assert row["allowed"] is (status == "FEASIBLE")
    assert snapshot == facts
    assert len(row["contributors"]) == 1
    assert "trace" not in result["primitive_support"]


@pytest.mark.parametrize("second_condition,overall,calls", [
    ({"field": "available", "operator": "equals", "value": True}, "FEASIBLE", 1),
    ({"field": "missing", "operator": "equals", "value": True}, "NEEDS_CONTEXT", 0),
    ({"field": "available", "operator": "unsupported", "value": True}, "NEEDS_CONTEXT", 0),
])
def test_union_multiple_contributors_retain_false_conditions_and_require_applicable_witness(second_condition, overall, calls):
    from cais_spade_llm.resources.recovery_parameter_evidence import evaluate_resource_primitive_condition

    model = _model()
    model["events"] = model["events"][:1]
    support = model["events"][0]["primitive_support"]
    support[0]["when"] = [{"predicate": "always", "condition": {"field": "available", "operator": "equals", "value": False}}]
    support.append({"primitive": "move_to_named_pose", "params": {"pose_name": "home"},
                    "when": [{"predicate": "always", "condition": second_condition}]})
    owner = Mock(return_value={"allowed": True, "feasibility_status": "FEASIBLE"})
    condition = lambda **kwargs: evaluate_resource_primitive_condition(**kwargs, resource_evidence={}, resource_jid="xarm6@localhost")
    result = validate_primitive_support(task=_task(), recovery_des_model=model, recovery_snapshot={"available": True},
        validate_primitive=owner, evaluate_condition=condition)
    assert result["feasibility_status"] == overall
    assert owner.call_count == calls
    row = result["primitive_support"]["primitives"][0]
    assert len(row["contributors"]) == len(row["condition_checks"]) == 2
    assert row["condition_checks"][0]["status"] == "not_applicable"
    assert row["parameter_attempts"][0]["feasibility_status"] == "NOT_APPLICABLE"
