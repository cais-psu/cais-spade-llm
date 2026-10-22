from __future__ import annotations

"""Recovery task synthesis feasibility and evidence-handoff regression tests."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union


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


def test_owner_snapshot_supplies_missing_starting_facts_without_rewriting_proposal(machine_successors):
    task, model = machine_successors
    snapshot = task.pop("expected_start_state")
    before = deepcopy((task, snapshot))
    support = derive_primitive_union(task=task, recovery_des_model=model, recovery_snapshot=snapshot)
    assert support["requested_changes"] == ["job_state"]
    assert [row["primitive"] for row in support["primitives"]] == ["pause_job"]
    assert (task, snapshot) == before


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


def test_union_replays_saved_buffer_proposal_without_chaining_or_source_changes():
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
    assert capture_path.read_bytes() == source
    assert saved == json.loads(source)


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
    assert [(row["primitive"], row["params"]) for row in prepared["parameter_witnesses"]] == [("heat", {"target": 50})]
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
    assert prepared["context_requests"] == []
    assert all("speed" not in row["params"] for row in prepared["parameter_witnesses"])


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
