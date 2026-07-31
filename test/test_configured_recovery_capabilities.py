"""Focused tests for resource-owned runtime fact grounding."""

from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cais_spade_llm.agents.central_controller.central_controller_agent import (
    CentralControllerAgent,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
    multi_turn,
)
from cais_spade_llm.agents.resource_agent.printing_agent import PrintingAgent
from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
from cais_spade_llm.resources.capability_engine import (
    compound_recovery_state,
    configured_capability_errors,
    configured_event_successor,
)
from cais_spade_llm.resources.robot.robot_tasks import robot_task_registry

_EXPECTED_INITIAL_EVENTS = [
    "pick_approach",
    "pick_grasp",
    "place_approach",
    "move_home",
    "place_insert",
    "place_release",
]


async def _configured_action(**_facts: object) -> dict[str, bool]:
    return {"success": True}


def _configured_capabilities(
    robot_name: str = "xarm6",
    environment: str = "gazebo",
) -> dict:
    path = Path(
        f"cais_spade_llm/initialization/resources/robot_{robot_name}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    static_capabilities = deepcopy(
        payload[robot_name][environment]["static_capabilities"]
    )
    return static_capabilities


class _RobotCapabilityHarness:
    _executable_required_runtime_facts = staticmethod(
        ResourceAgent._executable_required_runtime_facts
    )
    _executable_runtime_fact_names = staticmethod(
        ResourceAgent._executable_runtime_fact_names
    )
    _configured_event_required_runtime_facts = (
        ResourceAgent._configured_event_required_runtime_facts
    )
    _capability_execution_arguments = (
        ResourceAgent._capability_execution_arguments
    )
    _capability_atomic_transition = (
        ResourceAgent._capability_atomic_transition
    )
    _bind_capabilities = ResourceAgent._bind_capabilities
    _evaluate_capability_transition = (
        ResourceAgent._evaluate_capability_transition
    )
    _enabled_capability_instances = (
        ResourceAgent._enabled_capability_instances
    )
    _future_goal_capability_instances = (
        ResourceAgent._future_goal_capability_instances
    )
    public_capability_state_projections = (
        ResourceAgent.public_capability_state_projections
    )
    public_state_values = ResourceAgent.public_state_values
    public_locations = RobotAgent.public_locations
    public_action_target = ResourceAgent.public_action_target
    public_resource_state_context = (
        RobotAgent.public_resource_state_context
    )
    public_part_state_context = RobotAgent.public_part_state_context
    capability_runtime_facts = RobotAgent.capability_runtime_facts
    capability_runtime_fact_tables = (
        RobotAgent.capability_runtime_fact_tables
    )
    capability_state_valuation = RobotAgent.capability_state_valuation
    capability_goal_requirements = (
        RobotAgent.capability_goal_requirements
    )
    _resolve_generated_capability = RobotAgent._resolve_generated_capability
    _generated_successor_from_action_target = (
        RobotAgent._generated_successor_from_action_target
    )
    _generated_successor_consistency_finding = (
        RobotAgent._generated_successor_consistency_finding
    )
    _is_pose_in_workspace = RobotAgent._is_pose_in_workspace
    check_recovery_physical_feasibility = (
        RobotAgent.check_recovery_physical_feasibility
    )
    configured_execution_successor = RobotAgent.configured_execution_successor

    @classmethod
    def capability_runtime_fact_names(cls) -> set[str]:
        return RobotAgent.capability_runtime_fact_names()

    def __init__(
        self,
        *,
        capabilities: dict | None = None,
        jid: str = "xarm6@localhost",
    ) -> None:
        self.jid = jid
        self.static_capabilities = deepcopy(
            capabilities or _configured_capabilities()
        )
        self._configured_capability_declarations = deepcopy(
            self.static_capabilities
        )
        self.executables = {
            str(event.get("event_name") or ""): _configured_action
            for event in self.static_capabilities.get("events") or []
        }


def _idle_snapshot() -> dict:
    return {
        "resource_jid": "xarm6@localhost",
        "current_state": "idle",
        "current_location": None,
        "current_pose_ref": None,
        "held_part": None,
        "gripper_state": "open",
        "carried_entity_location": "xarm6@localhost_gripper",
    }


def _part_context(part_name: str, location: str) -> dict:
    return {
        "part_name": part_name,
        "current_state": "ready",
        "current_location": location,
    }


def _generated_pick_task(part_name: str, location: str) -> dict:
    return {
        "event_name": f"secure_{part_name.lower()}",
        "resource_jid": "xarm6@localhost",
        "part_name": part_name,
        "expected_start_state": compound_recovery_state({
            "resource_state": "idle",
            "resource_location": None,
            "held_part": None,
            "part_state": "ready",
            "part_location": location,
        }),
        "expected_end_state": compound_recovery_state({
            "resource_state": "picked",
            "resource_location": location,
            "held_part": part_name,
            "part_state": "in_gripper",
            "part_location": "xarm6@localhost_gripper",
        }),
    }


def test_public_state_values_publish_each_location_scope() -> None:
    """A part's location vocabulary is its own, not the resource workspace."""
    harness = _RobotCapabilityHarness()
    published = harness.public_state_values(resource_snapshot=_idle_snapshot())
    state_values = published["state_values"]

    resource_locations = state_values["resource_state"]["location"]
    part_locations = state_values["part_state"]["location"]

    assert "home" in resource_locations
    assert "observed_pose" not in resource_locations

    # A resource-only pose is not somewhere a part can rest.
    assert "home" not in part_locations
    assert "observed_pose" in part_locations
    # carried_part_location is declared as a parameter value, so the holder
    # token becomes authorable without being a manifest domain member.
    assert "xarm6@localhost_gripper" in part_locations


def test_robot_manifests_have_direct_capabilities_without_parameter_bindings() -> None:
    for robot_name in ("xarm6", "ur5e"):
        for environment in ("gazebo", "real"):
            capabilities = _configured_capabilities(robot_name, environment)
            assert [
                event["event_name"] for event in capabilities["events"]
            ] == _EXPECTED_INITIAL_EVENTS
            assert all(
                "parameter_bindings" not in event
                for event in capabilities["events"]
            )
            resource_location = capabilities["state_variables"][
                "resource_location"
            ]
            part_location = capabilities["state_variables"][
                "part_location"
            ]
            assert "observed_pose" not in resource_location["domain"]
            assert "parameter_values" not in resource_location
            assert "observed_pose" in part_location["domain"]
            assert configured_capability_errors(
                capabilities,
                executable_names=set(_EXPECTED_INITIAL_EVENTS),
                runtime_fact_names=RobotAgent.capability_runtime_fact_names(),
                executable_required_facts={
                    event_name: set()
                    for event_name in _EXPECTED_INITIAL_EVENTS
                },
            ) == []


def test_configuration_validation_fails_closed() -> None:
    model = {
        "state_variables": {
            "resource_state": {
                "scope": "resource",
                "domain": ["idle", "done"],
                "parameter_values": ["unsupported_fact"],
            }
        },
        "events": [
            {
                "event_name": "finish",
                "guards": {
                    "resource_state": {"approximately": "idle"}
                },
                "updates": {
                    "resource_state": {
                        "set_from_param": "unsupported_fact"
                    }
                },
                "parameter_bindings": {},
            }
        ],
    }
    errors = configured_capability_errors(
        model,
        executable_names={"different_function"},
        runtime_fact_names={"resource_jid"},
    )
    assert any("has no executable function" in error for error in errors)
    assert any("must not declare parameter_bindings" in error for error in errors)
    assert any("unsupported operator 'approximately'" in error for error in errors)
    assert any("unsupported runtime fact 'unsupported_fact'" in error for error in errors)
    assert configured_capability_errors({}) == [
        "static_capabilities.state_variables must be a nonempty object"
    ]

    class UnsupportedFactResource(ResourceAgent):
        async def finish(self) -> dict[str, bool]:
            return {"success": True}

    startup_capabilities = {
        "state_variables": {
            "resource_state": {
                "scope": "resource",
                "domain": ["idle"],
                "parameter_values": ["program_ref"],
            }
        },
        "events": [
            {
                "event_name": "finish",
                "guards": {},
                "updates": {
                    "resource_state": {
                        "set_from_param": "program_ref"
                    }
                },
            }
        ],
    }
    with pytest.raises(
        ValueError,
        match="unsupported runtime fact 'program_ref'",
    ):
        UnsupportedFactResource(
            "resource@localhost",
            "password",
            name="resource",
            function_names=["finish"],
            static_capabilities=startup_capabilities,
        )


def test_printing_agent_fails_closed_without_configured_capabilities() -> None:
    with pytest.raises(ValueError, match="state_variables"):
        PrintingAgent(
            "printer@localhost",
            "password",
            name="printer",
            function_names=["pause_job", "resume_job", "cancel_job"],
            static_capabilities={},
        )


def test_exact_name_substitution_grounds_lg_and_mg_differently() -> None:
    harness = _RobotCapabilityHarness()
    pick_grasp = next(
        event
        for event in harness.static_capabilities["events"]
        if event["event_name"] == "pick_grasp"
    )
    initial = {
        "resource_state": "at_pick",
        "resource_location": "prusa-mk4-1",
        "held_part": None,
        "gripper_state": "open",
        "part_state": "ready",
        "part_location": "prusa-mk4-1",
    }
    lg = configured_event_successor(
        pick_grasp,
        initial,
        state_variables=harness.static_capabilities["state_variables"],
        runtime_facts={
            "part_name": "LG",
            "carried_part_location": "xarm6@localhost_gripper",
        },
    )
    mg = configured_event_successor(
        pick_grasp,
        initial,
        state_variables=harness.static_capabilities["state_variables"],
        runtime_facts={
            "part_name": "MG",
            "carried_part_location": "xarm6@localhost_gripper",
        },
    )
    assert lg["successor"]["held_part"] == "LG"
    assert mg["successor"]["held_part"] == "MG"
    assert lg["successor"]["part_location"] == "xarm6@localhost_gripper"


def test_calculated_successor_must_fit_the_configured_field_domain() -> None:
    state_variables = {
        "job_state": {
            "scope": "resource",
            "domain": ["idle", "done"],
        }
    }
    result = configured_event_successor(
        {
            "event_name": "load_program",
            "guards": {"job_state": {"equals": "idle"}},
            "updates": {
                "job_state": {"set_from_param": "program_ref"}
            },
        },
        {"job_state": "idle"},
        state_variables=state_variables,
        runtime_facts={"program_ref": "program-42"},
    )

    assert result["enabled"] is False
    assert result["constraint_code"] == "state_value_outside_ra_domain"
    assert result["evidence"] == {
        "field": "job_state",
        "value": "program-42",
    }


def test_runtime_facts_are_rebuilt_from_each_request_context() -> None:
    harness = _RobotCapabilityHarness()
    first = harness.capability_runtime_facts(
        task={
            "part_name": "LG",
            "action_target": {
                "source_location": "candidate_source",
                "target_location": "candidate_target",
            },
            "expected_end_state": {"part_location": "home"},
        },
        resource_snapshot={
            **_idle_snapshot(),
            "current_pose_ref": "home",
            "carried_entity_location": "first_gripper",
        },
        part_context={
            "part_name": "LG",
            "current_location": "prusa-mk4-1",
            "goal_location": "assembly_board-v1",
        },
        grounded_action={},
    )
    second = harness.capability_runtime_facts(
        task={
            "part_name": "MG",
            "destination_location": "inspection",
        },
        resource_snapshot={
            **_idle_snapshot(),
            "current_pose_ref": "service",
            "carried_entity_location": "second_gripper",
        },
        part_context={
            "part_name": "MG",
            "current_location": "prusa-mk3",
            "goal_location": "inspection",
        },
        grounded_action={},
    )
    assert first == {
        "resource_jid": "xarm6@localhost",
        "part_name": "LG",
        "origin_resource_location": "prusa-mk4-1",
        "destination_location": "assembly_board-v1",
        "part_goal_location": "assembly_board-v1",
        "carried_part_location": "first_gripper",
        "current_pose_ref": "home",
    }
    assert second["part_name"] == "MG"
    assert second["origin_resource_location"] == "prusa-mk3"
    assert second["destination_location"] == "inspection"
    assert second["part_goal_location"] == "inspection"
    assert second["carried_part_location"] == "second_gripper"
    assert second["current_pose_ref"] == "service"
    assert "candidate_source" not in first.values()
    assert "candidate_target" not in first.values()
    assert first["destination_location"] != "home"


def test_sixth_event_is_discovered_without_engine_changes() -> None:
    capabilities = _configured_capabilities()
    capabilities["events"].append(
        {
            "event_name": "inspect_part",
            "guards": {"resource_state": {"equals": "idle"}},
            "updates": {"resource_state": {"set": "failed"}},
        }
    )
    harness = _RobotCapabilityHarness(capabilities=capabilities)
    transition = harness._evaluate_capability_transition(
        task={
            "event_name": "inspect_part",
            "expected_start_state": {"resource_state": "idle"},
            "expected_end_state": {"resource_state": "failed"},
        },
        resource_snapshot=_idle_snapshot(),
    )
    assert transition["allowed"] is True
    assert transition["calculated_successor"]["resource_state"]["condition"] == (
        "failed"
    )
    assert transition["_atomic_transitions"][0]["event_name"] == "inspect_part"


def test_robot_task_program_keeps_no_configured_transition_effects() -> None:
    transition_targets = {
        "current_state",
        "held_part",
        "gripper_state",
        "recovery_pose_ref",
    }
    for task_name in _EXPECTED_INITIAL_EVENTS:
        program = robot_task_registry()[task_name].program
        assert not {
            effect.target for effect in program.effects
        }.intersection(transition_targets)

    harness = _RobotCapabilityHarness()
    successor = harness.configured_execution_successor(
        event_name="pick_approach",
        execution_arguments={
            "part_name": "LG",
            "origin_resource_location": "prusa-mk4-1",
        },
        runtime_state={
            "_current_state": "idle",
            "_recovery_pose_ref": None,
            "_held_part": None,
            "_gripper_state": "open",
        },
    )
    assert successor is not None
    assert successor["resource_state"] == "at_pick"
    assert successor["resource_location"] == "prusa-mk4-1"


def test_missing_runtime_fact_disables_only_affected_capability() -> None:
    harness = _RobotCapabilityHarness()
    transition = harness._evaluate_capability_transition(
        task={
            "event_name": "pick_approach",
            "resource_jid": "xarm6@localhost",
            "part_name": "LG",
            "expected_start_state": {
                "resource_state": "idle",
                "held_part": None,
            },
            "expected_end_state": {"resource_state": "at_pick"},
        },
        resource_snapshot=_idle_snapshot(),
        part_context={"part_name": "LG"},
    )
    assert transition["allowed"] is False
    assert transition["constraint_code"] == "runtime_fact_unavailable"
    assert "missing_runtime_facts" not in transition["evidence"]


def test_exact_event_and_generated_event_use_private_successors() -> None:
    harness = _RobotCapabilityHarness()
    context = _part_context("LG", "prusa-mk4-1")
    exact = harness._evaluate_capability_transition(
        task={
            "event_name": "pick_approach",
            "resource_jid": "xarm6@localhost",
            "part_name": "LG",
            "origin_resource_location": "prusa-mk4-1",
            "expected_end_state": {"resource_state": "at_pick"},
        },
        resource_snapshot=_idle_snapshot(),
        part_context=context,
    )
    generated = harness._evaluate_capability_transition(
        task=_generated_pick_task("LG", "prusa-mk4-1"),
        resource_snapshot=_idle_snapshot(),
        part_context=context,
    )
    assert exact["allowed"] is True
    assert exact["calculated_successor"]["resource_state"]["location"] == (
        "prusa-mk4-1"
    )
    assert generated["allowed"] is True
    assert [
        row["event_name"] for row in generated["_atomic_transitions"]
    ] == ["secure_lg"]
    assert generated["calculated_successor"]["held_part"] == "LG"


def test_disabled_exact_event_name_remains_strict() -> None:
    harness = _RobotCapabilityHarness()
    gripper_location = "xarm6@localhost_gripper"
    transition = harness._evaluate_capability_transition(
        task={
            "event_name": "place_release",
            "resource_jid": "xarm6@localhost",
            "part_name": "MCP",
            "expected_end_state": {
                "resource_state": "placed",
                "resource_location": "prusa-mk3",
                "held_part": None,
                "part_state": "ready",
                "part_location": "prusa-mk3",
            },
        },
        resource_snapshot={
            **_idle_snapshot(),
            "current_state": "picked",
            "current_location": "prusa-mk4-1",
            "current_pose_ref": "prusa-mk4-1",
            "held_part": "MCP",
            "gripper_state": "closed",
            "carried_entity_location": gripper_location,
        },
        part_context={
            "part_name": "MCP",
            "current_state": "in_gripper",
            "current_location": gripper_location,
            "goal_location": "assembly_board-v1",
        },
    )

    assert transition["allowed"] is False
    assert transition["constraint_code"] in {
        "runtime_fact_unavailable",
        "unsatisfied_guard_predicate",
    }
    assert transition["_atomic_transitions"] == []


def test_future_goal_query_discovers_guard_disabled_atomic_event() -> None:
    capabilities = _configured_capabilities("ur5e", "gazebo")
    harness = _RobotCapabilityHarness(
        jid="ur5e@localhost",
        capabilities=capabilities,
    )
    snapshot = {
        "resource_jid": "ur5e@localhost",
        "current_state": "idle",
        "current_location": "home",
        "current_pose_ref": "home",
        "held_part": None,
        "gripper_state": "open",
        "carried_entity_location": "ur5e@localhost_gripper",
    }
    part_contexts = [
        {
            "part_name": "LG",
            "current_state": "misplaced",
            "current_location": "observed_pose",
            "observed_pose": {"x": 0.0, "y": 0.2, "z": 1.035},
            "goal_location": "assembly_board-v1",
        }
    ]
    goal_conditions = [
        {
            "condition_id": "goal_lg_state",
            "entity_kind": "part",
            "entity": "LG",
            "field": "state",
            "expected": "assembled",
        },
        {
            "condition_id": "goal_lg_location",
            "entity_kind": "part",
            "entity": "LG",
            "field": "location",
            "expected": "assembly_board-v1",
        },
    ]

    future = harness._future_goal_capability_instances(
        resource_snapshot=snapshot,
        part_contexts=part_contexts,
        goal_conditions=goal_conditions,
    )
    enabled = harness._enabled_capability_instances(
        resource_snapshot=snapshot,
        part_contexts=part_contexts,
    )

    assert [row["task"]["event_name"] for row in future] == [
        "place_insert"
    ]
    assert future[0]["task"]["expected_end_state"]["part_state"]["condition"] == (
        "assembled"
    )
    assert future[0]["task"]["expected_end_state"]["part_state"]["location"] == (
        "assembly_board-v1"
    )
    assert future[0]["resource_enabled"] is False
    assert "place_insert" not in {
        row["task"]["event_name"] for row in enabled
    }

    enabled_future = harness._future_goal_capability_instances(
        resource_snapshot={
            **snapshot,
            "current_state": "positioned",
            "current_location": "assembly_board-v1",
            "current_pose_ref": "assembly_board-v1",
            "held_part": "LG",
            "gripper_state": "closed",
        },
        part_contexts=[
            {
                **part_contexts[0],
                "current_state": "in_transit",
                "current_location": "ur5e@localhost_gripper",
            }
        ],
        goal_conditions=goal_conditions,
    )
    assert enabled_future[0]["resource_enabled"] is True


def test_generated_events_require_declared_values_and_fields() -> None:
    harness = _RobotCapabilityHarness()
    context = _part_context("LG", "prusa-mk4-1")
    undeclared = _generated_pick_task("LG", "prusa-mk4-1")
    undeclared["expected_end_state"]["resource_state"]["condition"] = (
        "lg_secured"
    )
    result = harness._evaluate_capability_transition(
        task=undeclared,
        resource_snapshot=_idle_snapshot(),
        part_context=context,
    )
    assert result["allowed"] is False
    assert result["constraint_code"] == "state_value_outside_ra_domain"

    unsupported = _generated_pick_task("LG", "prusa-mk4-1")
    unsupported["expected_end_state"]["unsupported_field"] = "value"
    result = harness._evaluate_capability_transition(
        task=unsupported,
        resource_snapshot=_idle_snapshot(),
        part_context=context,
    )
    assert result["constraint_code"] == "disallowed_outline_state_field"


def test_cartesian_action_target_produces_typed_pose_location() -> None:
    harness = _RobotCapabilityHarness()
    generated_task = {
        "event_name": "retreat_from_assembly_board",
        "resource_jid": "xarm6@localhost",
        "action_target": {"x": 0.1, "y": 0.0, "z": 1.1},
        "expected_end_state": {
            "resource_state": {
                "condition": "idle",
            },
            "held_part": None,
        },
    }

    grounded = harness._evaluate_capability_transition(
        task=generated_task,
        resource_snapshot=_idle_snapshot(),
        grounded_action={},
    )
    assert grounded["allowed"] is True
    assert grounded["calculated_successor"]["resource_state"] == {
        "condition": "idle",
        "location": {
            "frame": "world",
            "units": "m",
            "x": 0.1,
            "y": 0.0,
            "z": 1.1,
        },
    }

    invented_location = deepcopy(generated_task)
    invented_location["action_target"] = {"location": "lg_pick_pose"}
    unavailable = harness._evaluate_capability_transition(
        task=invented_location,
        resource_snapshot=_idle_snapshot(),
        grounded_action={},
    )
    assert unavailable["constraint_code"] == "unknown_location_token"


def test_generated_location_pose_is_checked_by_physical_feasibility() -> None:
    harness = _RobotCapabilityHarness()
    result = harness.check_recovery_physical_feasibility(
        part_context={},
        recovery_snapshot=_idle_snapshot(),
        grounded_action={
            "resource_jid": "xarm6@localhost",
            "task_kind": "resource_action",
            "target": {
                "target_location": "outside_pose",
                "pose": {"x": 0.0, "y": 0.5, "z": 1.1},
            },
            "expected_effect": {
                "resource": {
                    "current_state": "stabilized_for_transfer",
                    "location": "outside_pose",
                },
                "part": {},
            },
            "effect_scope": "resource_only",
        },
    )
    assert result["allowed"] is False
    assert result["constraint_code"] == "workspace_unreachable"


def test_generated_part_condition_must_be_declared() -> None:
    harness = _RobotCapabilityHarness()
    task = _generated_pick_task("PART_A", "prusa-mk4-1")
    task["expected_end_state"]["part_state"]["condition"] = (
        "secured_for_transfer"
    )
    part_context = _part_context("PART_A", "prusa-mk4-1")

    allowed = harness._evaluate_capability_transition(
        task=task,
        resource_snapshot=_idle_snapshot(),
        part_context=part_context,
    )
    assert allowed["allowed"] is False
    assert allowed["constraint_code"] == "state_value_outside_ra_domain"

    mismatched = deepcopy(task)
    mismatched["expected_end_state"]["part_state"]["condition"] = "in_gripper"
    mismatched["expected_end_state"]["part_state"]["location"] = (
        "prusa-mk4-1"
    )
    rejected = harness._evaluate_capability_transition(
        task=mismatched,
        resource_snapshot=_idle_snapshot(),
        part_context=part_context,
    )
    assert rejected["constraint_code"] == "conflicting_location_grounding"


def _held_part_action_target_task(part_location: Any) -> dict:
    return {
        "event_name": "reposition_with_held_part",
        "resource_jid": "xarm6@localhost",
        "part_name": "PART_A",
        "action_target": {"x": 0.4, "y": 0.3, "z": 1.2},
        "expected_end_state": compound_recovery_state({
            "resource_state": "picked",
            "resource_location": None,
            "held_part": "PART_A",
            "part_state": "in_gripper",
            "part_location": part_location,
        }),
    }


def _ground_held_part_action_target(task: dict) -> tuple[dict, dict | None]:
    harness = _RobotCapabilityHarness()
    return harness._generated_successor_from_action_target(
        task=task,
        initial_valuation={"held_part": "PART_A"},
        successor={"held_part": "PART_A"},
        runtime_fact_tables=[
            {"carried_part_location": "xarm6@localhost_gripper"}
        ],
    )


def test_generated_action_target_treats_null_location_as_unspecified() -> None:
    completed, conflict = _ground_held_part_action_target(
        _held_part_action_target_task(None)
    )
    assert conflict is None
    assert completed["resource_location"] == {
        "frame": "world",
        "units": "m",
        "x": 0.4,
        "y": 0.3,
        "z": 1.2,
    }
    assert completed["part_location"] == "xarm6@localhost_gripper"


def test_generated_action_target_accepts_held_part_gripper_location() -> None:
    completed, conflict = _ground_held_part_action_target(
        _held_part_action_target_task("xarm6@localhost_gripper")
    )
    assert conflict is None
    assert completed["part_location"] == "xarm6@localhost_gripper"


def test_generated_action_target_still_rejects_wrong_held_part_location() -> None:
    _completed, conflict = _ground_held_part_action_target(
        _held_part_action_target_task("prusa-mk4-1")
    )
    assert conflict is not None
    assert conflict["constraint_code"] == "conflicting_location_grounding"


def test_enabledness_uses_manifest_order_and_exact_part_name_order() -> None:
    harness = _RobotCapabilityHarness()
    enabled = harness._enabled_capability_instances(
        resource_snapshot=_idle_snapshot(),
        part_contexts=[
            _part_context("MG", "prusa-mk4-1"),
            _part_context("LG", "prusa-mk4-1"),
        ],
    )
    assert [
        (row["task"]["event_name"], row["task"].get("part_name"))
        for row in enabled
    ] == [
        ("pick_approach", "LG"),
        ("pick_approach", "MG"),
        ("move_home", None),
    ]


def test_public_state_projections_are_dynamic_and_hide_private_fields() -> None:
    harness = _RobotCapabilityHarness()
    projections = harness.public_capability_state_projections(
        resource_snapshot=_idle_snapshot(),
        part_contexts=[
            _part_context("MCP", "prusa-mk4-1"),
            _part_context("LG", "prusa-mk4-1"),
        ],
    )

    assert [
        (
            row["resource_jid"],
            row.get("part_name"),
        )
        for row in projections
    ] == [
        ("xarm6@localhost", None),
        ("xarm6@localhost", "LG"),
        ("xarm6@localhost", "MCP"),
    ]
    assert projections[0]["state"] == {
        "resource_state": {
            "condition": "idle",
            "location": None,
        },
        "held_part": None,
    }
    assert projections[1]["state"] == {
        "part_state": {
            "condition": "ready",
            "location": "prusa-mk4-1",
        },
    }
    serialized = json.dumps(projections, sort_keys=True)
    for private_name in (
        "gripper_state",
        "domain",
        "scope",
        "guards",
        "updates",
        "events",
        "runtime_facts",
    ):
        assert private_name not in serialized


def test_robot_public_state_context_exposes_bounds_and_observed_xyz() -> None:
    harness = _RobotCapabilityHarness()
    projections = harness.public_capability_state_projections(
        resource_snapshot=_idle_snapshot(),
        part_contexts=[
            {
                "part_name": "PART_A",
                "current_state": "misplaced",
                "current_location": None,
                "current_holder_resource_jid": None,
                "observed_pose": {
                    "x": 0.0,
                    "y": 0.2,
                    "z": 1.035,
                },
            }
        ],
    )

    assert projections[0]["workspace_bounds"] == {
        "x_min_m": -0.6,
        "x_max_m": 0.6,
        "y_min_m": -1.0,
        "y_max_m": 0.1,
        "z_min_m": 0.9,
        "z_max_m": 1.5,
    }
    assert projections[1]["state"]["part_state"]["location"] == (
        "observed_pose"
    )
    assert projections[1]["observed_pose"] == {
        "x": 0.0,
        "y": 0.2,
        "z": 1.035,
    }
    assert "workspace_reachability" not in projections[1]


def test_robot_public_state_context_fails_closed_without_workspace_bounds() -> None:
    capabilities = _configured_capabilities()
    capabilities.pop("workspace_bounds")
    harness = _RobotCapabilityHarness(capabilities=capabilities)

    with pytest.raises(
        ValueError,
        match="workspace_bounds capability data is unavailable",
    ):
        harness.public_capability_state_projections(
            resource_snapshot=_idle_snapshot(),
            part_contexts=[],
        )


def test_public_state_projection_fails_closed_without_public_fields() -> None:
    capabilities = _configured_capabilities()
    for declaration in capabilities["state_variables"].values():
        declaration["private"] = True
    harness = _RobotCapabilityHarness(capabilities=capabilities)

    with pytest.raises(
        ValueError,
        match="configured state_variables has no public fields",
    ):
        harness.public_capability_state_projections(
            resource_snapshot=_idle_snapshot(),
            part_contexts=[_part_context("LG", "prusa-mk4-1")],
        )


def test_public_state_values_use_templates_and_field_values() -> None:
    harness = _RobotCapabilityHarness()
    part_contexts = [
        {
            "part_name": "LG",
            "current_state": "misplaced",
            "current_location": None,
            "observed_pose": {
                "x": 0.0,
                "y": 0.2,
                "z": 1.035,
            },
            "goal_location": "assembly_board-v1",
        },
        {
            "part_name": "MCP",
            "current_state": "in_gripper",
            "current_location": "xarm6@localhost_gripper",
            "goal_location": "assembly_board-v1",
        },
        {
            "part_name": "P3",
            "current_state": "ready",
            "current_location": "prusa-mk4-1",
            "goal_location": "assembly_board-v1",
        },
    ]
    snapshot = {
        **_idle_snapshot(),
        "current_state": "picked",
        "held_part": "MCP",
        "gripper_state": "closed",
    }
    state_values = harness.public_state_values(
        resource_snapshot=snapshot,
        part_contexts=part_contexts,
    )

    assert state_values["resource_jid"] == "xarm6@localhost"
    assert state_values["state_values"]["held_part"] == {
        "values": [None],
        "template": "part_name",
    }
    assert set(state_values["state_values"]) == {
        "resource_state",
        "held_part",
        "part_state",
    }
    assert set(state_values["state_values"]["resource_state"]) == {
        "condition",
        "location",
    }
    assert set(state_values["state_values"]["part_state"]) == {
        "condition",
        "location",
    }
    serialized = json.dumps(state_values, sort_keys=True)
    for private_name in (
        "gripper_state",
        "events",
        "guards",
        "updates",
        "runtime_facts",
        "pick_grasp",
    ):
        assert private_name not in serialized


def test_public_state_values_fail_closed_without_public_fields() -> None:
    capabilities = _configured_capabilities()
    for declaration in capabilities["state_variables"].values():
        declaration["private"] = True
    harness = _RobotCapabilityHarness(capabilities=capabilities)

    with pytest.raises(
        ValueError,
        match="configured state_variables has no public fields",
    ):
        harness.public_state_values(
            resource_snapshot=_idle_snapshot(),
            part_contexts=[_part_context("LG", "prusa-mk4-1")],
        )


def test_public_state_values_do_not_depend_on_duplicate_part_bindings() -> None:
    harness = _RobotCapabilityHarness()

    duplicated = harness.public_state_values(
        resource_snapshot=_idle_snapshot(),
        part_contexts=[
            _part_context("LG", "prusa-mk4-1"),
            _part_context("LG", "prusa-mk4-1"),
        ],
    )
    empty = harness.public_state_values(
        resource_snapshot=_idle_snapshot(),
        part_contexts=[],
    )
    assert duplicated == empty


def test_held_part_values_follow_the_request_part_inventory() -> None:
    harness = _RobotCapabilityHarness()
    part_names = ["PART_A", "PART_B", "PART_C", "PART_D"]

    state_values = harness.public_state_values(
        resource_snapshot=_idle_snapshot(),
        part_contexts=[
            _part_context(part_name, "prusa-mk4-1")
            for part_name in part_names
        ],
    )

    assert state_values["state_values"]["held_part"] == {
        "values": [None],
        "template": "part_name",
    }
    serialized = json.dumps(
        state_values["state_values"]["held_part"],
        sort_keys=True,
    )
    for part_name in [*part_names, "LG", "MCP"]:
        assert part_name not in serialized


def test_other_resource_gripper_is_not_published_as_an_origin() -> None:
    harness = _RobotCapabilityHarness()
    public_locations = harness.public_locations(
        resource_snapshot=_idle_snapshot(),
        part_contexts=[
            {
                "part_name": "PART_A",
                "current_state": "in_gripper",
                "current_location": "ur5e@localhost_gripper",
                "current_holder_resource_jid": "ur5e@localhost",
                "goal_location": "assembly_board-v1",
            }
        ],
    )

    serialized = json.dumps(public_locations, sort_keys=True)
    assert "ur5e@localhost_gripper" not in serialized
    assert "xarm6@localhost_gripper" not in serialized


def test_generated_complete_valuation_stages_part_and_moves_home() -> None:
    harness = _RobotCapabilityHarness()
    gripper_location = "xarm6@localhost_gripper"
    transition = harness._evaluate_capability_transition(
        task={
            "event_name": "stage_mcp_and_free_xarm6",
            "resource_jid": "xarm6@localhost",
            "part_name": "MCP",
            "expected_end_state": {
                "resource_state": "idle",
                "resource_location": "home",
                "held_part": None,
                "part_state": "ready",
                "part_location": "prusa-mk3",
            },
        },
        resource_snapshot={
            **_idle_snapshot(),
            "current_state": "picked",
            "current_location": "prusa-mk4-1",
            "current_pose_ref": "prusa-mk4-1",
            "held_part": "MCP",
            "gripper_state": "closed",
            "carried_entity_location": gripper_location,
        },
        part_context={
            "part_name": "MCP",
            "current_state": "in_gripper",
            "current_location": gripper_location,
            "goal_location": "assembly_board-v1",
        },
    )

    assert transition["allowed"] is True
    assert transition["calculated_successor"] == {
        "resource_state": {
            "condition": "idle",
            "location": "home",
        },
        "held_part": None,
        "part_state": {
            "condition": "ready",
            "location": "prusa-mk3",
        },
    }
    assert [
        row["event_name"] for row in transition["_atomic_transitions"]
    ] == ["stage_mcp_and_free_xarm6"]


def test_generated_assembled_at_home_is_rejected_by_location_domain() -> None:
    harness = _RobotCapabilityHarness()
    result = harness._evaluate_capability_transition(
        task={
            "event_name": "assemble_mcp_at_home",
            "resource_jid": "xarm6@localhost",
            "part_name": "MCP",
            "expected_end_state": {
                "resource_state": "idle",
                "resource_location": "home",
                "held_part": None,
                "part_state": "assembled",
                "part_location": "home",
            },
        },
        resource_snapshot={
            **_idle_snapshot(),
            "current_state": "picked",
            "held_part": "MCP",
            "gripper_state": "closed",
        },
        part_context={
            "part_name": "MCP",
            "current_state": "in_gripper",
            "current_location": "xarm6@localhost_gripper",
            "goal_location": "assembly_board-v1",
        },
    )

    assert result["allowed"] is False
    assert result["constraint_code"] == "state_value_outside_ra_domain"


def test_place_insert_requires_goal_and_place_release_requires_staging() -> None:
    harness = _RobotCapabilityHarness()
    gripper_location = "xarm6@localhost_gripper"
    snapshot = {
        **_idle_snapshot(),
        "current_state": "positioned",
        "current_location": "prusa-mk3",
        "current_pose_ref": "prusa-mk3",
        "held_part": "MCP",
        "gripper_state": "closed",
        "carried_entity_location": gripper_location,
    }
    part_context = {
        "part_name": "MCP",
        "current_state": "in_transit",
        "current_location": gripper_location,
        "goal_location": "assembly_board-v1",
    }
    insert = harness._evaluate_capability_transition(
        task={
            "event_name": "place_insert",
            "resource_jid": "xarm6@localhost",
            "part_name": "MCP",
            "expected_end_state": {"part_state": "assembled"},
        },
        resource_snapshot={
            **snapshot,
            "current_location": "assembly_board-v1",
            "current_pose_ref": "assembly_board-v1",
        },
        part_context=part_context,
    )
    insert_at_staging = harness._evaluate_capability_transition(
        task={
            "event_name": "place_insert",
            "resource_jid": "xarm6@localhost",
            "part_name": "MCP",
            "expected_end_state": {
                "resource_state": "placed",
                "resource_location": "prusa-mk3",
                "held_part": None,
                "part_state": "assembled",
                "part_location": "prusa-mk3",
            },
        },
        resource_snapshot=snapshot,
        part_context=part_context,
    )
    release = harness._evaluate_capability_transition(
        task={
            "event_name": "place_release",
            "resource_jid": "xarm6@localhost",
            "part_name": "MCP",
            "expected_end_state": {
                "held_part": None,
                "part_state": "ready",
                "part_location": "prusa-mk3",
            },
        },
        resource_snapshot=snapshot,
        part_context=part_context,
    )

    assert insert["allowed"] is True
    assert [row["event_name"] for row in insert["_atomic_transitions"]] == [
        "place_insert"
    ]
    assert insert_at_staging["allowed"] is False
    assert insert_at_staging["constraint_code"] == (
        "unsatisfied_guard_predicate"
    )
    assert release["allowed"] is True
    assert release["calculated_successor"]["part_state"]["condition"] == (
        "ready"
    )


def test_generated_observed_pose_pick_uses_overlay_semantics() -> None:
    harness = _RobotCapabilityHarness()
    gripper_location = "xarm6@localhost_gripper"
    part_context = {
        "part_name": "LG",
        "current_state": "misplaced",
        "current_location": None,
        "observed_pose": {
            "x": 0.0,
            "y": 0.2,
            "z": 1.035,
        },
        "goal_location": "assembly_board-v1",
    }
    transition = harness._evaluate_capability_transition(
        task={
            "event_name": "recover_lg_from_observation",
            "resource_jid": "xarm6@localhost",
            "part_name": "LG",
            "expected_end_state": {
                "resource_state": "picked",
                "held_part": "LG",
                "part_state": "in_gripper",
                "part_location": gripper_location,
            },
        },
        resource_snapshot=_idle_snapshot(),
        part_context=part_context,
    )

    assert transition["allowed"] is True
    assert transition["calculated_successor"]["held_part"] == "LG"
    assert [
        row["event_name"] for row in transition["_atomic_transitions"]
    ] == ["recover_lg_from_observation"]


def test_observed_pose_pick_approach_is_not_enabled() -> None:
    harness = _RobotCapabilityHarness()
    observed_context = {
        "part_name": "LG",
        "current_state": "misplaced",
        "current_location": "observed_pose",
        "observed_pose": {
            "x": 0.0,
            "y": 0.2,
            "z": 1.035,
        },
    }

    observed_enabled = harness._enabled_capability_instances(
        resource_snapshot=_idle_snapshot(),
        part_contexts=[observed_context],
    )
    known_location_enabled = harness._enabled_capability_instances(
        resource_snapshot=_idle_snapshot(),
        part_contexts=[_part_context("LG", "prusa-mk4-1")],
    )

    assert "pick_approach" not in {
        row["task"]["event_name"] for row in observed_enabled
    }
    assert any(
        row["task"]["event_name"] == "pick_approach"
        and row["task"].get("part_name") == "LG"
        and row["task"]["expected_end_state"]["resource_state"]["location"]
        == "prusa-mk4-1"
        for row in known_location_enabled
    )


def test_current_dynamic_state_value_is_valid_without_a_new_binding() -> None:
    harness = _RobotCapabilityHarness()
    snapshot = {
        **_idle_snapshot(),
        "current_state": "picked",
        "held_part": "MCP",
        "gripper_state": "closed",
    }
    current_value = harness._evaluate_capability_transition(
        task={
            "event_name": "move_home",
            "expected_start_state": {"held_part": "MCP"},
            "expected_end_state": {"held_part": "MCP"},
        },
        resource_snapshot=snapshot,
    )
    assert current_value["constraint_code"] == "unsatisfied_guard_predicate"

    new_unbound_value = harness._evaluate_capability_transition(
        task={
            "event_name": "move_home",
            "expected_start_state": {"held_part": "LG"},
            "expected_end_state": {"held_part": "LG"},
        },
        resource_snapshot=snapshot,
    )
    assert new_unbound_value["constraint_code"] == (
        "state_value_outside_ra_domain"
    )


def test_synthetic_non_robot_provider_uses_its_own_fact_names() -> None:
    class MillingLikeHarness:
        _executable_required_runtime_facts = staticmethod(
            ResourceAgent._executable_required_runtime_facts
        )
        _executable_runtime_fact_names = staticmethod(
            ResourceAgent._executable_runtime_fact_names
        )
        _configured_event_required_runtime_facts = (
            ResourceAgent._configured_event_required_runtime_facts
        )
        _capability_execution_arguments = (
            ResourceAgent._capability_execution_arguments
        )
        _capability_atomic_transition = (
            ResourceAgent._capability_atomic_transition
        )
        _bind_capabilities = ResourceAgent._bind_capabilities
        _evaluate_capability_transition = (
            ResourceAgent._evaluate_capability_transition
        )
        _resolve_generated_capability = (
            ResourceAgent._resolve_generated_capability
        )
        _generated_successor_from_action_target = (
            ResourceAgent._generated_successor_from_action_target
        )
        _generated_successor_consistency_finding = (
            ResourceAgent._generated_successor_consistency_finding
        )
        _future_goal_capability_instances = (
            ResourceAgent._future_goal_capability_instances
        )
        capability_goal_requirements = (
            ResourceAgent.capability_goal_requirements
        )
        capability_runtime_facts = ResourceAgent.capability_runtime_facts
        capability_runtime_fact_tables = (
            ResourceAgent.capability_runtime_fact_tables
        )
        capability_state_valuation = ResourceAgent.capability_state_valuation
        public_capability_state_projections = (
            ResourceAgent.public_capability_state_projections
        )
        public_state_values = ResourceAgent.public_state_values

        @classmethod
        def capability_runtime_fact_names(cls) -> set[str]:
            return {"resource_jid", "program_ref"}

    capabilities = {
        "state_variables": {
            "job_state": {
                "scope": "resource",
                "domain": ["idle"],
                "parameter_values": ["program_ref"],
            }
        },
        "events": [
            {
                "event_name": "load_program",
                "guards": {"job_state": {"equals": "idle"}},
                "updates": {
                    "job_state": {"set_from_param": "program_ref"}
                },
            }
        ],
    }
    harness = MillingLikeHarness()
    harness.jid = "resource@localhost"
    harness.static_capabilities = capabilities
    harness._configured_capability_declarations = capabilities
    harness.executables = {"load_program": _configured_action}
    assert configured_capability_errors(
        capabilities,
        executable_names={"load_program"},
        runtime_fact_names=harness.capability_runtime_fact_names(),
    ) == []
    transition = harness._evaluate_capability_transition(
        task={
            "event_name": "load_program",
            "program_ref": "program-42",
            "expected_end_state": {"job_state": "program-42"},
        },
        resource_snapshot={
            "resource_jid": "resource@localhost",
            "job_state": "idle",
            "program_ref": "program-42",
        },
        part_context={"program_ref": "program-42"},
    )
    assert transition["allowed"] is True
    assert transition["calculated_successor"]["job_state"] == "program-42"
    generated_transition = harness._evaluate_capability_transition(
        task={
            "event_name": "schedule_inspection",
            "expected_start_state": {"job_state": "idle"},
            "expected_end_state": {"job_state": "program-42"},
        },
        resource_snapshot={
            "resource_jid": "resource@localhost",
            "job_state": "idle",
            "program_ref": "program-42",
        },
    )
    assert generated_transition["allowed"] is True
    assert generated_transition["calculated_successor"] == {
        "job_state": "program-42"
    }
    assert harness.public_capability_state_projections(
        resource_snapshot={
            "resource_jid": "resource@localhost",
            "job_state": "idle",
        },
        part_contexts=[{"part_name": "LG"}],
    ) == [
        {
            "resource_jid": "resource@localhost",
            "state": {"job_state": "idle"},
        }
    ]
    assert harness.public_state_values(
        resource_snapshot={
            "resource_jid": "resource@localhost",
            "job_state": "idle",
            "program_ref": "program-42",
        },
        part_contexts=[{"part_name": "LG"}],
    ) == {
        "resource_jid": "resource@localhost",
        "state_values": {"job_state": ["idle", "program-42"]},
    }
    future = harness._future_goal_capability_instances(
        resource_snapshot={
            "resource_jid": "resource@localhost",
            "job_state": "idle",
            "program_ref": "program-42",
        },
        part_contexts=[],
        goal_conditions=[
            {
                "condition_id": "job_goal",
                "entity_kind": "resource",
                "entity": "resource@localhost",
                "field": "job_state",
                "expected": "program-42",
            }
        ],
    )
    assert [row["task"]["event_name"] for row in future] == [
        "load_program"
    ]
    session_state = {
        "symbolic_resources": {
            "resource@localhost": {
                "resource_jid": "resource@localhost",
                "job_state": "idle",
            }
        },
        "symbolic_parts": {},
        "public_capability_state_projections": [
            {
                "resource_jid": "resource@localhost",
                "state": {"job_state": "idle"},
            }
        ],
    }
    multi_turn._apply_task_effects_to_symbolic_state(
        {
            "resource_jid": "resource@localhost",
            "expected_end_state": transition["calculated_successor"],
        },
        session_state,
    )
    assert session_state["public_capability_state_projections"] == [
        {
            "resource_jid": "resource@localhost",
            "state": {"job_state": "program-42"},
        }
    ]


def test_public_validation_does_not_expose_private_transition_data() -> None:
    harness = _RobotCapabilityHarness()
    harness.get_recovery_snapshot = _idle_snapshot
    harness.recovery_validation_resource_matches = (
        ResourceAgent.recovery_validation_resource_matches.__get__(harness)
    )
    harness.recovery_physical_validation_snapshot = (
        ResourceAgent.recovery_physical_validation_snapshot
    )
    harness._grounded_action_from_atomic_transitions = (
        ResourceAgent._grounded_action_from_atomic_transitions.__get__(
            harness
        )
    )
    harness.check_recovery_physical_feasibility = (
        lambda **_kwargs: {"allowed": True}
    )
    response = ResourceAgent.validate_recovery_outline_physical_candidates(
        harness,
        {
            "candidates": [
                {
                    "candidate_index": 0,
                    "task": _generated_pick_task("LG", "prusa-mk4-1"),
                    "physical_input": {
                        "part_context": _part_context(
                            "LG", "prusa-mk4-1"
                        )
                    },
                }
            ]
        },
    )
    row = response["results"][0]
    assert row["allowed"] is True
    assert row["transition_feasibility"]["calculated_successor"][
        "held_part"
    ] == "LG"
    assert "calculated_successor" not in row
    assert "successor_status" not in row
    assert "successor_status" not in row["transition_feasibility"]
    serialized_public = json.dumps(
        {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        },
        sort_keys=True,
    )
    for private_name in (
        "runtime_facts",
        "guards",
        "updates",
        "capability_witness",
        "capability_trace",
        "fingerprint",
    ):
        assert private_name not in serialized_public
    assert [
        item["event_name"] for item in row["_cca_atomic_transitions"]
    ] == ["secure_lg"]


def test_execution_recalculates_from_fresh_snapshot() -> None:
    harness = _RobotCapabilityHarness()
    harness.logger = logging.getLogger("test.fresh_capability_execution")
    harness._current_state = "failed"
    changed_snapshot = {**_idle_snapshot(), "current_state": "failed"}
    with patch(
        "cais_spade_llm.resources.resource_primitives."
        "get_resource_recovery_snapshot",
        return_value=changed_snapshot,
    ):
        result = asyncio.run(
            ResourceAgent.execute_recovery_macro(
                harness,
                macro_name="secure_lg",
                event_name="secure_lg",
                part_name="LG",
                outline_expected_start_state={
                    "resource_state": "idle",
                    "held_part": None,
                    "part_state": "ready",
                    "part_location": "prusa-mk4-1",
                },
                expected_end_state={
                    "resource_state": "picked",
                    "held_part": "LG",
                    "part_state": "in_gripper",
                    "part_location": "xarm6@localhost_gripper",
                },
                part_context=_part_context("LG", "prusa-mk4-1"),
            )
        )
    assert result["status"] == "revalidation_required"
    assert result["observations"]["constraint_code"] == (
        "expected_start_state_mismatch"
    )


def test_cca_validates_every_private_atomic_transition() -> None:
    async def run_validation() -> tuple[list[str], dict]:
        validated_event_names: list[str] = []

        def validate(**kwargs: object) -> dict:
            validated_event_names.append(
                str(dict(kwargs.get("task") or {}).get("event_name") or "")
            )
            return {
                "is_safe": True,
                "findings": [],
                "safety_ctx": {},
                "safety_dfa_states_before": {},
                "safety_dfa_states_after": {},
            }

        async def ready(*, timeout_s: float) -> bool:
            del timeout_s
            return True

        owner = SimpleNamespace(
            jid="cca@localhost",
            logger=logging.getLogger("test.cca_atomic_transitions"),
            loop=asyncio.get_running_loop(),
            presence=None,
            web=None,
            safety_file=None,
            safety_rules=[],
            safety_monitor=SimpleNamespace(
                current_states={},
                running_aps=set(),
            ),
            _wait_for_safety_monitor_ready=ready,
        )
        behaviour = CentralControllerAgent._RecoveryOutlineSafetyValidation()
        behaviour.set_agent(owner)
        message = SimpleNamespace(
            sender="product@localhost",
            body=json.dumps(
                {
                    "request_id": "request",
                    "product_jid": "product@localhost",
                    "candidates": [
                        {
                            "candidate_index": 0,
                            "_cca_atomic_transitions": [
                                {
                                    "event_name": "pick_approach",
                                    "before": {
                                        "resource_state": "idle"
                                    },
                                    "after": {
                                        "resource_state": "at_pick"
                                    },
                                },
                                {
                                    "event_name": "pick_grasp",
                                    "before": {
                                        "resource_state": "at_pick"
                                    },
                                    "after": {
                                        "resource_state": "picked"
                                    },
                                },
                            ],
                            "safety_input": {
                                "task": {
                                    "resource_jid": "xarm6@localhost",
                                    "part_name": "LG",
                                },
                                "signature": {},
                                "pre_resources": {
                                    "xarm6@localhost": {}
                                },
                                "pre_parts": {"LG": {}},
                                "projected_resources": {},
                                "projected_parts": {},
                                "llm_input": {},
                            },
                        }
                    ],
                }
            ),
        )

        async def receive(*, timeout: float) -> object:
            del timeout
            return message

        sent: list[object] = []

        async def send(
            _behaviour: object,
            reply: object,
            **_kwargs: object,
        ) -> str:
            sent.append(reply)
            return "sent"

        behaviour.receive = receive
        with patch(
            "cais_spade_llm.agents.central_controller."
            "central_controller_agent."
            "validate_outline_macro_recovery_safety",
            side_effect=validate,
        ), patch(
            "cais_spade_llm.agents.central_controller."
            "central_controller_agent.send_agent_message",
            side_effect=send,
        ):
            await behaviour.run()
        return validated_event_names, json.loads(sent[0].body)

    event_names, response = asyncio.run(run_validation())
    assert event_names == ["pick_approach", "pick_grasp"]
    assert response["results"][0]["is_safe"] is True
    assert "atomic_validation_results" not in response["results"][0]
