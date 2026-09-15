from __future__ import annotations

"""Tests for selected-RA context capture and direct primitive composition."""


import asyncio
import dataclasses
import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.spec2primitives.adapters import in_process_robot_agent
from cais_spade_llm.spec2primitives.adapters.dual_gazebo import DUAL_GAZEBO_NAME
from cais_spade_llm.spec2primitives.adapters.in_process_robot_agent import (
    InProcessRobotAgentCompositionRuntime,
)
from cais_spade_llm.spec2primitives.agents.ra import (
    composition_context,
    context_handoff,
    primitive_composition,
)
from cais_spade_llm.spec2primitives.agents.ra.context_handoff import (
    RAContextHandoffError,
    SelectedRAAssignmentEnvelope,
    activate_selected_ra_context,
    read_phase_5_1_diagnostic,
)
from cais_spade_llm.spec2primitives.agents.ra.primitive_composition import (
    PrimitiveCompositionError,
    author_primitive_program_candidate,
    read_primitive_composition_diagnostic,
)
from cais_spade_llm.spec2primitives.agents.ra.validation_scope import (
    GAZEBO_PICK_PLACE_SCOPE, VALIDATION_SCOPE, supported_primitive_symbols,
)
from cais_spade_llm.spec2primitives.tests.test_pa_completion import (
    persist_native_completion_fixture,
)

_EXPECTED_XARM6_SYNTHESIS_SYMBOLS = [
    "compute_pick_targets",
    "compute_place_targets",
    "detect_parts",
    "grasp_part",
    "move_cartesian",
    "move_relative",
    "move_to_named_pose",
    "release_part",
]


class _AssignedContextRuntime:
    def __init__(
        self,
        *,
        self_jid: str = "xarm6@localhost",
        transform: Callable[[dict[str, object]], None] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.self_jid = self_jid
        self.transform = transform
        self.failure = failure
        self.assignments: list[SelectedRAAssignmentEnvelope] = []

    async def request_assigned_context(
        self,
        assignment: SelectedRAAssignmentEnvelope,
    ) -> dict[str, object]:
        assignment.assert_addressed_to(self.self_jid)
        self.assignments.append(assignment)
        if self.failure is not None:
            raise self.failure
        response: dict[str, object] = {
            "resource_jid": self.self_jid,
            "assignment_fingerprint": assignment.fingerprint,
            "robot_state": {
                "execution_mode": "simulation",
                "controller_ready": True,
                "held_part": None,
                "current_state": "idle",
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "gripper_state": "open",
            },
            "primitive_catalog": _valid_catalog(),
        }
        if self.transform is not None:
            self.transform(response)
        return response


class _LiveRobotAgent:
    def __init__(
        self,
        *,
        jid: str = "xarm6@localhost",
        execution_mode: str = "simulation",
        alive: bool = True,
        robot_state: dict[str, object] | None = None,
        primitive_catalog: list[dict[str, object]] | None = None,
        composition_response: dict[str, object] | None = None,
        feasibility_allowed: bool = True,
    ) -> None:
        self.jid = jid
        self.execution_mode = execution_mode
        self.alive = alive
        self.robot_state = (
            robot_state
            if robot_state is not None
            else {
                "resource_jid": jid,
                "current_state": "idle",
                "controller_ready": False,
            }
        )
        self.primitive_catalog = (
            primitive_catalog if primitive_catalog is not None else _raw_live_catalog()
        )
        self.composition_response = composition_response or {
            "action": _program_action([("compute_pick_targets", {}), ("grasp_part", {})])
        }
        self.composition_calls: list[dict[str, object]] = []
        self.feasibility_allowed = feasibility_allowed
        self.feasibility_calls: list[dict[str, object]] = []

    def is_alive(self) -> bool:
        return self.alive

    def get_recovery_snapshot(self) -> dict[str, object]:
        return deepcopy(self.robot_state)

    def recovery_synthesis_primitive_catalog(self) -> list[dict[str, object]]:
        return deepcopy(self.primitive_catalog)

    def check_recovery_physical_feasibility(
        self,
        *,
        part_context: dict[str, object],
        recovery_snapshot: dict[str, object],
        operation_kind: str,
        grounded_action: dict[str, object],
    ) -> dict[str, object]:
        self.feasibility_calls.append(
            {
                "part_context": deepcopy(part_context),
                "recovery_snapshot": deepcopy(recovery_snapshot),
                "operation_kind": operation_kind,
                "grounded_action": deepcopy(grounded_action),
            }
        )
        return {
            "allowed": self.feasibility_allowed,
            "reason": ("workspace accepted" if self.feasibility_allowed else "workspace rejected"),
        }

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, object],
        tools: list[dict[str, object]] | None = None,
        max_tool_rounds: int = 3,
        include_agent_instructions: bool = True,
    ) -> dict[str, object]:
        self.composition_calls.append(
            {
                "prompt": prompt,
                "response_format": deepcopy(response_format),
                "tools": tools,
                "max_tool_rounds": max_tool_rounds,
                "include_agent_instructions": include_agent_instructions,
            }
        )
        return deepcopy(self.composition_response)


class _ProgramRuntime:
    def __init__(self, actions: list[dict[str, Any]], *, on_call: Callable | None = None) -> None:
        self.actions = actions
        self.calls: list[dict[str, Any]] = []
        self.on_call = on_call

    async def author_composition_action(
        self,
        assignment: SelectedRAAssignmentEnvelope,
        *,
        prompt: str,
        response_format: Mapping,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "assignment": assignment,
                "prompt": prompt,
                "response_format": deepcopy(response_format),
            }
        )
        if self.on_call is not None:
            self.on_call()
        return {"action": deepcopy(self.actions[min(len(self.calls) - 1, len(self.actions) - 1)])}


class _LiveRobotAgentHost:
    def __init__(
        self,
        resource_agents: list[_LiveRobotAgent],
        *,
        system_running: bool = True,
        gazebo_state: str = "stopped",
        readiness: tuple[bool, str] = (True, ""),
        startup_agent: _LiveRobotAgent | None = None,
        startup_error: str | None = None,
    ) -> None:
        self.resource_agents = resource_agents
        self._spec2primitives_robot_agent: _LiveRobotAgent | None = None
        self.system_running = system_running
        self.execution_mode = "simulation"
        self.robot_env = "gazebo"
        self.gazebo_state = gazebo_state
        self.readiness = readiness
        self.startup_agent = startup_agent
        self.startup_error = startup_error
        self.runtime_calls = 0
        self.readiness_calls = 0
        self.readiness_forces: list[bool] = []
        self.start_calls = 0
        self.full_system_start_calls = 0

    def ros2_proc_status(self, name: str) -> str:
        assert name == DUAL_GAZEBO_NAME
        return self.gazebo_state

    def simulation_start_ready(self, force: bool = False) -> tuple[bool, str]:
        self.readiness_calls += 1
        self.readiness_forces.append(force)
        return self.readiness

    async def start_spec2primitives_robot_agent(
        self,
        resource_jid: str,
        execution_mode: str,
    ) -> _LiveRobotAgent | None:
        self.start_calls += 1
        assert resource_jid == "xarm6@localhost"
        assert execution_mode == "simulation"
        if self.startup_error is not None:
            raise RuntimeError(self.startup_error)
        self._spec2primitives_robot_agent = self.startup_agent
        return self.startup_agent

    async def start_system(self) -> None:
        self.full_system_start_calls += 1
        raise AssertionError("Phase 5.1 must not start the full Agent System")

    async def _run_on_agent_runtime(self, coroutine: Any) -> Any:
        self.runtime_calls += 1
        return await coroutine


class _PlanOnlyRuntime:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def validate_state_locations(self, request):
        from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
            OfflineMoveItPlanning,
        )

        self.requests.append(deepcopy(request))
        return await OfflineMoveItPlanning().validate_plan_only_allocation(request)

    def validate(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(deepcopy(request))
        endpoint = {
            "status": "accepted",
            "message": "collision-aware plan accepted",
            "error_code": 1,
        }
        return {
            "status": "accepted",
            "current_state": dict(endpoint),
            "desired_state": dict(endpoint),
            "feedback": None,
        }


def _plan_only_request(resource_jid: str) -> dict[str, object]:
    request = {
        "process_symbol": "assembly",
        "process_iri": "https://cais-spade-llm.local/process/assembly",
        "feature_iri": "https://example.local/feature_0001",
        "resource_symbol": resource_jid.split("@", 1)[0],
        "resource_iri": f"https://cais-spade-llm.local/resource/{resource_jid.split('@', 1)[0]}",
        "resource_jid": resource_jid,
        "execution_mode": "simulation",
        "moveit_group": "selected_arm",
        "end_effector_link": "selected_tool0",
        "target_frame": "world",
        "validation_scope": "endpoint_motion",
        "checked_constraints": [
            "positional_ik",
            "collision_aware_endpoints",
            "path_between_endpoints",
        ],
        "unvalidated_constraints": [
            "grasping",
            "end_effector_orientation",
            "attached_object_geometry",
            "assembly_tolerance",
            "force_contact",
            "insertion_constraints",
        ],
        "current_state": {
            "state_iri": "https://example.local/currentstate_0001",
            "evidence_handle": "neutral_candidate_0001",
            "translation_m": [0.1, 0.2, 0.3],
            "location_record_ref": "current.json",
            "location_record_sha256": "0" * 64,
        },
        "desired_state": {
            "state_iri": "https://example.local/desiredstate_0001",
            "evidence_handle": "neutral_candidate_0002",
            "translation_m": [0.4, 0.5, 0.6],
            "location_record_ref": "desired.json",
            "location_record_sha256": "1" * 64,
        },
        "mode": "plan_only",
        "motion_executed": False,
        "request_fingerprint": "2" * 64,
    }

    request["state_locations"] = {
        state: [request.pop(state)] for state in ("current_state", "desired_state")
    }
    for field in ("checked_constraints", "unvalidated_constraints", "request_fingerprint"):
        request.pop(field)
    request.update(
        validation_scope="moveit_state_location_reachability",
        motion_plan_service="/plan_kinematic_path",
        position_tolerance_m=0.005,
    )
    return request


@pytest.mark.parametrize("resource_jid", ["xarm6@localhost", "ur5e@localhost"])
def test_plan_only_adapter_checks_each_arm_without_robot_agent_activation(
    monkeypatch: pytest.MonkeyPatch,
    resource_jid: str,
) -> None:
    host = _LiveRobotAgentHost([], system_running=False, gazebo_state="running")
    host.execution_mode = "physical"
    host.robot_env = "unchanged"
    moveit = _PlanOnlyRuntime()
    runtime = InProcessRobotAgentCompositionRuntime(
        host,
        moveit_plan_only_runtime=moveit,
    )

    def _forbidden_lookup(_resource_jid: str) -> None:
        raise AssertionError("Planning must not look up a RobotAgent.")

    monkeypatch.setattr(runtime, "_selected_agent_or_none", _forbidden_lookup)
    request = _plan_only_request(resource_jid)
    response = asyncio.run(runtime.validate_plan_only_allocation(request))

    assert response["status"] == "accepted"
    assert moveit.requests == [request]
    assert host.resource_agents == []
    assert host.start_calls == host.full_system_start_calls == 0
    assert host.runtime_calls == host.readiness_calls == 0
    assert host.system_running is False
    assert host.execution_mode == "physical"
    assert host.robot_env == "unchanged"


def test_location_planning_bypasses_static_robot_agent_precheck() -> None:
    selected = _LiveRobotAgent(feasibility_allowed=False)
    moveit = _PlanOnlyRuntime()
    runtime = InProcessRobotAgentCompositionRuntime(
        _LiveRobotAgentHost([selected], gazebo_state="running"),
        moveit_plan_only_runtime=moveit,
    )

    response = asyncio.run(
        runtime.validate_plan_only_allocation(_plan_only_request("xarm6@localhost"))
    )

    assert response["status"] == "accepted"
    assert selected.feasibility_calls == []
    assert len(moveit.requests) == 1


@pytest.mark.parametrize("variant", ["stopped", "status_error"])
def test_plan_only_adapter_requires_running_spec2primitives_gazebo(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    host = _LiveRobotAgentHost([], system_running=False)
    moveit = _PlanOnlyRuntime()
    runtime = InProcessRobotAgentCompositionRuntime(host, moveit_plan_only_runtime=moveit)

    if variant == "status_error":

        def _unavailable_status(_name: str) -> str:
            raise RuntimeError("Process status is unavailable.")

        monkeypatch.setattr(host, "ros2_proc_status", _unavailable_status)

    with pytest.raises(RAContextHandoffError, match="Dual Gazebo"):
        asyncio.run(runtime.validate_plan_only_allocation(_plan_only_request("xarm6@localhost")))

    assert moveit.requests == []
    assert host.resource_agents == []
    assert host.start_calls == host.full_system_start_calls == 0
    assert host.runtime_calls == host.readiness_calls == 0
    assert host.system_running is False
    assert host.execution_mode == "simulation"
    assert host.robot_env == "gazebo"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_mode", "physical"),
        ("mode", "execute"),
        ("motion_executed", True),
        ("validation_scope", "cartesian_pick_place"),
        ("resource_jid", ""),
    ],
)
def test_plan_only_adapter_rejects_invalid_request_before_runtime_access(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    host = _LiveRobotAgentHost([], system_running=False, gazebo_state="running")
    moveit = _PlanOnlyRuntime()
    runtime = InProcessRobotAgentCompositionRuntime(host, moveit_plan_only_runtime=moveit)

    def _forbidden_status(_name: str) -> str:
        raise AssertionError("Invalid requests must not read runtime status.")

    monkeypatch.setattr(host, "ros2_proc_status", _forbidden_status)
    request = _plan_only_request("xarm6@localhost")
    request[field] = value

    with pytest.raises(ValueError):
        asyncio.run(runtime.validate_plan_only_allocation(request))

    assert moveit.requests == []
    assert host.start_calls == host.full_system_start_calls == 0
    assert host.runtime_calls == host.readiness_calls == 0


def test_retired_cartesian_request_is_rejected_before_planning() -> None:
    xarm6 = _LiveRobotAgent(jid="xarm6@localhost", feasibility_allowed=False)
    ur5e = _LiveRobotAgent(jid="ur5e@localhost", feasibility_allowed=False)
    moveit = _PlanOnlyRuntime()
    runtime = InProcessRobotAgentCompositionRuntime(
        _LiveRobotAgentHost([xarm6, ur5e]),
        moveit_plan_only_runtime=moveit,
    )
    request = _plan_only_request("xarm6@localhost")
    request["motion_mode"] = "cartesian_pick_place"
    request["validation_scope"] = "cartesian_pick_place"

    with pytest.raises(ValueError):
        asyncio.run(runtime.validate_plan_only_allocation(request))
    assert moveit.requests == []
    assert xarm6.feasibility_calls == ur5e.feasibility_calls == []


def test_in_process_ra_adapter_captures_exact_live_state_and_atomic_catalog(
    tmp_path: Path,
) -> None:
    persist_native_completion_fixture(tmp_path)
    raw_catalog = _raw_live_catalog()
    agent = _LiveRobotAgent(primitive_catalog=raw_catalog)
    host = _LiveRobotAgentHost([agent])
    runtime = InProcessRobotAgentCompositionRuntime(host)

    captured = asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert host.start_calls == 0
    assert host.runtime_calls == 1
    assert captured.robot_state.robot_state["controller_ready"] is False
    assert [
        entry["primitive_symbol"] for entry in captured.primitive_catalog.primitive_catalog
    ] == _EXPECTED_XARM6_SYNTHESIS_SYMBOLS
    detect_parts = next(
        entry
        for entry in captured.primitive_catalog.primitive_catalog
        if entry["primitive_symbol"] == "detect_parts"
    )
    assert detect_parts["typed_parameters"] == (
        [{"name": "part_name", "type": "string", "required": False}]
    )
    assert detect_parts["typed_results"] == [{"name": "pose", "type": "object"}]
    assert detect_parts["parameter_schemas"] == raw_catalog[2]["params"]
    assert detect_parts["result_schemas"] == raw_catalog[2]["output_schema"]
    assert detect_parts["conditions"] == {"controller_ready": True}
    assert detect_parts["effects"] == {"part_observed": True}
    assert raw_catalog == _raw_live_catalog()


def test_in_process_ra_adapter_asks_exact_robot_agent_for_primitive_program(tmp_path: Path) -> None:
    """Author directly from captured context with one isolated selected-RA call."""
    runtime, agent, _ = _prepare_composition(tmp_path)
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert candidate.path.name == "candidate.json"
    assert candidate.record["primitive_steps"] == [
        {"primitive_symbol": "compute_pick_targets", "params": {}},
        {"primitive_symbol": "grasp_part", "params": {}},
    ]
    assert len(agent.composition_calls) == 1
    call = agent.composition_calls[0]
    assert call["tools"] is None
    assert call["max_tool_rounds"] == 0
    assert call["include_agent_instructions"] is False
    composition_input = _composition_input_from_prompt(call["prompt"])
    assert composition_input["target_feature"]["product_requirement"] == "assemble medium gear"
    assert "ontology_projection" not in composition_input
    view = read_primitive_composition_diagnostic(tmp_path)
    final_view_path = sorted((tmp_path / "products/grounding/product_context").glob("view_*.json"))[
        -1
    ]
    assert (
        view["composition_input"]["ontology_projection"]["assertions"]
        == _read_json(final_view_path)["assertions"]
    )
    variants = call["response_format"]["schema"]["properties"]["action"]["anyOf"]
    proposal = next(item for item in variants if item["properties"]["kind"]["enum"] == ["propose"])
    assert (
        proposal["properties"]["primitive_steps"]["items"]["properties"]["primitive_symbol"]["enum"]
        == [symbol for symbol in _EXPECTED_XARM6_SYNTHESIS_SYMBOLS if symbol in supported_primitive_symbols(VALIDATION_SCOPE)]
    )
    assert not (tmp_path / "composition/primitive_program_drafts").exists()


def test_composition_rejects_assignment_inconsistent_ontology_projection(
    tmp_path: Path,
) -> None:
    completion = persist_native_completion_fixture(tmp_path).to_record()
    final_view_path = sorted((tmp_path / "products/grounding/product_context").glob("view_*.json"))[
        -1
    ]
    final_view = _read_json(final_view_path)
    final_view["assertions"] = [
        assertion
        for assertion in final_view["assertions"]
        if assertion["predicate"] != "http://PAonto.com#runsOnResource"
    ]
    final_view["abox_fingerprint"] = _fingerprint(final_view["assertions"])
    final_view["fingerprint"] = _record_fingerprint(final_view)
    _write_json(final_view_path, final_view)

    completion_path = tmp_path / "interaction_record/context_completion_0001.json"
    completion["abox_fingerprint"] = final_view["abox_fingerprint"]
    completion["fingerprint"] = _record_fingerprint(completion)
    _write_json(completion_path, completion)

    adapter = InProcessRobotAgentCompositionRuntime(_LiveRobotAgentHost([_LiveRobotAgent()]))
    asyncio.run(activate_selected_ra_context(adapter, tmp_path))
    runtime = _ProgramRuntime([_program_action([("release_part", {})])])
    with pytest.raises(RAContextHandoffError, match="runsOnResource"):
        asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert runtime.calls == []
    assert not (tmp_path / "composition/primitive_program_candidates").exists()


def test_composition_pins_context_and_preserves_phase_4(tmp_path: Path) -> None:
    """A program pins its own authority without copying or editing source records."""
    adapter, _, _ = _prepare_composition(tmp_path)
    context = context_handoff.load_selected_ra_context_snapshot(tmp_path)
    paths = {
        "pa_completion": tmp_path / "interaction_record/context_completion_0001.json",
        "assignment": context.assignment_path,
        "robot_state": context.robot_state_path,
        "primitive_catalog": context.primitive_catalog_path,
    }
    before = {name: path.read_bytes() for name, path in paths.items()}
    original_records = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    runtime = _ProgramRuntime(
        [_program_action([("release_part", {}), ("move_cartesian", {}), ("release_part", {})])]
    )
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    request = _read_json(candidate.path.parent / "request.json")
    assert request["context_refs"] == {
        name: {
            "ref": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(before[name]).hexdigest(),
        }
        for name, path in paths.items()
    }
    assert "draft_ref" not in request
    assert candidate.record["primitive_steps"] == [
        {"primitive_symbol": symbol, "params": {}}
        for symbol in ("release_part", "move_cartesian", "release_part")
    ]
    diagnostic = read_primitive_composition_diagnostic(tmp_path)
    composition_input = diagnostic["composition_input"]
    assert diagnostic["status"] == "proposed"
    assert set(composition_input) == {
        "target_feature",
        "selected_resource",
        "ontology_projection",
        "robot_state",
        "primitive_catalog",
        "grounded_context",
        "resolved_task_records",
    }
    assert composition_input["robot_state"] == context.robot_state.robot_state
    assert composition_input["primitive_catalog"] == composition_context._composition_catalog_view(
        context.primitive_catalog.primitive_catalog
    )
    assert composition_input["ontology_projection"]["assertions"]
    target_feature = composition_input["target_feature"]
    assert target_feature["product_requirement"] == "assemble medium gear"
    assert (
        target_feature["desired_state"]["statement"]["text"]
        == "The medium gear is assembled as requested."
    )
    assert {item["name"] for item in target_feature["resolved_state_values"]} == {
        "medium_gear_location",
        "assembly_board_shaft_location",
    }
    association = target_feature["assembly_feature_association"][0]
    assert [item["state_name"] for item in association["assembly_features"]] == [
        "current_state",
        "current_state",
    ]
    assert all(
        set(item) == {"record_ref", "record_type"}
        for item in composition_input["grounded_context"]["typed_records"]
    )
    delivered = _composition_input_from_prompt(runtime.calls[0]["prompt"])
    assert delivered["target_feature"]["product_requirement"] == target_feature["product_requirement"]
    assert "value_ref" not in json.dumps(delivered["target_feature"])
    assert delivered["robot_state"] == composition_input["robot_state"]
    assert "structural_steps" not in delivered
    serialized = json.dumps(candidate.to_record())
    assert "composition_input" not in serialized
    assert "ontology_projection" not in serialized
    assert {name: path.read_bytes() for name, path in paths.items()} == before
    assert not (tmp_path / "composition/primitive_program_drafts").exists()
    second = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert second.path.parent.name == "attempt_0002"
    assert second.record["primitive_steps"] == candidate.record["primitive_steps"]
    assert {path: path.read_bytes() for path in original_records} == original_records


def test_ra_receives_resolved_target_state_values_without_persisting_them(tmp_path: Path) -> None:
    persist_native_completion_fixture(tmp_path, include_target_state_values=True)
    adapter = InProcessRobotAgentCompositionRuntime(_LiveRobotAgentHost([_LiveRobotAgent()]))
    asyncio.run(activate_selected_ra_context(adapter, tmp_path))
    runtime = _ProgramRuntime([_program_action([("grasp_part", {}), ("move_cartesian", {})])])
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    composition_input = _composition_input_from_prompt(runtime.calls[0]["prompt"])
    assert "task" not in composition_input
    target_feature = composition_input["target_feature"]
    assert [item["name"] for item in target_feature["desired_state"]["state_values"]] == [
        "assembly_board_shaft_location",
        "specified_finish",
        "specified_coating",
    ]
    resolved = {item["name"]: item for item in target_feature["resolved_state_values"]}
    assert resolved["specified_finish"]["resolved_value"] == "matte"
    assert resolved["specified_coating"]["resolved_value"] == "primer"
    assert "value_ref" not in json.dumps(resolved)
    serialized = json.dumps(candidate.to_record())
    assert "target_feature" not in serialized
    assert "specified_finish" not in serialized


def test_composition_diagnostic_blocks_mismatched_context_role(tmp_path: Path) -> None:
    _prepare_composition(tmp_path)
    runtime = _ProgramRuntime([_program_action([("grasp_part", {})])])
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    request_path = candidate.path.parent / "request.json"
    request = _read_json(request_path)
    request["context_refs"]["robot_state"] = request["context_refs"]["assignment"]
    _write_program_record(request_path, request)
    diagnostic = read_primitive_composition_diagnostic(tmp_path)
    assert diagnostic["status"] == "blocked"
    assert diagnostic["candidate"] is None
    assert "wrong role" in diagnostic["message"]
    assert len(runtime.calls) == 1


@pytest.mark.parametrize("unsupported", [True, False])
def test_composition_handles_unsupported_and_unknown_symbols(
    tmp_path: Path, unsupported: bool
) -> None:
    _prepare_composition(tmp_path)
    action = (
        {"kind": "unsupported", "reason": "The catalog has no insertion behavior."}
        if unsupported
        else _program_action([("invented_primitive", {})])
    )
    runtime = _ProgramRuntime([action])
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert candidate.record["status"] == ("unsupported" if unsupported else "invalid")
    diagnostic = read_primitive_composition_diagnostic(tmp_path)
    assert diagnostic["status"] == candidate.record["status"]
    assert isinstance(diagnostic["composition_input"], dict)
    assert not (tmp_path / "composition/primitive_program_drafts").exists()


def test_default_xarm6_robot_agent_owns_eight_synthesis_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cais_spade_llm import agent_creator
    from cais_spade_llm.resources.resource_primitives import build_primitive_reference_card

    monkeypatch.setattr(agent_creator, "ROBOT_ENV", "gazebo")
    monkeypatch.setattr(agent_creator, "_EXECUTION_MODE_OVERRIDE", "dry_run")
    monkeypatch.setitem(agent_creator.ALLOWED_FUNCS, "xarm6", set())
    agent = agent_creator.create_resource_agents(
        ["cais_spade_llm/initialization/resources/robot_xarm6.json"],
        "cais_spade_llm/initialization/cca.json",
    )[0]

    assert "pick_approach" in agent.executables
    assert "execute_recovery_macro" in agent.executables
    assert [binding["scenario_id"] for binding in agent.failure_scenarios] == ["lg_slippage"]
    assert [
        entry["name"] for entry in agent.recovery_synthesis_primitive_catalog()
    ] == _EXPECTED_XARM6_SYNTHESIS_SYMBOLS
    assert agent.get_recovery_snapshot()["current_pose"] == {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
    }
    recovery_catalog = agent.recovery_synthesis_primitive_catalog()
    entries = {entry["name"]: entry for entry in recovery_catalog}
    assert "current_state" in entries["grasp_part"]["effects"]
    pick = entries["compute_pick_targets"]
    assert "product_geometry" not in pick["required_params"]
    assert pick["params"]["product_geometry"]["properties"]["board_center"]["properties"]["z"]["type"] == "number"
    assert "insert_pose" in entries["compute_place_targets"]["output_schema"]
    card = build_primitive_reference_card(recovery_catalog)
    assert "product_geometry:object" in card
    assert "product_geometry:object required" not in card


@pytest.mark.parametrize(
    ("resource_name", "resource_path"),
    [
        (
            "xarm6",
            "cais_spade_llm/initialization/resources/robot_xarm6.json",
        ),
        (
            "ur5e",
            "cais_spade_llm/initialization/resources/robot_ur5e.json",
        ),
    ],
)
def test_context_only_robot_agent_has_no_tools_failures_or_controller(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    resource_name: str,
    resource_path: str,
) -> None:
    import logging

    from cais_spade_llm import agent_creator
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    def _forbidden_controller(_agent: RobotAgent) -> object:
        raise AssertionError("context-only RobotAgent constructed a controller")

    monkeypatch.setattr(agent_creator, "ROBOT_ENV", "gazebo")
    monkeypatch.setattr(agent_creator, "_EXECUTION_MODE_OVERRIDE", "simulation")
    monkeypatch.setattr(RobotAgent, "_build_controller", _forbidden_controller)
    monkeypatch.setitem(agent_creator.ALLOWED_FUNCS, resource_name, set())
    caplog.set_level("INFO", logger=f"agent:{resource_name}")
    # Agent handlers can bypass pytest's root capture handler. Capture directly
    # here so the lifecycle test also avoids writing application log files.
    monkeypatch.setattr(logging.getLogger(f"agent:{resource_name}"), "handlers", [caplog.handler])

    agent = agent_creator.create_resource_agents(
        [resource_path],
        "cais_spade_llm/initialization/cca.json",
        robot_context_only=True,
    )[0]

    assert agent.context_only is True
    assert agent.execution_mode == "simulation"
    assert agent.executables == {}
    assert agent.function_info == []
    assert agent.failure_scenarios == []
    assert agent._controller is None
    assert agent.enable_controller_prewarm is False
    assert agent_creator.ALLOWED_FUNCS[resource_name] == set()
    snapshot = agent.get_recovery_snapshot()
    assert snapshot["function_names"] == []
    assert snapshot["current_pose"] is None
    assert snapshot["current_pose_captured_at"] is None
    assert snapshot["controller_ready"] is False
    assert snapshot["perception_ready"] is False
    assert snapshot["tf_ready"] is False
    assert snapshot["tcp_ready"] is False
    synthesis_symbols = [entry["name"] for entry in agent.recovery_synthesis_primitive_catalog()]
    assert synthesis_symbols
    composer_catalog = in_process_robot_agent._phase_5_1_primitive_catalog(
        agent.recovery_synthesis_primitive_catalog()
    )
    context_handoff._validate_primitive_catalog(composer_catalog)
    assert all(
        "parameter_schemas" in entry and "result_schemas" in entry for entry in composer_catalog
    )
    if resource_name == "xarm6":
        assert synthesis_symbols == _EXPECTED_XARM6_SYNTHESIS_SYMBOLS
    messages = [
        record.getMessage() for record in caplog.records if record.name == f"agent:{resource_name}"
    ]
    assert any("profile=context_only tools=[]" in message for message in messages)
    assert not any("failure_scenarios" in message for message in messages)
    assert not any("pick_approach" in message for message in messages)
    assert not any("lg_slippage" in message for message in messages)


def test_in_process_ra_adapter_starts_only_selected_robot_agent_from_gazebo(
    tmp_path: Path,
) -> None:
    persist_native_completion_fixture(tmp_path)
    started_agent = _LiveRobotAgent()
    host = _LiveRobotAgentHost(
        [],
        system_running=False,
        gazebo_state="running",
        startup_agent=started_agent,
    )
    runtime = InProcessRobotAgentCompositionRuntime(host)

    captured = asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert host.start_calls == 1
    assert host.full_system_start_calls == 0
    assert host.readiness_calls == 1
    assert host.readiness_forces == [True]
    assert host.system_running is False
    assert host.resource_agents == []
    assert host.execution_mode == "simulation"
    assert host.robot_env == "gazebo"
    assert captured.robot_state.robot_state["resource_jid"] == "xarm6@localhost"
    assert read_phase_5_1_diagnostic(tmp_path).status == "context_captured"


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("gazebo_stopped", "Dual Gazebo Environment is not running"),
        ("startup_failed", "XMPP startup failed"),
        ("agent_missing", "is not running"),
    ],
)
def test_in_process_ra_adapter_startup_failures_preserve_phase_4(
    tmp_path: Path,
    variant: str,
    message: str,
) -> None:
    persist_native_completion_fixture(tmp_path)
    completion_path = tmp_path / "interaction_record/context_completion_0001.json"
    completion_bytes = completion_path.read_bytes()
    host = _LiveRobotAgentHost(
        [],
        system_running=False,
        gazebo_state="stopped" if variant == "gazebo_stopped" else "running",
        startup_agent=None,
        startup_error="XMPP startup failed" if variant == "startup_failed" else None,
    )
    runtime = InProcessRobotAgentCompositionRuntime(host)

    with pytest.raises(RAContextHandoffError, match=message):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert completion_path.read_bytes() == completion_bytes
    resource_root = tmp_path / "resources/xarm6@localhost"
    assert not (resource_root / "validation").exists()
    assert not (resource_root / "robot_state").exists()
    assert not (resource_root / "primitive_catalog_snapshot").exists()
    assert read_phase_5_1_diagnostic(tmp_path).status == "waiting_for_ra"
    assert host.start_calls == (0 if variant == "gazebo_stopped" else 1)
    assert host.full_system_start_calls == 0


def test_in_process_ra_adapter_times_out_waiting_for_spec2primitives_gazebo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persist_native_completion_fixture(tmp_path)
    completion_path = tmp_path / "interaction_record/context_completion_0001.json"
    completion_bytes = completion_path.read_bytes()
    host = _LiveRobotAgentHost(
        [],
        system_running=False,
        gazebo_state="running",
        readiness=(False, "Waiting for ROS services."),
    )
    monkeypatch.setattr(
        in_process_robot_agent,
        "_ROBOT_AGENT_STARTUP_TIMEOUT_SECONDS",
        0.0,
    )
    runtime = InProcessRobotAgentCompositionRuntime(host)

    with pytest.raises(RAContextHandoffError, match="did not become ready"):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert host.start_calls == 0
    assert completion_path.read_bytes() == completion_bytes
    _assert_no_ra_context_snapshots(tmp_path)


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("different_jid", "is not running"),
        ("offline", "is not running"),
        ("duplicate", "is not unique"),
        ("execution_mode", "execution_mode"),
    ],
)
def test_in_process_ra_adapter_fails_closed_without_exact_compatible_live_agent(
    tmp_path: Path,
    variant: str,
    message: str,
) -> None:
    persist_native_completion_fixture(tmp_path)
    completion_path = tmp_path / "interaction_record/context_completion_0001.json"
    completion_bytes = completion_path.read_bytes()
    if variant == "different_jid":
        agents = [_LiveRobotAgent(jid="ur5e@localhost")]
    elif variant == "offline":
        agents = [_LiveRobotAgent(alive=False)]
    elif variant == "duplicate":
        agents = [_LiveRobotAgent(), _LiveRobotAgent()]
    else:
        agents = [_LiveRobotAgent(execution_mode="physical")]
    runtime = InProcessRobotAgentCompositionRuntime(_LiveRobotAgentHost(agents))

    with pytest.raises(RAContextHandoffError, match=message):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert completion_path.read_bytes() == completion_bytes
    assert (tmp_path / "composition/selected_ra_assignments/assignment_0001.json").is_file()
    _assert_no_ra_context_snapshots(tmp_path)
    assert read_phase_5_1_diagnostic(tmp_path).status == "waiting_for_ra"


@pytest.mark.parametrize(
    "variant",
    [
        "composite",
        "composite_expansion",
        "hidden",
        "untyped",
        "duplicate",
        "empty_catalog",
        "empty_state",
    ],
)
def test_in_process_ra_adapter_rejects_malformed_context(
    tmp_path: Path,
    variant: str,
) -> None:
    persist_native_completion_fixture(tmp_path)
    catalog = _raw_live_catalog()
    if variant == "composite":
        catalog[0]["primitive_steps"] = []
    elif variant == "composite_expansion":
        catalog[0]["composite_expansion"] = []
    elif variant == "hidden":
        catalog[0]["synthesis_hidden"] = True
    elif variant == "untyped":
        catalog[0]["params"] = {"part_name": {"description": "part"}}
    elif variant == "duplicate":
        duplicate = deepcopy(catalog[0])
        catalog.insert(1, duplicate)
    elif variant == "empty_catalog":
        catalog = []
    robot_state = {} if variant == "empty_state" else None
    runtime = InProcessRobotAgentCompositionRuntime(
        _LiveRobotAgentHost(
            [
                _LiveRobotAgent(
                    robot_state=robot_state,
                    primitive_catalog=catalog,
                )
            ]
        )
    )

    with pytest.raises(RAContextHandoffError):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    _assert_no_ra_context_snapshots(tmp_path)


def test_in_process_ra_adapter_retries_same_phase_4_assignment(
    tmp_path: Path,
) -> None:
    persist_native_completion_fixture(tmp_path)
    completion_path = tmp_path / "interaction_record/context_completion_0001.json"
    completion_bytes = completion_path.read_bytes()
    agent = _LiveRobotAgent(alive=False)
    runtime = InProcessRobotAgentCompositionRuntime(_LiveRobotAgentHost([agent]))

    with pytest.raises(RAContextHandoffError, match="is not running"):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))
    assignment_bytes = (
        tmp_path / "composition/selected_ra_assignments/assignment_0001.json"
    ).read_bytes()

    agent.alive = True
    captured = asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert captured.assignment_path.read_bytes() == assignment_bytes
    assert completion_path.read_bytes() == completion_bytes
    assert read_phase_5_1_diagnostic(tmp_path).status == "context_captured"


def test_phase_5_1_diagnostic_waits_for_phase_4_then_reports_selected_ra(
    tmp_path: Path,
) -> None:
    waiting = read_phase_5_1_diagnostic(tmp_path).to_view()

    assert waiting["status"] == "waiting_for_phase_4"
    assert waiting["selected_resource_jid"] is None
    assert waiting["primitive_catalog"] == []

    persist_native_completion_fixture(tmp_path)
    ready = read_phase_5_1_diagnostic(tmp_path).to_view()

    assert ready["status"] == "ready_for_assignment"
    assert ready["product_requirement"] == "assemble medium gear"
    assert ready["selected_resource_jid"] == "xarm6@localhost"
    assert ready["selected_execution_mode"] == "simulation"
    assert ready["assignment_ref"] is None
    assert ready["primitive_count"] == 0


def test_phase_5_1_rejects_audit_only_completion_v7_before_assignment(
    tmp_path: Path,
) -> None:
    persist_native_completion_fixture(tmp_path)
    completion_path = next((tmp_path / "interaction_record").glob("context_completion_*.json"))
    completion = _read_json(completion_path)
    proposal_path = tmp_path / str(completion["ontology_projection_ref"])
    proposal = _read_json(proposal_path)
    proposal["schema_version"] = 9
    proposal["fingerprint"] = _record_fingerprint(proposal)
    _write_json(proposal_path, proposal)
    completion["schema_version"] = 7
    completion["ontology_projection_sha256"] = hashlib.sha256(
        proposal_path.read_bytes()
    ).hexdigest()
    completion["fingerprint"] = _record_fingerprint(completion)
    _write_json(completion_path, completion)

    with pytest.raises(RAContextHandoffError, match="Start a fresh interaction"):
        asyncio.run(activate_selected_ra_context(_AssignedContextRuntime(), tmp_path))

    assert not (tmp_path / "composition/selected_ra_assignments").exists()


def test_phase_5_1_rejects_ur5e_response_after_reassignment(tmp_path: Path) -> None:
    from cais_spade_llm.spec2primitives.agents.pa.resource_reassignment import (
        reassign_completed_interaction,
    )
    from cais_spade_llm.spec2primitives.tests.test_pa_completion import (
        _reassignment_runtime,
        _ReassignmentAgent,
    )

    persist_native_completion_fixture(tmp_path, resource_symbol="ur5e")

    class DelayedUR5eContext(_AssignedContextRuntime):
        async def request_assigned_context(self, assignment):
            response = await super().request_assigned_context(assignment)
            await reassign_completed_interaction(
                interaction_root=tmp_path,
                runtime=_reassignment_runtime(),
                product_agent=_ReassignmentAgent(),
                requested_resource_symbol="xarm6",
            )
            return response

    with pytest.raises(RAContextHandoffError, match="assignment changed during"):
        asyncio.run(
            activate_selected_ra_context(
                DelayedUR5eContext(self_jid="ur5e@localhost"),
                tmp_path,
            )
        )
    assert not (tmp_path / "resources/ur5e@localhost").exists()
    diagnostic = read_phase_5_1_diagnostic(tmp_path)
    assert diagnostic.status == "ready_for_assignment"
    assert diagnostic.selected_resource_jid == "xarm6@localhost"


def test_phase_5_1_dispatches_assignment_and_appends_paired_snapshots(
    tmp_path: Path,
) -> None:
    persist_native_completion_fixture(tmp_path)
    runtime = _AssignedContextRuntime()

    first = asyncio.run(activate_selected_ra_context(runtime, tmp_path))
    first_state_bytes = first.robot_state_path.read_bytes()
    first_catalog_bytes = first.primitive_catalog_path.read_bytes()
    second = asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert len(runtime.assignments) == 2
    assert runtime.assignments[0] == runtime.assignments[1]
    assert first.assignment.product_requirement == "assemble medium gear"
    assert first.assignment.selected_resource_jid == "xarm6@localhost"
    assert runtime.assignments[0].fingerprint == first.assignment.fingerprint
    assert first.assignment_path.relative_to(tmp_path).as_posix() == (
        "composition/selected_ra_assignments/assignment_0001.json"
    )
    assert first.robot_state_path.name == "snapshot_0001.json"
    assert first.primitive_catalog_path.name == "snapshot_0001.json"
    assert second.robot_state_path.name == "snapshot_0002.json"
    assert second.primitive_catalog_path.name == "snapshot_0002.json"
    assert first.robot_state_path.read_bytes() == first_state_bytes
    assert first.primitive_catalog_path.read_bytes() == first_catalog_bytes
    assert [entry["primitive_symbol"] for entry in first.primitive_catalog.primitive_catalog] == [
        "detect_parts",
        "move_pose",
    ]

    assignment_record = _read_json(first.assignment_path)
    catalog_record = _read_json(first.primitive_catalog_path)
    assert "schema_version" not in assignment_record
    assert assignment_record["process_symbol"] == "assembly"
    assert assignment_record["selected_resource_symbol"] == "xarm6"
    assert assignment_record["allocation_label"] == ("resource assignment validated by MoveIt")
    assert assignment_record["validation_scope"] == "moveit_state_location_reachability"
    assert assignment_record["motion_executed"] is False
    assert assignment_record["current_state_evidence"]["location_handles"]
    assert assignment_record["desired_state_evidence"]["location_handles"]
    assert assignment_record["registry_snapshot_ref"].endswith(
        "resource_registry_snapshot_0001.json"
    )
    assert assignment_record["workcell_snapshot_ref"].endswith(
        "predefined_workcell_snapshot_0001.json"
    )
    assert assignment_record["evidence_presentation_ref"].endswith(
        "evidence_presentation_record.json"
    )
    assert assignment_record["allocation_presentation_ref"].endswith(
        "allocation_presentation_record.json"
    )
    assert "typed_context_refs" not in assignment_record
    assert "ontology_projection_ref" not in assignment_record
    assert "composition_input" not in assignment_record
    assert "ontology_projection" not in assignment_record
    assert "assertions" not in assignment_record
    assert catalog_record["assignment_fingerprint"] == first.assignment.fingerprint
    assert (
        catalog_record["robot_state_ref"] == first.robot_state_path.relative_to(tmp_path).as_posix()
    )
    assert catalog_record["robot_state_fingerprint"] == first.robot_state.fingerprint
    assert not (tmp_path / "composition/context_bundles").exists()
    assert not (tmp_path / "resources/xarm6@localhost/primitive_program_drafts").exists()
    assert not (tmp_path / "composition/missing_context_batches").exists()
    assert not (tmp_path / "resources/xarm6@localhost/primitive_steps").exists()
    validation_records = list(
        (tmp_path / "resources/xarm6@localhost/validation").glob(
            "*/plan_only_feasibility_validation_record.json"
        )
    )
    assert validation_records == []
    assert assignment_record["motion_validation_performed"] is True

    diagnostic = read_phase_5_1_diagnostic(tmp_path).to_view()
    assert diagnostic["status"] == "context_captured"
    assert diagnostic["assignment_ref"] == (
        "composition/selected_ra_assignments/assignment_0001.json"
    )
    assert diagnostic["state_snapshot_count"] == 2
    assert diagnostic["catalog_snapshot_count"] == 2
    assert diagnostic["latest_state_ref"].endswith("snapshot_0002.json")
    assert diagnostic["latest_catalog_ref"].endswith("snapshot_0002.json")
    assert diagnostic["robot_state"]["current_state"] == "idle"
    assert diagnostic["primitive_symbols"] == ["detect_parts", "move_pose"]
    assert diagnostic["primitive_count"] == 2
    assert diagnostic["catalog_fingerprint"] == (second.primitive_catalog.catalog_fingerprint)


def test_phase_5_assignment_rejects_retired_cartesian_contract(
    tmp_path: Path,
) -> None:
    persist_native_completion_fixture(tmp_path)
    capture = asyncio.run(activate_selected_ra_context(_AssignedContextRuntime(), tmp_path))
    assignment = _read_json(capture.assignment_path)
    # Exercise the historical schema reader without permitting historical RA activation.
    assignment["schema_version"] = 3
    assignment.pop("motion_validation_performed")
    assignment.update(
        robot_agent_validation_ref="resources/xarm6@localhost/validation/historical.json",
        robot_agent_validation_sha256="a" * 64,
        robot_agent_validation_fingerprint="b" * 64,
    )
    assignment["allocation_label"] = "validated Cartesian pick-place allocation"
    assignment["validation_scope"] = "cartesian_pick_place"
    assignment["checked_constraints"] = [
        "live_tf",
        "collision_aware_cartesian_pick_path",
        "collision_aware_cartesian_transfer_place_path",
        "complete_path_fraction",
    ]
    assignment["unvalidated_constraints"] = [
        "grasp_contact",
        "gripper_actuation",
        "attached_part_collision_geometry",
        "assembly_tolerance",
        "force_control",
        "final_constrained_insertion_stroke",
    ]
    assignment["fingerprint"] = _record_fingerprint(assignment)

    with pytest.raises(RAContextHandoffError, match="Start a fresh interaction"):
        context_handoff._assignment_from_mapping(assignment)


def test_phase_5_1_restart_preserves_legacy_catalog_and_exposes_latest_synthesis(
    tmp_path: Path,
) -> None:
    persist_native_completion_fixture(tmp_path)
    completion_path = tmp_path / "interaction_record/context_completion_0001.json"
    completion_bytes = completion_path.read_bytes()

    def _catalog(symbols: list[str]) -> list[dict[str, object]]:
        template = _valid_catalog()[0]
        return [
            {
                **deepcopy(template),
                "primitive_symbol": symbol,
                "invocation_binding": symbol,
            }
            for symbol in symbols
        ]

    legacy_runtime = _AssignedContextRuntime(
        transform=lambda response: response.update(
            {"primitive_catalog": _catalog([f"legacy_{index}" for index in range(16)])}
        )
    )
    first = asyncio.run(activate_selected_ra_context(legacy_runtime, tmp_path))
    first_catalog_bytes = first.primitive_catalog_path.read_bytes()
    current_runtime = _AssignedContextRuntime(
        transform=lambda response: response.update(
            {"primitive_catalog": _catalog(_EXPECTED_XARM6_SYNTHESIS_SYMBOLS)}
        )
    )

    second = asyncio.run(activate_selected_ra_context(current_runtime, tmp_path))
    diagnostic = read_phase_5_1_diagnostic(tmp_path).to_view()

    assert first.primitive_catalog_path.read_bytes() == first_catalog_bytes
    assert first.primitive_catalog_path.name == "snapshot_0001.json"
    assert second.primitive_catalog_path.name == "snapshot_0002.json"
    assert completion_path.read_bytes() == completion_bytes
    assert diagnostic["catalog_snapshot_count"] == 2
    assert diagnostic["primitive_symbols"] == _EXPECTED_XARM6_SYNTHESIS_SYMBOLS


def test_changed_phase_4_selection_prevents_ra_dispatch(tmp_path: Path) -> None:
    completion = persist_native_completion_fixture(tmp_path).to_record()
    selection_path = tmp_path / str(completion["resource_selection_ref"])
    selection = _read_json(selection_path)
    selection["selected_resource_jid"] = "ur5e@localhost"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    runtime = _AssignedContextRuntime()

    with pytest.raises(
        RAContextHandoffError,
        match="unchanged PAContextGroundingCompletion",
    ):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert runtime.assignments == []
    assert not (tmp_path / "composition/selected_ra_assignments").exists()


@pytest.mark.parametrize(
    "variant",
    [
        "process",
        "state",
        "reachability",
        "validation",
        "registry",
        "workcell",
        "evidence_presentation",
        "allocation_presentation",
    ],
)
def test_assignment_envelope_rejects_altered_authority_lineage(
    tmp_path: Path,
    variant: str,
) -> None:
    persist_native_completion_fixture(tmp_path)
    context = asyncio.run(activate_selected_ra_context(_AssignedContextRuntime(), tmp_path))
    assignment = _read_json(context.assignment_path)
    if variant == "process":
        assignment["process_symbol"] = "altered_process"
    elif variant == "state":
        assignment["current_state_evidence"]["location_handles"] = ["altered_evidence"]
    else:
        fingerprint_field = {
            "reachability": "reachability_check_fingerprint",
            "validation": "robot_agent_validation_fingerprint",
            "registry": "registry_snapshot_fingerprint",
            "workcell": "workcell_snapshot_fingerprint",
            "evidence_presentation": "evidence_presentation_fingerprint",
            "allocation_presentation": "allocation_presentation_fingerprint",
        }[variant]
        assignment[fingerprint_field] = "0" * 64
    assignment["fingerprint"] = _record_fingerprint(assignment)
    _write_json(context.assignment_path, assignment)

    diagnostic = read_phase_5_1_diagnostic(tmp_path)

    assert diagnostic.status == "blocked"
    assert diagnostic.failure


def test_assignment_addressed_to_another_ra_is_rejected(tmp_path: Path) -> None:
    persist_native_completion_fixture(tmp_path)
    runtime = _AssignedContextRuntime(self_jid="ur5e@localhost")

    with pytest.raises(RAContextHandoffError, match="addressed to a different RA"):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert runtime.assignments == []
    assert (tmp_path / "composition/selected_ra_assignments/assignment_0001.json").is_file()
    _assert_no_ra_context_snapshots(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("resource_jid", "ur5e@localhost", "response JID"),
        ("assignment_fingerprint", "0" * 64, "assignment fingerprint"),
    ],
)
def test_response_must_match_assignment(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    persist_native_completion_fixture(tmp_path)

    def transform(response: dict[str, object]) -> None:
        response[field] = value

    runtime = _AssignedContextRuntime(transform=transform)
    with pytest.raises(RAContextHandoffError, match=message):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert len(runtime.assignments) == 1
    _assert_no_ra_context_snapshots(tmp_path)


@pytest.mark.parametrize(
    "variant",
    ["empty", "duplicate", "composite", "non_finite"],
)
def test_invalid_ra_context_is_not_persisted(tmp_path: Path, variant: str) -> None:
    persist_native_completion_fixture(tmp_path)

    def transform(response: dict[str, object]) -> None:
        if variant == "empty":
            response["primitive_catalog"] = []
        elif variant == "duplicate":
            catalog = _valid_catalog()
            catalog[1]["primitive_symbol"] = catalog[0]["primitive_symbol"]
            response["primitive_catalog"] = catalog
        elif variant == "composite":
            catalog = _valid_catalog()
            catalog[0]["primitive_steps"] = []
            response["primitive_catalog"] = catalog
        else:
            state = deepcopy(response["robot_state"])
            state["position"]["x"] = float("nan")
            response["robot_state"] = state

    runtime = _AssignedContextRuntime(transform=transform)
    with pytest.raises(RAContextHandoffError):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert len(runtime.assignments) == 1
    assert (tmp_path / "composition/selected_ra_assignments/assignment_0001.json").is_file()
    _assert_no_ra_context_snapshots(tmp_path)


def test_unpaired_snapshot_prevents_ra_dispatch(tmp_path: Path) -> None:
    persist_native_completion_fixture(tmp_path)
    state_path = tmp_path / "resources/xarm6@localhost/robot_state/snapshot_0001.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{}\n", encoding="utf-8")
    runtime = _AssignedContextRuntime()

    with pytest.raises(RAContextHandoffError, match="revisions are unpaired"):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert runtime.assignments == []
    diagnostic = read_phase_5_1_diagnostic(tmp_path)
    assert diagnostic.status == "blocked"
    assert diagnostic.failure is not None
    assert "revisions are unpaired" in diagnostic.failure


def test_runtime_failure_leaves_only_assignment_audit(tmp_path: Path) -> None:
    persist_native_completion_fixture(tmp_path)
    runtime = _AssignedContextRuntime(failure=RuntimeError("RA unavailable"))

    with pytest.raises(RuntimeError, match="RA unavailable"):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert len(runtime.assignments) == 1
    assert (tmp_path / "composition/selected_ra_assignments/assignment_0001.json").is_file()
    _assert_no_ra_context_snapshots(tmp_path)
    diagnostic = read_phase_5_1_diagnostic(tmp_path)
    assert diagnostic.status == "waiting_for_ra"
    assert diagnostic.assignment_ref == ("composition/selected_ra_assignments/assignment_0001.json")


def test_composition_requires_complete_catalog_declarations_before_ra_call(tmp_path: Path) -> None:
    persist_native_completion_fixture(tmp_path)
    asyncio.run(activate_selected_ra_context(_AssignedContextRuntime(), tmp_path))
    runtime = _ProgramRuntime([_program_action([("move_pose", {})])])
    with pytest.raises(PrimitiveCompositionError, match="Capture a fresh RobotAgent context"):
        asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert runtime.calls == []
    assert not (tmp_path / "composition/primitive_program_candidates").exists()


def test_catalog_declarations_cannot_disagree_with_typed_snapshot(tmp_path: Path) -> None:
    persist_native_completion_fixture(tmp_path)

    def change_schema(response: dict[str, Any]) -> None:
        response["primitive_catalog"][0]["parameter_schemas"] = {"part_name": {"type": "number"}}

    with pytest.raises(RAContextHandoffError, match="type disagrees"):
        asyncio.run(
            activate_selected_ra_context(_AssignedContextRuntime(transform=change_schema), tmp_path)
        )

    _assert_no_ra_context_snapshots(tmp_path)


def test_composition_recapture_uses_new_context_and_preserves_old_candidate(
    tmp_path: Path,
) -> None:
    adapter, _, _ = _prepare_composition(tmp_path)
    runtime = _ProgramRuntime([_program_action([("release_part", {})])])
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    before = candidate.path.read_bytes()
    asyncio.run(activate_selected_ra_context(adapter, tmp_path))

    view = read_primitive_composition_diagnostic(tmp_path)
    assert view["status"] == "ready_for_composition"
    assert view["attempt_count"] == 0
    assert candidate.path.read_bytes() == before


def _prepare_composition(
    tmp_path: Path,
) -> tuple[InProcessRobotAgentCompositionRuntime, _LiveRobotAgent, str]:
    persist_native_completion_fixture(tmp_path)
    catalog = _raw_live_catalog()
    catalog[0]["params"]["target_pose"] = {
        "type": "object",
        "description": "Observed target in metres.",
    }
    catalog[5]["params"]["dz"]["description"] = "Relative displacement in metres."
    agent = _LiveRobotAgent(primitive_catalog=catalog)
    runtime = InProcessRobotAgentCompositionRuntime(_LiveRobotAgentHost([agent]), contexts_root=tmp_path)
    asyncio.run(activate_selected_ra_context(runtime, tmp_path))
    view = read_primitive_composition_diagnostic(tmp_path)
    current_ref = view["composition_input"]["target_feature"]["current_state"]["state_values"][0][
        "value_ref"
    ]["record_ref"]
    return runtime, agent, current_ref


def _program_action(steps: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "kind": "propose",
        "primitive_steps": [
            {"primitive_symbol": symbol, "params": json.dumps(params)} for symbol, params in steps
        ],
    }


def test_gazebo_execution_reads_selected_context_only_configuration_and_interlocks(
    tmp_path: Path,
) -> None:
    runtime, selected, _ = _prepare_composition(tmp_path)
    assignment = primitive_composition._load_inputs(tmp_path).assignment
    selected.context_only = True
    selected.controller_config = {"gripper": {"joint": "configured_joint"}}
    host = runtime._host
    host.system_running = False
    host.gazebo_state = "running"
    hardware = {"overall": "stopped"}
    host.hardware_stack_status = lambda robot: hardware
    authority = asyncio.run(runtime.execution_configuration(assignment))
    assert authority["configuration"] == selected.controller_config
    assert authority["primitive_catalog"] == in_process_robot_agent._phase_5_1_primitive_catalog(
        selected.primitive_catalog
    )
    assert host.start_calls == host.full_system_start_calls == 0
    assert selected.context_only is True
    assert not hasattr(selected, "controller")
    assert host.readiness_forces[-1] is True
    hardware["overall"] = "running"
    with pytest.raises(RAContextHandoffError, match="Hardware stack"):
        asyncio.run(runtime.execution_configuration(assignment))
    hardware["overall"] = "stopped"
    selected.context_only = False
    with pytest.raises(RAContextHandoffError, match="context-only"):
        asyncio.run(runtime.execution_configuration(assignment))


@pytest.mark.parametrize("agent_state", ["shared", "standalone", "missing", "stopped"])
@pytest.mark.parametrize("start_if_needed", [False, True])
def test_gazebo_execution_reads_configuration_without_startup_or_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_state: str, start_if_needed: bool,
) -> None:
    runtime, selected, _ = _prepare_composition(tmp_path)
    assignment = primitive_composition._load_inputs(tmp_path).assignment
    selected.context_only = True
    selected.controller_config = {"gripper": {"joint": "configured_joint"}}
    host = runtime._host
    host.system_running = False
    host.gazebo_state = "running"
    host.hardware_stack_status = lambda robot: {"overall": "stopped"}
    if agent_state != "shared":
        host.resource_agents = []
    if agent_state in {"standalone", "stopped"}:
        host._spec2primitives_robot_agent = selected
    selected.alive = agent_state != "stopped"
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Loading execution configuration needs no agent startup or ROS readiness probe.")
    monkeypatch.setattr(host, "simulation_start_ready", forbidden)
    monkeypatch.setattr(host, "start_spec2primitives_robot_agent", forbidden)
    monkeypatch.setattr(host, "_run_on_agent_runtime", forbidden)
    result = asyncio.run(runtime.execution_configuration(assignment, start_if_needed=start_if_needed))
    if agent_state == "missing":
        manifest = Path(__file__).resolve().parents[2] / "initialization/resources/robot_xarm6.json"
        expected = json.loads(manifest.read_text())["xarm6"]["gazebo"]["controller"]
        assert result["configuration"] == expected
    else:
        assert result["configuration"] == selected.controller_config
    assert host.start_calls == host.full_system_start_calls == 0


@pytest.mark.parametrize("fault, message", [
    ("hardware", "Hardware stack"),
    ("environment", "not exclusively available"),
    ("execution_mode", "not exclusively available"),
    ("system_running", "not exclusively available"),
    ("gazebo_stopped", "Environment is not running"),
    ("agent_ambiguous", "is not unique"),
    ("agent_mode", "execution_mode does not match"),
    ("context_only", "context-only"),
])
def test_gazebo_execution_configuration_keeps_simulation_ownership(
    tmp_path: Path, fault: str, message: str,
) -> None:
    runtime, selected, _ = _prepare_composition(tmp_path)
    assignment = primitive_composition._load_inputs(tmp_path).assignment
    selected.context_only = fault != "context_only"
    selected.controller_config = {"gripper": {"joint": "configured_joint"}}
    host = runtime._host
    host.system_running = fault == "system_running"
    host.gazebo_state = "stopped" if fault == "gazebo_stopped" else "running"
    host.hardware_stack_status = lambda robot: {"overall": "running" if fault == "hardware" else "stopped"}
    if fault == "environment":
        host.robot_env = "real"
    if fault == "execution_mode":
        host.execution_mode = "physical"
    if fault == "agent_mode":
        selected.execution_mode = "physical"
    if fault == "agent_ambiguous":
        host.resource_agents.append(_LiveRobotAgent(jid=selected.jid))
    with pytest.raises(RAContextHandoffError, match=message):
        asyncio.run(runtime.execution_configuration(assignment))
    assert host.start_calls == host.full_system_start_calls == 0


@pytest.mark.parametrize("known", [True, False])
@pytest.mark.parametrize("new_launch", [False, True])
def test_gazebo_execution_custody_is_used_by_later_context_capture(
    tmp_path: Path,
    known: bool,
    new_launch: bool,
) -> None:
    from cais_spade_llm.spec2primitives.agents.ra.refinement_records import append_record

    root = tmp_path / "interaction"
    runtime, selected, _ = _prepare_composition(root)
    runtime = InProcessRobotAgentCompositionRuntime(runtime._host, contexts_root=tmp_path)
    if new_launch:
        runtime._host._ros2_procs = {DUAL_GAZEBO_NAME: SimpleNamespace(pid=123, poll=lambda: None)}
        runtime._host._gazebo_launch_timing_snapshot = lambda: {
            "name": DUAL_GAZEBO_NAME, "pid": 123, "t0": 1.0,
        }
        selected.robot_state["held_part"] = "medium gear"
    assignment = primitive_composition._load_inputs(root).assignment
    directory = root / "execution/run_1"
    request_ref = append_record(
        root,
        directory,
        "request.json",
        {
            "record_type": "PrimitiveExecutionRequest",
            "resource_jid": assignment.selected_resource_jid,
            "total_steps": 1,
            "candidate_ref": {},
            "created_at_ns": 1,
        },
    )
    append_record(
        root,
        directory,
        "result.json",
        {
            "record_type": "PrimitiveExecutionResult",
            "request_ref": request_ref,
            "status": "stopped",
            "message": "Stopped after grasp.",
            "record_refs": [],
            "last_event_ref": None,
            "custody_known": known,
            "held_part": "medium gear",
            "gripper_state": "closed",
        },
    )
    if not known and not new_launch:
        with pytest.raises(ValueError, match="uncertain"):
            asyncio.run(runtime.request_assigned_context(assignment))
        return
    response = asyncio.run(runtime.request_assigned_context(assignment))
    assert response["robot_state"]["held_part"] == (None if new_launch else "medium gear")
    assert response["robot_state"]["gripper_state"] == (None if new_launch else "closed")
    assert "model_name" not in json.dumps(response["robot_state"])
    assert selected.robot_state.get("held_part") == ("medium gear" if new_launch else None)


def test_composition_proposes_with_unbound_parameters_and_ignores_draft_history(
    tmp_path: Path,
) -> None:
    """Old drafts cannot seed the program or prevent a proposal with missing geometry."""
    _prepare_composition(tmp_path)
    draft = tmp_path / "composition/primitive_program_drafts/draft_0001.json"
    draft.parent.mkdir()
    draft.write_text("Historical draft must never reach the composer.")
    old_attempt = tmp_path / "composition/primitive_program_candidates/attempt_0001"
    old_attempt.mkdir(parents=True)
    _write_program_record(
        old_attempt / "request.json",
        {
            "record_type": "PrimitiveCompositionRequest",
            "draft_ref": draft.relative_to(tmp_path).as_posix(),
            "draft_sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
            "prompt": "Historical program must never reach the composer.",
        },
    )
    (old_attempt / "candidate.json").write_text("Immutable historical candidate.")
    preserved = {path: path.read_bytes() for path in (draft, *old_attempt.iterdir())}
    steps = [("grasp_part", {}), ("move_cartesian", {"x": 0.2}), ("release_part", {})]
    runtime = _ProgramRuntime([_program_action(steps)])

    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))

    assert candidate.record["status"] == "proposed"
    assert candidate.path.parent.name == "attempt_0002"
    assert candidate.record["primitive_steps"] == [
        {"primitive_symbol": symbol, "params": params} for symbol, params in steps
    ]
    assert len(runtime.calls) == 1
    assert all("Historical" not in call["prompt"] for call in runtime.calls)
    assert {path: path.read_bytes() for path in preserved} == preserved
    diagnostic = read_primitive_composition_diagnostic(tmp_path)
    assert diagnostic["status"] == "proposed"
    assert diagnostic["attempt_count"] == 1
    assert "Omit unavailable measured inputs" in runtime.calls[0]["prompt"]
    assert "including required inputs" in runtime.calls[0]["prompt"]
    variants = runtime.calls[0]["response_format"]["schema"]["properties"]["action"]["anyOf"]
    assert ["missing_context"] not in [item["properties"]["kind"]["enum"] for item in variants]


def test_composition_keeps_event_loop_responsive_during_evidence_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slow evidence I/O permits connection heartbeats between RA decisions."""
    _, _, current_ref = _prepare_composition(tmp_path)
    runtime = _ProgramRuntime([_program_action([("move_cartesian", {})])])
    blocked: list[str] = []

    async def scenario() -> None:
        loop = asyncio.get_running_loop()

        def slow_check(reader: Callable) -> Callable:
            def read(*args: Any, **kwargs: Any) -> Any:
                heartbeat = Event()
                loop.call_soon_threadsafe(heartbeat.set)
                if not heartbeat.wait(timeout=0.5):
                    blocked.append(reader.__name__)
                return reader(*args, **kwargs)

            return read

        with monkeypatch.context() as patch:
            for name in (
                "_load_inputs",
                "_read_composition_history",
                "_validate_steps",
            ):
                patch.setattr(
                    primitive_composition, name, slow_check(getattr(primitive_composition, name))
                )
            candidate = await author_primitive_program_candidate(runtime, tmp_path)
        assert candidate.record["status"] == "proposed"

    asyncio.run(scenario())
    assert len(runtime.calls) == 1
    assert blocked == [], f"Evidence checks blocked connection heartbeats: {blocked}"


def test_composition_preserves_ra_decisions_in_one_bounded_prompt(tmp_path: Path) -> None:
    _, agent, current_ref = _prepare_composition(tmp_path)
    inputs = primitive_composition._load_inputs(tmp_path)
    location = {"value_ref": {"record_ref": current_ref, "field_path": "/translated_location_m/0"}}
    steps = [("compute_pick_targets", {"part_name": "medium gear", "target_pose": {"x": location}}),
             ("move_cartesian", {"x": location, "y": {"result_ref": {"step_index": 1, "field_path": "/target_pose/y"}}, "z": 1.2}),
             ("move_cartesian", {"x": 0, "y": 0, "z": 1.27})]
    runtime = _ProgramRuntime([_program_action(steps)])
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert candidate.record["status"] == "proposed"
    assert candidate.record["primitive_steps"] == [{"primitive_symbol": symbol, "params": params} for symbol, params in steps]
    assert len(runtime.calls) == len(candidate.record["exchange_refs"]) == 1 and agent.composition_calls == []
    prompt = runtime.calls[0]["prompt"]
    delivered = _composition_input_from_prompt(prompt)
    assert delivered["primitive_catalog"] == inputs.composition_input["primitive_catalog"]
    assert delivered["robot_state"] == inputs.composition_input["robot_state"]
    assert delivered["target_feature"]["product_requirement"] == "assemble medium gear"
    assert len(prompt) <= 32000 and prompt.count('"primitive_catalog":') == 1
    example, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{"result_ref":'):])
    assert primitive_composition._schema(primitive_composition._result_schema(
        candidate.record["primitive_steps"][:1], example["result_ref"], inputs,
    ))["type"] == "number"
    assert "/declared_output" not in prompt
    assert not {"ontology_projection", "grounded_context"} & delivered.keys()
    assert all(word not in prompt for word in ("EXCHANGES", "observation_catalog", "CADSizeCorrespondenceRecord"))
    kinds = {variant["properties"]["kind"]["enum"][0] for variant in runtime.calls[0]["response_format"]["schema"]["properties"]["action"]["anyOf"]}
    assert kinds == {"propose", "unsupported"}


def test_composition_progress_precedes_its_single_authoring_call(tmp_path: Path) -> None:
    _prepare_composition(tmp_path)
    updates = []
    def inspect_call() -> None:
        assert "RA is authoring the primitive program" in updates[-1]
        assert not list((tmp_path / "composition/primitive_program_candidates").glob("*/candidate.json"))
    runtime = _ProgramRuntime([_program_action([("move_cartesian", {"x": 0.2})])], on_call=inspect_call)
    async def progress(message: str) -> None:
        updates.append(message)
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path, progress=progress))
    assert candidate.record["status"] == "proposed" and len(runtime.calls) == 1
    assert f"Prompt: {len(runtime.calls[0]['prompt'])} characters." in updates[0]
    assert "RA proposal received in" in updates[1]
    assert all("evidence request" not in message for message in updates)


def test_composition_rejects_oversized_essential_contract_without_truncating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_composition(tmp_path)
    inputs = primitive_composition._load_inputs(tmp_path)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    payload = deepcopy(inputs.composition_input)
    payload["primitive_catalog"][0]["description"] = "essential contract " * 6400
    monkeypatch.setattr(primitive_composition, "_load_inputs", lambda root: dataclasses.replace(inputs, composition_input=payload))
    runtime = _ProgramRuntime([_program_action([("release_part", {})])])
    with pytest.raises(PrimitiveCompositionError, match="64000-character prompt budget"):
        asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert runtime.calls == []
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("scope", [VALIDATION_SCOPE, GAZEBO_PICK_PLACE_SCOPE])
def test_composition_scope_limits_choices_without_rewriting_catalog_or_steps(
    tmp_path: Path, scope: str,
) -> None:
    """Keep full contracts while the model can propose only implemented capabilities."""
    _capture_geometry_context(tmp_path)
    inputs = primitive_composition._load_inputs(tmp_path)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    steps = [("move_cartesian", {"x": 0.1}), ("release_part", {}), ("move_cartesian", {"z": 0.2})]
    runtime = _ProgramRuntime([_program_action(steps)])
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path, validation_scope=scope))
    assert candidate.record["status"] == "proposed"
    assert candidate.record["primitive_steps"] == [
        {"primitive_symbol": symbol, "params": params} for symbol, params in steps
    ]
    delivered = _composition_input_from_prompt(runtime.calls[0]["prompt"])
    assert delivered["primitive_catalog"] == inputs.composition_input["primitive_catalog"]
    variants = runtime.calls[0]["response_format"]["schema"]["properties"]["action"]["anyOf"]
    proposal = next(item for item in variants if item["properties"]["kind"]["enum"] == ["propose"])
    selectable = proposal["properties"]["primitive_steps"]["items"]["properties"]["primitive_symbol"]["enum"]
    assert selectable == [symbol for symbol in inputs.catalog if symbol in supported_primitive_symbols(scope)]
    assert set(inputs.catalog) - set(selectable) == {"detect_parts", "move_relative", "move_to_named_pose"}
    assert json.dumps(selectable) in runtime.calls[0]["prompt"]
    assert read_primitive_composition_diagnostic(tmp_path)["candidate"] == candidate.record
    assert all(path.read_bytes() == data for path, data in original.items())


def test_scope_with_no_supported_catalog_entries_can_report_unsupported(tmp_path: Path) -> None:
    """A resource with no implemented operation must not send an empty enum to the model."""
    _capture_geometry_context(tmp_path)
    inputs = primitive_composition._load_inputs(tmp_path)
    inputs.composition_input["primitive_catalog"] = [inputs.catalog["detect_parts"]]
    schema = primitive_composition._response_format(inputs, validation_scope=GAZEBO_PICK_PLACE_SCOPE)
    kinds = {item["properties"]["kind"]["enum"][0]
             for item in schema["schema"]["properties"]["action"]["anyOf"]}
    assert "propose" not in kinds and "unsupported" in kinds
    assert "execution support are []" in primitive_composition._composition_prompt(
        inputs, validation_scope=GAZEBO_PICK_PLACE_SCOPE,
    )


def test_parameterized_composition_uses_only_selected_ra_isolated_llm(tmp_path: Path) -> None:
    runtime, agent, _ = _prepare_composition(tmp_path)
    agent.composition_response = {
        "action": _program_action(
            [
                ("release_part", {}),
                ("move_cartesian", {"x": 0, "y": 0, "z": 0.1}),
            ]
        )
    }

    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))

    assert candidate.record["status"] == "proposed"
    call = agent.composition_calls[-1]
    assert call["tools"] is None
    assert call["max_tool_rounds"] == 0
    assert call["include_agent_instructions"] is False
    assert call["response_format"]["name"] == "spec2primitives_primitive_composition"
    assert agent.feasibility_calls == []
    assert [step["primitive_symbol"] for step in candidate.record["primitive_steps"]] == [
        "release_part",
        "move_cartesian",
    ]  # Structural acceptance deliberately does not certify this sequence's semantics.


@pytest.mark.parametrize(
    ("symbol", "params", "reason"),
    [
        ("invented_primitive", {}, "unknown primitive_symbol"),
        ("release_part", {"invented": 1}, "unknown parameters"),
        ("move_cartesian", {"x": False, "y": 0, "z": 0}, "declared type number"),
        ("move_cartesian", {"x": "0.1", "y": 0, "z": 0}, "declared type number"),
        (
            "move_cartesian",
            {
                "x": 0,
                "y": 0,
                "z": {"result_ref": {"step_index": 1, "field_path": "/target_pose/x"}},
            },
            "earlier step",
        ),
        (
            "move_cartesian",
            {
                "x": 0,
                "y": 0,
                "z": {"value_ref": {"record_ref": "evaluations/answer.json", "field_path": "/z"}},
            },
            "not approved",
        ),
    ],
)
def test_parameterized_composition_rejects_without_repair(
    tmp_path: Path,
    symbol: str,
    params: dict[str, Any],
    reason: str,
) -> None:
    _prepare_composition(tmp_path)
    action = _program_action([(symbol, params)])
    runtime = _ProgramRuntime([action])

    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))

    assert candidate.record["status"] == "invalid"
    assert reason in candidate.record["reason"]
    assert len(runtime.calls) == 1
    assert candidate.record["primitive_steps"] == [{"primitive_symbol": symbol, "params": params}]
    trace = read_primitive_composition_diagnostic(tmp_path)["trace"]
    assert trace[0]["response"] == {"action": action}


@pytest.mark.parametrize("pointer", [
    "/target_pose/missing", "/undeclared", "/target_pose/~2",
    "/declared_output/approach_pose/x", "/declared_output/target_pose/z", "/declared_output/tz",
])
def test_parameterized_composition_rejects_undeclared_result_paths(
    tmp_path: Path, pointer: str
) -> None:
    _prepare_composition(tmp_path)
    runtime = _ProgramRuntime(
        [
            _program_action(
                [
                    ("compute_pick_targets", {"part_name": "medium gear"}),
                    (
                        "move_cartesian",
                        {
                            "x": 0,
                            "y": 0,
                            "z": {"result_ref": {"step_index": 1, "field_path": pointer}},
                        },
                    ),
                ]
            )
        ]
    )
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert candidate.record["status"] == "invalid"
    assert candidate.record["primitive_steps"][1]["params"]["z"]["result_ref"]["field_path"] == pointer
    assert len(runtime.calls) == len(candidate.record["exchange_refs"]) == 1
    if "undeclared result field" in candidate.record["reason"]:
        assert pointer in candidate.record["reason"]
        assert "step 1 (compute_pick_targets)" in candidate.record["reason"]
        assert "Available top-level result fields:" in candidate.record["reason"]
        assert "target_pose" in candidate.record["reason"]


def test_fitting_result_paths_select_declared_fields_without_wrappers(tmp_path: Path) -> None:
    """Accept pick and insertion dependencies while their measured inputs await binding."""
    _capture_geometry_context(tmp_path)

    def pose(step_index: int, name: str, fields: tuple[str, ...] = ("x", "y", "z")) -> dict[str, Any]:
        return {field: {"result_ref": {"step_index": step_index, "field_path": f"/{name}/{field}"}}
                for field in fields}

    pick_ctx = {name: {"result_ref": {"step_index": 1, "field_path": f"/{name}"}}
                for name in ("part_name", "tz", "pick_tcp_z", "tcp_offset_z", "part_height")}
    orientation = ("x", "y", "z", "qx", "qy", "qz", "qw")
    steps = [
        ("compute_pick_targets", {"part_name": "medium gear", "prefer_live_detection": False,
                                  "target_pose_source": "observed_bounds_center"}),
        ("move_cartesian", pose(1, "approach_pose")),
        ("move_cartesian", pose(1, "target_pose")),
        ("grasp_part", {"part_name": "medium gear"}),
        ("move_cartesian", pose(1, "approach_pose")),
        ("compute_place_targets", {"part_name": "medium gear", "pick_ctx": pick_ctx}),
        ("move_cartesian", pose(6, "approach_pose", orientation)),
        ("move_cartesian", pose(6, "pre_insert_pose", orientation)),
        ("move_cartesian", pose(6, "insert_pose", orientation)),
        ("release_part", {"part_name": "medium gear"}),
        ("move_cartesian", pose(6, "pre_insert_pose", orientation)),
    ]
    runtime = _ProgramRuntime([_program_action(steps)])
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert candidate.record["status"] == "proposed", candidate.record["reason"]
    assert candidate.record["primitive_steps"] == [
        {"primitive_symbol": symbol, "params": params} for symbol, params in steps
    ]
    view = read_primitive_composition_diagnostic(tmp_path)
    assert view["status"] == "proposed"
    assert any(issue["status"] == "deferred" for issue in view["binding_issues"])
    assert any(issue["step_index"] == 1 and issue["status"] == "missing" for issue in view["binding_issues"])
    assert len(runtime.calls) == 1


@pytest.mark.parametrize("reference,pointer", [("evaluations/answer.json", ""), ("../answer.json", ""), (None, "/translated_location_m/01")])
def test_composition_rejects_unapproved_or_traversing_value_refs(tmp_path: Path, reference: str | None, pointer: str) -> None:
    _, _, current_ref = _prepare_composition(tmp_path)
    runtime = _ProgramRuntime([_program_action([("move_cartesian", {"x": {"value_ref": {
        "record_ref": reference or current_ref, "field_path": pointer,
    }}})])])
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert candidate.record["status"] == "invalid" and len(runtime.calls) == 1
    assert "error" in read_primitive_composition_diagnostic(tmp_path)["trace"][0]["result"]


def test_composition_load_validates_completion_once_per_integrity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Share validation inside one load without caching across integrity checks."""
    from cais_spade_llm.spec2primitives.agents.pa import grounding_contracts

    _prepare_composition(tmp_path)
    validate = grounding_contracts._validated_two_decision_completion
    calls = []

    def checked(root: Path, value: Mapping[str, object]):
        calls.append(root)
        return validate(root, value)

    monkeypatch.setattr(grounding_contracts, "_validated_two_decision_completion", checked)
    inputs = primitive_composition._load_inputs(tmp_path)
    assert calls == [tmp_path]
    primitive_composition._assert_inputs_unchanged(inputs)
    assert calls == [tmp_path, tmp_path]


@pytest.mark.parametrize("changed", ["typed_record", "assignment", "native_source", "other_arm"])
def test_parameterized_composition_stops_changed_evidence_and_preserves_response(
    tmp_path: Path,
    changed: str,
) -> None:
    _, _, current_ref = _prepare_composition(tmp_path)
    from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import load_pa_context_grounding_completion

    completion = load_pa_context_grounding_completion(tmp_path).to_record()

    def change_evidence() -> None:
        if changed == "assignment":
            ref = completion["resource_assignment_delta_ref"]
        elif changed == "native_source":
            location = _read_json(tmp_path / current_ref)
            segmentation = _read_json(tmp_path / location["source_segmentation"]["ref"])
            ref = segmentation["cameras"][0]["source_artifacts"]["rgb"]["ref"]
        elif changed == "other_arm":
            other = next(
                _read_json(tmp_path / item["ref"])
                for item in completion["tool_call_refs"]
                if _read_json(tmp_path / item["ref"])["arguments"]["resource_symbol"] == "ur5e"
            )
            ref = other["result_ref"]
        else:
            ref = current_ref
        path = tmp_path / ref
        path.write_bytes(path.read_bytes() + b"\n")

    action = _program_action([("release_part", {})])
    runtime = _ProgramRuntime([action], on_call=change_evidence)
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert candidate.record["status"] != "proposed"
    assert len(runtime.calls) == 1
    exchange = _read_json(candidate.path.parent / "exchange_0001.json")
    assert exchange["response"] == {"action": action}
    assert read_primitive_composition_diagnostic(tmp_path)["status"] == "blocked"


@pytest.mark.parametrize("kind", ["read_record", "read_records", "query_ontology", "request_context"])
def test_composition_rejects_retired_evidence_actions_and_keeps_independent_attempts(tmp_path: Path, kind: str) -> None:
    _prepare_composition(tmp_path)
    runtime = _ProgramRuntime([{"kind": kind}])
    first = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert first.record["status"] == "invalid" and len(runtime.calls) == len(first.record["exchange_refs"]) == 1
    before = first.path.read_bytes()
    fresh = _ProgramRuntime([{"kind": "unsupported", "reason": "A required capability is unavailable."}])
    second = asyncio.run(author_primitive_program_candidate(fresh, tmp_path))
    assert second.path.parent.name == "attempt_0002" and second.record["status"] == "unsupported"
    assert first.path.read_bytes() == before and "EXCHANGES" not in fresh.calls[0]["prompt"]
    assert read_primitive_composition_diagnostic(tmp_path)["attempt_count"] == 2


def test_parameterized_composition_blocks_changed_trace_before_ra_call(tmp_path: Path) -> None:
    _prepare_composition(tmp_path)
    runtime = _ProgramRuntime([_program_action([("release_part", {})])])
    first = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    path = first.path.parent / "exchange_0001.json"
    path.write_text(path.read_text() + "\n")
    fresh = _ProgramRuntime([_program_action([("release_part", {})])])
    with pytest.raises(PrimitiveCompositionError, match="exchange changed"):
        asyncio.run(author_primitive_program_candidate(fresh, tmp_path))
    assert fresh.calls == []


@pytest.mark.parametrize("params", ['{"part_name":"a","part_name":"b"}', '{"part_name":NaN}'])
def test_parameterized_composition_rejects_ambiguous_json(tmp_path: Path, params: str) -> None:
    _prepare_composition(tmp_path)
    runtime = _ProgramRuntime(
        [
            {
                "kind": "propose",
                "primitive_steps": [
                    {"primitive_symbol": "grasp_part", "params": params},
                ],
            }
        ]
    )
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert candidate.record["status"] == "invalid"
    assert len(runtime.calls) == 1


def _valid_catalog() -> list[dict[str, Any]]:
    return [
        {
            "primitive_symbol": "detect_parts",
            "operation_description": (
                "Detect parts via perception service. Optionally filter by part name."
            ),
            "typed_parameters": [{"name": "part_name", "type": "string", "required": False}],
            "typed_results": [{"name": "parts", "type": "array"}],
            "invocation_binding": "detect_parts",
            "truthful_limits": ["requires the configured perception service"],
            "direct_evidence": [],
            "evaluator_endpoints": [],
            "conditions": {},
            "effects": {},
        },
        {
            "primitive_symbol": "move_pose",
            "operation_description": (
                "Move end-effector to an absolute pose with explicit quaternion orientation."
            ),
            "typed_parameters": [
                {"name": "x", "type": "number", "required": True},
                {"name": "y", "type": "number", "required": True},
                {"name": "z", "type": "number", "required": True},
                {"name": "qx", "type": "number", "required": True},
                {"name": "qy", "type": "number", "required": True},
                {"name": "qz", "type": "number", "required": True},
                {"name": "qw", "type": "number", "required": True},
                {"name": "speed", "type": "number", "required": False},
            ],
            "typed_results": [
                {"name": "success", "type": "boolean"},
                {"name": "message", "type": "string"},
            ],
            "invocation_binding": "move_pose",
            "truthful_limits": [
                "requires a usable controller",
                "does not prove IK or collision feasibility before validation",
            ],
            "direct_evidence": [],
            "evaluator_endpoints": [],
            "conditions": {},
            "effects": {"current_pose": {"pose_absolute_from_params": ["x", "y", "z"]}},
        },
    ]


def _raw_live_catalog() -> list[dict[str, object]]:
    entries = {
        "compute_pick_targets": {
            "params": {"part_name": {"type": "string"}},
            "required_params": ["part_name"],
            "output_schema": {"target_pose": {"x": "number", "y": "number"}},
            "primitive_kind": "pick",
        },
        "compute_place_targets": {
            "params": {"part_name": {"type": "string"}},
            "required_params": ["part_name"],
            "output_schema": {"target_pose": {"x": "number", "y": "number"}},
            "primitive_kind": "place",
        },
        "detect_parts": {
            "params": {"part_name": {"type": "string"}},
            "required_params": [],
            "output_schema": {"pose": {"x": "number", "y": "number", "z": "number"}},
            "primitive_kind": "observe",
            "preconditions": {"controller_ready": True},
            "effects": {"part_observed": True},
        },
        "grasp_part": {
            "params": {"part_name": {"type": "string"}},
            "required_params": ["part_name"],
            "primitive_kind": "pick",
        },
        "move_cartesian": {
            "params": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "required_params": ["x", "y", "z"],
            "primitive_kind": "motion",
        },
        "move_relative": {
            "params": {
                "dx": {"type": "number"},
                "dy": {"type": "number"},
                "dz": {"type": "number"},
            },
            "required_params": ["dx", "dy", "dz"],
            "primitive_kind": "motion",
        },
        "move_to_named_pose": {
            "params": {"pose_name": {"type": "string"}},
            "required_params": ["pose_name"],
            "primitive_kind": "home",
        },
        "release_part": {
            "params": {"part_name": {"type": "string"}},
            "required_params": [],
            "primitive_kind": "release",
        },
    }
    return [
        {
            "name": name,
            "resource_type": "robot",
            "description": f"Robot synthesis primitive {name}.",
            "params": dict(entries[name].get("params") or {}),
            "required_params": list(entries[name].get("required_params") or []),
            "preconditions": dict(entries[name].get("preconditions") or {}),
            "effects": dict(entries[name].get("effects") or {}),
            "primitive_kind": entries[name]["primitive_kind"],
            "output_schema": dict(entries[name].get("output_schema") or {}),
            "semantic_summary": f"Robot synthesis primitive {name}.",
        }
        for name in _EXPECTED_XARM6_SYNTHESIS_SYMBOLS
    ]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _assert_no_ra_context_snapshots(root: Path) -> None:
    """Allow Phase 4 validation evidence but no Phase 5 RA context capture."""
    resource_root = root / "resources/xarm6@localhost"
    assert not (resource_root / "validation").exists()
    assert not (resource_root / "robot_state").exists()
    assert not (resource_root / "primitive_catalog_snapshot").exists()


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint", None)
    return _fingerprint(payload)


def _composition_input_from_prompt(value: object) -> dict[str, Any]:
    marker = "COMPOSITION_INPUT\n"
    _prefix, separator, serialized = str(value).partition(marker)
    assert separator == marker
    composition_input = json.loads(serialized.split("\n\nEXCHANGES", 1)[0])
    assert isinstance(composition_input, dict)
    return composition_input


@pytest.mark.parametrize("artifact", ["completion", "proposal"])
@pytest.mark.parametrize("retired_field", ["schema_version", "semantic_review_ref"])
def test_incompatible_completion_cannot_activate_ra_or_compose(tmp_path, artifact, retired_field):
    completion = persist_native_completion_fixture(tmp_path).to_record()
    path = tmp_path / (
        "interaction_record/context_completion_0001.json"
        if artifact == "completion"
        else completion["ontology_projection_ref"]
    )
    record = _read_json(path)
    record[retired_field] = 1 if retired_field == "schema_version" else "old/review.json"
    _write_json(path, record)

    class UncalledRuntime:
        async def request_assigned_context(self, assignment):
            raise AssertionError("Historical completion must not contact RA.")

    with pytest.raises(RAContextHandoffError, match="requires"):
        asyncio.run(activate_selected_ra_context(UncalledRuntime(), tmp_path))
    with pytest.raises((RAContextHandoffError, PrimitiveCompositionError)):
        asyncio.run(author_primitive_program_candidate(UncalledRuntime(), tmp_path))
    assert not (tmp_path / "composition").exists()


def _write_program_record(path: Path, value: Mapping[str, Any]) -> None:
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    _write_json(path, {**payload, "fingerprint": fingerprint})


def _declared_geometry_catalog() -> list[dict[str, Any]]:
    """Use actual shared metadata without constructing a controller or running a helper."""
    from cais_spade_llm.function_analyzer import FunctionAnalyzer
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        GazeboPickPlaceController,
    )
    from cais_spade_llm.resources.robot.robot_primitives import ROBOT_OBSERVATION_OUTPUT_SCHEMA_MAP

    catalog = _raw_live_catalog()
    for entry in catalog:
        function = getattr(GazeboPickPlaceController, entry["name"])
        entry["params"] = FunctionAnalyzer().analyze_function(function)["parameters"]["properties"]
        entry["required_params"] = [
            name
            for name, parameter in inspect.signature(function).parameters.items()
            if name != "self" and parameter.default is inspect.Parameter.empty
        ]
        metadata = FunctionAnalyzer._extract_yaml_frontmatter(function)
        entry["preconditions"] = metadata.get("preconditions", {})
        entry["effects"] = metadata.get("effects", {})
        entry["output_schema"] = deepcopy(
            ROBOT_OBSERVATION_OUTPUT_SCHEMA_MAP.get(entry["name"], {})
        )
    return catalog


def _capture_geometry_context(tmp_path: Path, *, frame: str | None = "world") -> Any:
    completion = persist_native_completion_fixture(tmp_path).to_record()
    agent = _LiveRobotAgent(primitive_catalog=_declared_geometry_catalog())
    agent.controller_config = {
        "move_group": {"frame_id": frame, "ee_link": "configured_ee", "tcp_link": "configured_tcp"}
    }
    adapter = InProcessRobotAgentCompositionRuntime(
        _LiveRobotAgentHost([agent]), contexts_root=tmp_path,
    )
    captured = asyncio.run(activate_selected_ra_context(adapter, tmp_path))
    return agent, adapter, captured, completion["typed_context_refs"][0]["ref"]


def test_composition_projects_minimal_contracts_and_filters_every_state_read(
    tmp_path: Path,
) -> None:
    agent, adapter, _, _ = _capture_geometry_context(tmp_path)
    sentinel = "RECOVERY_RECIPE_MUST_NOT_REACH_COMPOSITION"
    agent.robot_state.update(
        {
            "held_part": None,
            "recovery_adapter": {"examples": [sentinel]},
            "resource_facets": {
                "manipulator": {"held_part": None, "capability_decompositions": [sentinel]}
            },
            "function_names": [sentinel],
        }
    )
    captured = asyncio.run(activate_selected_ra_context(adapter, tmp_path))
    state_ref = captured.robot_state_path.relative_to(tmp_path).as_posix()
    catalog_ref = captured.primitive_catalog_path.relative_to(tmp_path).as_posix()
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    runtime = _ProgramRuntime([_program_action([("release_part", {}), ("grasp_part", {"part_name": "medium gear"})])])
    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    view = read_primitive_composition_diagnostic(tmp_path)
    assert candidate.record["status"] == view["status"] == "proposed"
    assert all(sentinel not in call["prompt"] for call in runtime.calls)
    assert "compute_pick_targets and compute_place_targets" not in runtime.calls[0]["prompt"]
    inputs = primitive_composition._load_inputs(tmp_path)
    assert primitive_composition._evidence_value(inputs, state_ref, "/robot_state/held_part") is None
    with pytest.raises(PrimitiveCompositionError):
        primitive_composition._evidence_value(inputs, state_ref, "/robot_state/recovery_adapter")
    with pytest.raises(PrimitiveCompositionError):
        primitive_composition._evidence_value(inputs, catalog_ref, "")
    projected = {
        item["primitive_symbol"]: item for item in view["composition_input"]["primitive_catalog"]
    }
    authoritative = {
        item["primitive_symbol"]: item for item in captured.primitive_catalog.primitive_catalog
    }
    for symbol in ("grasp_part", "release_part"):
        assert set(projected[symbol]["conditions"]) == {"held_part"}
        assert set(projected[symbol]["effects"]) == {"held_part"}
        assert set(authoritative[symbol]["effects"]) == {
            "held_part",
            "current_state",
            "gripper_state",
        }
        assert "model_name" in authoritative[symbol]["parameter_schemas"]
        assert projected[symbol]["parameter_schemas"]["part_name"] == (
            authoritative[symbol]["parameter_schemas"]["part_name"]
        )
    assert "model_name" not in json.dumps(projected)
    assert projected["grasp_part"]["effects"]["held_part"] == {
        "set_from_param_any_of": ["part_name"]
    }
    assert not view["binding_issues"]
    assert "composition contract" in runtime.calls[0][
        "prompt"
    ]
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_nested_metadata_preserves_callable_contracts_for_recovery() -> None:
    from cais_spade_llm.function_analyzer import FunctionAnalyzer
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        GazeboPickPlaceController,
    )

    catalog = {item["name"]: item for item in _declared_geometry_catalog()}
    pick = catalog["compute_pick_targets"]
    geometry = pick["params"]["product_geometry"]
    assert geometry["x-grounding-fields"] == ["board_center", "part_height_m"]
    assert geometry["properties"]["board_center"]["x-grounding-fields"] == ["z"]
    assert "required" not in geometry
    assert "product_geometry" not in pick["required_params"]
    assert pick["params"]["detected_parts"]["items"]["properties"]["part_name"]["type"] == "string"
    move = catalog["move_cartesian"]
    assert move["required_params"] == ["x", "y", "z"]
    assert move["params"]["x"]["x-frame-source"] == "controller_config.move_group.frame_id"
    assert catalog["grasp_part"]["required_params"] == ["model_name"]
    assert (
        catalog["grasp_part"]["params"]["model_name"]["x-binding-role"] == "controller_identifier"
    )
    assert "current_state" in catalog["grasp_part"]["effects"]
    geometry["properties"]["board_center"]["properties"]["z"]["type"] = "string"
    fresh = FunctionAnalyzer().analyze_function(GazeboPickPlaceController.compute_pick_targets)
    assert (
        fresh["parameters"]["properties"]["product_geometry"]["properties"]["board_center"][
            "properties"
        ]["z"]["type"]
        == "number"
    )


def test_geometry_gaps_and_conditional_results_preserve_the_exact_program(tmp_path: Path) -> None:
    _, _, captured, current_ref = _capture_geometry_context(tmp_path)
    observed = {"value_ref": {"record_ref": current_ref, "field_path": "/translated_location_m/0"}}
    steps = [
        ("compute_pick_targets", {"part_name": "medium gear", "target_pose": {"x": observed}}),
        (
            "compute_place_targets",
            {
                "pick_ctx": {"result_ref": {"step_index": 1, "field_path": ""}},
                "product_geometry": {"board_center": {}},
                "destination_location": "middle gear shaft destination reference",
            },
        ),
        (
            "move_cartesian",
            {"z": {"result_ref": {"step_index": 2, "field_path": "/insert_pose/z"}}},
        ),
        ("grasp_part", {"part_name": "medium gear"}),
        ("move_cartesian", {"x": observed}),
    ]
    candidate = asyncio.run(
        author_primitive_program_candidate(_ProgramRuntime([_program_action(steps)]), tmp_path)
    )
    before = candidate.path.read_bytes()
    view = read_primitive_composition_diagnostic(tmp_path)
    assert candidate.record["status"] == view["status"] == "proposed"
    assert candidate.record["primitive_steps"] == [
        {"primitive_symbol": s, "params": p} for s, p in steps
    ]
    issues = {
        (item["step_index"], item["parameter_path"], item["status"])
        for item in view["binding_issues"]
    }
    assert (1, "/product_geometry/board_center/z", "missing") in issues
    assert (1, "/product_geometry/part_height_m", "missing") in issues
    assert (1, "/target_pose/y", "missing") in issues
    assert (2, "/product_geometry/slot_floor_z_m", "missing") in issues
    assert (2, "/product_geometry/target_reference/target_point", "missing") in issues
    assert all((2, "/product_geometry/target_origin_pose/" + axis, "missing") in issues for axis in ("x", "y", "z"))
    assert (2, "/destination_location", "unverified") in issues
    assert (2, "/pick_ctx", "deferred") in issues
    assert (3, "/z", "deferred") in issues
    assert not any("model_name" in path for _, path, _ in issues)
    assert (
        5,
        "/x",
        "unverified",
    ) in issues  # Same frame does not make an object point an EE target.
    assert not any(item["status"] == "incompatible" for item in view["binding_issues"])
    assert captured.robot_state.robot_state["motion_context"] == {
        "source": "controller_config.move_group",
        "frame_id": "world",
        "ee_link": "configured_ee",
        "tcp_link": "configured_tcp",
    }
    assert candidate.path.read_bytes() == before
    assert "binding_issues" not in candidate.record  # Derived view, never a rewrite of RA's record.


@pytest.mark.parametrize("frame", ["world", "robot_base", None])
def test_binding_report_checks_configured_frames_without_converting(
    tmp_path: Path, frame: str | None
) -> None:
    _, _, _, current_ref = _capture_geometry_context(tmp_path, frame=frame)
    coordinate = {
        "value_ref": {"record_ref": current_ref, "field_path": "/translated_location_m/0"}
    }
    steps = [("move_cartesian", {"x": coordinate})]
    candidate = asyncio.run(
        author_primitive_program_candidate(_ProgramRuntime([_program_action(steps)]), tmp_path)
    )
    issues = read_primitive_composition_diagnostic(tmp_path)["binding_issues"]
    assert candidate.record["primitive_steps"][0]["params"]["x"] == coordinate
    incompatible = [issue for issue in issues if issue["status"] == "incompatible"]
    assert bool(incompatible) == (frame == "robot_base")
    if frame == "robot_base":
        assert (
            "world" in incompatible[0]["message"] and "no conversion" in incompatible[0]["message"]
        )
    if frame is None:
        assert any(issue["parameter_path"].endswith("/frame_id") for issue in issues)


@pytest.mark.parametrize(
    "params,reason",
    [
        ({"target_pose": {"x": "0.4"}}, "declared type number"),
        ({"target_pose": {"value_ref": {}, "x": 1}}, "only value_ref"),
        ({"product_geometry": {"part_height_m": -1}}, "numeric bounds"),
        ({"product_geometry": {"slot_xy": [0.1]}}, "declared length"),
    ],
)
def test_binding_types_and_malformed_references_remain_errors(
    tmp_path: Path, params: dict, reason: str
) -> None:
    _capture_geometry_context(tmp_path)
    symbol = (
        "compute_place_targets"
        if "slot_xy" in params.get("product_geometry", {})
        else "compute_pick_targets"
    )
    candidate = asyncio.run(
        author_primitive_program_candidate(
            _ProgramRuntime([_program_action([(symbol, params)])]), tmp_path
        )
    )
    assert candidate.record["status"] == "invalid"
    assert reason in candidate.record["reason"]
    assert candidate.record["primitive_steps"][0]["params"] == params


def test_binding_report_does_not_promote_raw_cad_or_labels_to_geometry(tmp_path: Path) -> None:
    _capture_geometry_context(tmp_path)
    steps = [
        (
            "compute_place_targets",
            {"product_geometry": {"record_type": "CADMeshRecord", "coordinate_frame": "CAD_local"}},
        ),
        ("compute_place_targets", {"destination_location": "middle gear shaft destination reference"}),
    ]
    candidate = asyncio.run(
        author_primitive_program_candidate(_ProgramRuntime([_program_action(steps)]), tmp_path)
    )
    issues = read_primitive_composition_diagnostic(tmp_path)["binding_issues"]
    assert candidate.record["status"] == "proposed"
    assert any(issue["step_index"] == 1 and issue["status"] == "incompatible" for issue in issues)
    assert any(
        issue["step_index"] == 2
        and issue["parameter_path"] == "/destination_location"
        and issue["status"] == "unverified"
        for issue in issues
    )


@pytest.mark.parametrize(
    "symbol,params",
    [
        ("grasp_part", {"part_name": "medium gear", "model_name": "medium gear"}),
        ("release_part", {"model_name": "simulated_instance"}),
        ("compute_pick_targets", {"product_geometry": {"model_name": "simulated_instance"}}),
        ("compute_pick_targets", {"target_pose": {"model_name": "simulated_instance"}}),
        ("compute_pick_targets", {"detected_parts": [{"model_name": "simulated_instance"}]}),
        ("compute_place_targets", {"pick_ctx": {"model_name": "simulated_instance"}}),
        (
            "compute_place_targets",
            {"product_geometry": {"metadata": [{"model_name": "simulated_instance"}]}},
        ),
    ],
)
def test_composition_rejects_execution_arguments_without_rewriting(
    tmp_path: Path, symbol: str, params: dict[str, Any]
) -> None:
    _capture_geometry_context(tmp_path)
    action = _program_action([(symbol, params)])
    runtime = _ProgramRuntime([action])

    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))

    assert candidate.record["status"] == "invalid"
    assert "model_name is execution-only" in candidate.record["reason"]
    assert candidate.record["primitive_steps"] == [{"primitive_symbol": symbol, "params": params}]
    view = read_primitive_composition_diagnostic(tmp_path)
    assert view["status"] == "invalid"
    assert view["trace"][-1]["response"] == {"action": action}
    assert len(runtime.calls) == 1


def test_composition_rejects_identifier_fields_in_referenced_objects(tmp_path: Path) -> None:
    agent, adapter, _, _ = _capture_geometry_context(tmp_path)
    agent.robot_state["execution_context"] = {"model_name": "simulated_instance"}
    captured = asyncio.run(activate_selected_ra_context(adapter, tmp_path))
    params = {
        "product_geometry": {
            "value_ref": {
                "record_ref": captured.robot_state_path.relative_to(tmp_path).as_posix(),
                "field_path": "/robot_state/execution_context",
            }
        }
    }
    candidate = asyncio.run(
        author_primitive_program_candidate(
            _ProgramRuntime([_program_action([("compute_pick_targets", params)])]), tmp_path
        )
    )
    assert candidate.record["status"] == "invalid"
    assert "model_name is execution-only" in candidate.record["reason"]
    assert candidate.record["primitive_steps"][0]["params"] == params


def test_composition_rejects_removed_helper_identifier_outputs(tmp_path: Path) -> None:
    _capture_geometry_context(tmp_path)
    steps = [
        ("compute_pick_targets", {"part_name": "medium gear"}),
        (
            "compute_place_targets",
            {"part_name": {"result_ref": {"step_index": 1, "field_path": "/model_name"}}},
        ),
    ]
    candidate = asyncio.run(
        author_primitive_program_candidate(_ProgramRuntime([_program_action(steps)]), tmp_path)
    )
    assert candidate.record["status"] == "invalid"
    assert "undeclared result field" in candidate.record["reason"]
    assert candidate.record["primitive_steps"] == [
        {"primitive_symbol": symbol, "params": params} for symbol, params in steps
    ]


def test_composition_projection_preserves_non_identifier_geometry_requirements(tmp_path: Path) -> None:
    agent, adapter, _, _ = _capture_geometry_context(tmp_path)
    pick = agent.primitive_catalog[0]
    geometry = pick["params"]["product_geometry"]
    geometry["required"] = ["model_name", "part_height_m"]
    geometry["x-grounding-fields"].append("model_name")
    pick["params"]["detected_parts"]["items"]["required"] = ["model_name", "part_name"]
    captured = asyncio.run(activate_selected_ra_context(adapter, tmp_path))
    before = captured.primitive_catalog_path.read_bytes()
    runtime = _ProgramRuntime(
        [_program_action([("compute_pick_targets", {"part_name": "medium gear"})])]
    )

    candidate = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))

    assert candidate.record["status"] == "proposed"
    delivered = _composition_input_from_prompt(runtime.calls[0]["prompt"])["primitive_catalog"]
    params = delivered[0]["parameter_schemas"]
    assert params["product_geometry"]["required"] == ["part_height_m"]
    assert params["product_geometry"]["x-grounding-fields"] == ["board_center", "part_height_m"]
    assert params["detected_parts"]["items"]["required"] == ["part_name"]
    assert "model_name" not in json.dumps(delivered)
    assert captured.primitive_catalog_path.read_bytes() == before


def test_saved_attempt_uses_its_recorded_catalog_and_next_attempt_uses_new_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _capture_geometry_context(tmp_path)
    original_steps = [
        ("compute_pick_targets", {"part_name": "medium gear"}),
        (
            "grasp_part",
            {
                "part_name": "medium gear",
                "model_name": {"result_ref": {"step_index": 1, "field_path": "/model_name"}},
            },
        ),
    ]
    # Reproduce the prior interface, which still exposed the simulator binding.
    with monkeypatch.context() as patch:
        patch.setattr(composition_context, "_without_model_name", deepcopy)
        old = asyncio.run(
            author_primitive_program_candidate(
                _ProgramRuntime([_program_action(original_steps)]), tmp_path
            )
        )
    assert old.record["status"] == "proposed"
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    reopened = read_primitive_composition_diagnostic(tmp_path)
    assert reopened["status"] == "proposed"
    assert reopened["candidate"] == old.record
    assert "model_name" in json.dumps(reopened["composition_input"]["primitive_catalog"])
    assert any(issue["parameter_path"] == "/model_name" for issue in reopened["binding_issues"])
    assert all(path.read_bytes() == value for path, value in before.items())

    steps = [("grasp_part", {"part_name": "medium gear"}), ("release_part", {})]
    runtime = _ProgramRuntime([_program_action(steps)])
    new = asyncio.run(author_primitive_program_candidate(runtime, tmp_path))
    assert new.record["status"] == "proposed"
    assert new.record["primitive_steps"] == [
        {"primitive_symbol": symbol, "params": params} for symbol, params in steps
    ]
    assert "model_name" not in runtime.calls[0]["prompt"]
    assert all(path.read_bytes() == value for path, value in before.items())
    reopened = read_primitive_composition_diagnostic(tmp_path)
    assert reopened["candidate"] == new.record
    assert reopened["attempt_count"] == 2
    assert not reopened["binding_issues"]
    assert "model_name" not in json.dumps(reopened["composition_input"]["primitive_catalog"])


def test_saved_request_catalog_must_still_match_pinned_context(tmp_path: Path) -> None:
    _capture_geometry_context(tmp_path)
    inputs = primitive_composition._load_inputs(tmp_path)
    request = {"prompt": primitive_composition._composition_prompt(inputs)}
    prefix, payload = request["prompt"].split("\n\nCOMPOSITION_INPUT\n", 1)
    recorded = json.loads(payload)
    recorded["primitive_catalog"][0]["primitive_symbol"] = "invented_primitive"
    request["prompt"] = prefix + "\n\nCOMPOSITION_INPUT\n" + json.dumps(recorded)

    with pytest.raises(PrimitiveCompositionError, match="catalog differs from its context"):
        primitive_composition._recorded_request_inputs(inputs, request)


@pytest.mark.parametrize("recorded_scope", [None, "unknown", "rigid_vertical_gear_assembly_direct_cartesian"])
def test_saved_authoring_scope_must_match_its_prompt(tmp_path: Path, recorded_scope: str | None) -> None:
    """An authoring request cannot relabel the scope the RA received."""
    from cais_spade_llm.spec2primitives.agents.ra.validation_scope import GAZEBO_PICK_PLACE_SCOPE

    _capture_geometry_context(tmp_path)
    inputs = primitive_composition._load_inputs(tmp_path)
    request = {
        "validation_scope": recorded_scope,
        "prompt": primitive_composition._composition_prompt(inputs, validation_scope=GAZEBO_PICK_PLACE_SCOPE),
    }
    with pytest.raises(ValueError, match="scope"):
        primitive_composition._recorded_request_inputs(inputs, request)


def test_result_object_type_check_is_recursive(tmp_path: Path) -> None:
    _, _, _, current_ref = _capture_geometry_context(tmp_path)
    inputs = primitive_composition._load_inputs(tmp_path)
    steps = [{"primitive_symbol": "compute_pick_targets", "params": {}}]
    with pytest.raises(PrimitiveCompositionError, match="result_ref type"):
        primitive_composition._validate_parameter(
            {"result_ref": {"step_index": 1, "field_path": "/target_pose"}},
            {"type": "object", "properties": {"x": {"type": "string"}}},
            inputs,
            steps,
        )


def test_shared_target_extractors_preserve_returned_fields_and_legacy_waypoints() -> None:
    from cais_spade_llm.resources.robot.robot_primitives import (
        ROBOT_OBSERVATION_OUTPUT_SCHEMA_MAP,
        _extract_pick_targets_output,
        _extract_place_targets_output,
    )

    orientation = {"qx": 0, "qy": 0, "qz": 0, "qw": 1}
    pick = {
        "success": True,
        "part_name": "medium gear",
        "model_name": "explicit_controller_id",
        **{
            key: 0.2
            for key in (
                "tx",
                "ty",
                "tz",
                "pick_z",
                "travel_z",
                "part_height",
                "tcp_offset_z",
                "pick_tcp_z",
                "start_x",
                "start_y",
                "start_z",
            )
        },
    }
    pick["target_pose"] = {"x": 0.3, "y": 0.4, "z": 0.5, **orientation}
    output, error = _extract_pick_targets_output({}, pick)
    assert error is None
    assert set(ROBOT_OBSERVATION_OUTPUT_SCHEMA_MAP["compute_pick_targets"]) <= output.keys()
    assert output["target_pose"] == pick["target_pose"]
    assert output["approach_pose"] == {"x": 0.2, "y": 0.2, "z": 0.2}
    place = {
        "success": True,
        "part_name": "medium gear",
        "model_name": "explicit_controller_id",
        **{
            key: 0.2
            for key in (
                "slot_x",
                "slot_y",
                "board_top_z",
                "place_z",
                "place_tcp_z",
                "part_height",
                "tcp_offset_z",
                "grasp_tcp_to_part_origin_z",
            )
        },
    }
    baseline, error = _extract_place_targets_output({}, place)
    assert error is None
    assert "pre_insert_pose" not in baseline and "insert_pose" not in baseline
    assert baseline["target_pose"] == {"x": 0.2, "y": 0.2, "z": 0.2}
    place.update(
        {
            "approach_pose": {"x": 0.2, "y": 0.2, "z": 0.4, **orientation},
            "pre_insert_pose": {"x": 0.2, "y": 0.2, "z": 0.3, **orientation},
            "insert_pose": {"x": 0.2, "y": 0.2, "z": 0.1, **orientation},
            "insertion_axis_world": {"x": 0, "y": 0, "z": -1},
            "target_reference": {"target_point": "inserted_part_origin"},
            "target_origin_pose": {"x": 0.2, "y": 0.2, "z": 0.1},
            "place_part_origin_z": 0.1,
        }
    )
    place["target_pose"] = deepcopy(place["pre_insert_pose"])
    output, error = _extract_place_targets_output({}, place)
    assert error is None
    assert set(ROBOT_OBSERVATION_OUTPUT_SCHEMA_MAP["compute_place_targets"]) == output.keys()
    for name, value in place.items():
        if name != "success":
            assert output[name] == value
    assert output["target_pose"] != output["insert_pose"]


@pytest.mark.parametrize("frame", ["world", "robot_base"])
def test_binding_assessment_reads_only_selected_evidence_and_defers_identifier(frame: str) -> None:
    from cais_spade_llm.spec2primitives.agents.ra.parameter_binding import assess_parameter_bindings

    catalog = {
        entry["primitive_symbol"]: entry
        for entry in in_process_robot_agent._phase_5_1_primitive_catalog(
            _declared_geometry_catalog()
        )
    }
    cad = {"record_type": "CADMeshRecord", "coordinate_frame": "CAD_local", "height_m": 0.02}
    reads = []

    def read(ref: str, pointer: str) -> dict:
        reads.append((ref, pointer))
        assert (ref, pointer) == ("selected_cad.json", "")
        return cad

    def result_schema(ref: dict) -> dict:
        assert ref == {"step_index": 1, "field_path": "/model_name"}
        return catalog["compute_pick_targets"]["result_schemas"]["model_name"]

    steps = [
        {
            "primitive_symbol": "compute_pick_targets",
            "params": {
                "product_geometry": {
                    "part_height_m": {
                        "value_ref": {"record_ref": "selected_cad.json", "field_path": "/height_m"}
                    },
                }
            },
        },
        {
            "primitive_symbol": "grasp_part",
            "params": {
                "model_name": {"result_ref": {"step_index": 1, "field_path": "/model_name"}},
            },
        },
        {
            "primitive_symbol": "compute_place_targets",
            "params": {
                "product_geometry": {
                    "value_ref": {"record_ref": "selected_cad.json", "field_path": ""}
                },
            },
        },
    ]
    before = deepcopy(steps)
    issues = assess_parameter_bindings(
        steps,
        catalog,
        {"motion_context": {"frame_id": frame, "ee_link": "ee", "tcp_link": "tcp"}},
        read_evidence=read,
        result_schema=result_schema,
    )
    assert steps == before
    assert reads == [("selected_cad.json", "")]
    assert any(
        issue["step_index"] == 3
        and issue["parameter_path"] == "/product_geometry"
        and issue["status"] == "incompatible"
        for issue in issues
    )
    assert any(
        issue["parameter_path"] == "/product_geometry/part_height_m"
        and issue["status"] == "incompatible"
        for issue in issues
    )
    assert any(
        issue["step_index"] == 2
        and issue["parameter_path"] == "/model_name"
        and issue["status"] == "deferred"
        for issue in issues
    )
    assert not any(issue["step_index"] == 2 and issue["status"] == "missing" for issue in issues)
    assert any(
        issue["parameter_path"] == "/robot_state/motion_context/frame_id"
        and issue["status"] == "incompatible"
        for issue in issues
    ) == (frame == "robot_base")


@pytest.mark.parametrize("mismatched", [False, True])
def test_selected_ra_adapter_owns_context_message_delivery_without_model_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatched: bool) -> None:
    from cais_spade_llm.spec2primitives.agents.pa import primitive_context_messages
    from cais_spade_llm.spec2primitives.agents.ra.refinement_records import append_record
    runtime, agent, _ = _prepare_composition(tmp_path)
    inputs = primitive_composition._load_inputs(tmp_path)
    reference = append_record(tmp_path, tmp_path / "composition/refinement_runs/run_0001/pa_0001", "request.json", {
        "record_type": "PrimitiveContextRequest",
        "assignment_fingerprint": "different" if mismatched else inputs.assignment.fingerprint,
    })
    calls = []
    async def deliver(selected: Any, **kwargs: Any) -> Any:
        calls.append((selected, kwargs))
        return {"record_type": "PrimitiveContextResponse", "request_ref": kwargs["request_ref"]}
    monkeypatch.setattr(primitive_context_messages, "request_context_message", deliver)
    async def request() -> Any:
        return await runtime.request_primitive_context(inputs.assignment, root=tmp_path, recipient="fixture-pa@localhost",
                                                       request_ref=reference, thread="fixture-thread", deadline=123.0)
    if mismatched:
        with pytest.raises(RAContextHandoffError, match="assignment"):
            asyncio.run(request())
        assert calls == []
    else:
        assert asyncio.run(request())["request_ref"] == reference
        assert calls[0][0] is agent and calls[0][1]["thread"] == "fixture-thread"
    assert agent.composition_calls == []
