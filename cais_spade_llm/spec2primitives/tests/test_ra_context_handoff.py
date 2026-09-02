"""Tests for selected-RA context capture and structural primitive drafts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from cais_spade_llm.spec2primitives.adapters import in_process_robot_agent
from cais_spade_llm.spec2primitives.adapters.dual_gazebo import DUAL_GAZEBO_NAME
from cais_spade_llm.spec2primitives.adapters.in_process_robot_agent import (
    InProcessRobotAgentCompositionRuntime,
)
from cais_spade_llm.spec2primitives.agents.ra.context_handoff import (
    RAContextHandoffError,
    SelectedRAAssignmentEnvelope,
    activate_selected_ra_context,
    read_phase_5_1_diagnostic,
)
from cais_spade_llm.spec2primitives.agents.ra.primitive_draft import (
    PrimitiveDraftError,
    author_primitive_program_draft,
    read_phase_5_2_diagnostic,
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
        draft_response: dict[str, object] | None = None,
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
        self.draft_response = draft_response or {
            "draft_status": "proposed",
            "primitive_symbols": ["compute_pick_targets", "grasp_part"],
            "unsupported_reason": None,
        }
        self.draft_calls: list[dict[str, object]] = []
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
        self.draft_calls.append(
            {
                "prompt": prompt,
                "response_format": deepcopy(response_format),
                "tools": tools,
                "max_tool_rounds": max_tool_rounds,
                "include_agent_instructions": include_agent_instructions,
            }
        )
        return deepcopy(self.draft_response)


class _StructuralDraftRuntime:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def author_structural_draft(
        self,
        assignment: SelectedRAAssignmentEnvelope,
        *,
        prompt: str,
        response_format: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append(
            {
                "assignment": assignment,
                "prompt": prompt,
                "response_format": deepcopy(response_format),
            }
        )
        return deepcopy(self.response)


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
    return {
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


def test_plan_only_adapter_contacts_only_the_pa_selected_robot_agent() -> None:
    xarm6 = _LiveRobotAgent(jid="xarm6@localhost")
    ur5e = _LiveRobotAgent(jid="ur5e@localhost")
    host = _LiveRobotAgentHost([xarm6, ur5e])
    moveit = _PlanOnlyRuntime()
    runtime = InProcessRobotAgentCompositionRuntime(
        host,
        moveit_plan_only_runtime=moveit,
    )

    response = asyncio.run(
        runtime.validate_plan_only_allocation(_plan_only_request("ur5e@localhost"))
    )

    assert response["status"] == "accepted"
    assert xarm6.feasibility_calls == []
    assert len(ur5e.feasibility_calls) == 2
    assert [call["grounded_action"]["target"]["pose"] for call in ur5e.feasibility_calls] == [
        {"x": 0.1, "y": 0.2, "z": 0.3},
        {"x": 0.4, "y": 0.5, "z": 0.6},
    ]
    assert len(moveit.requests) == 1
    assert moveit.requests[0]["resource_jid"] == "ur5e@localhost"
    assert moveit.requests[0]["motion_executed"] is False


def test_robot_agent_precheck_rejection_never_invokes_moveit() -> None:
    selected = _LiveRobotAgent(feasibility_allowed=False)
    moveit = _PlanOnlyRuntime()
    runtime = InProcessRobotAgentCompositionRuntime(
        _LiveRobotAgentHost([selected]),
        moveit_plan_only_runtime=moveit,
    )

    response = asyncio.run(
        runtime.validate_plan_only_allocation(_plan_only_request("xarm6@localhost"))
    )

    assert response["status"] == "rejected"
    assert response["current_state"]["status"] == "rejected"
    assert response["desired_state"]["status"] == "rejected"
    assert moveit.requests == []


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
    assert detect_parts["conditions"] == {"controller_ready": True}
    assert detect_parts["effects"] == {"part_observed": True}
    assert raw_catalog == _raw_live_catalog()


def test_in_process_ra_adapter_asks_exact_robot_agent_for_structural_draft(
    tmp_path: Path,
) -> None:
    completion = persist_native_completion_fixture(tmp_path).to_record()
    agent = _LiveRobotAgent()
    host = _LiveRobotAgentHost([agent])
    runtime = InProcessRobotAgentCompositionRuntime(host)
    asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    draft = asyncio.run(author_primitive_program_draft(runtime, tmp_path))

    assert draft.path.name == "draft_0001.json"
    assert draft.record["structural_steps"] == [
        {"step_index": 1, "primitive_symbol": "compute_pick_targets"},
        {"step_index": 2, "primitive_symbol": "grasp_part"},
    ]
    assert len(agent.draft_calls) == 1
    call = agent.draft_calls[0]
    assert call["tools"] is None
    assert call["max_tool_rounds"] == 0
    assert call["include_agent_instructions"] is False
    assert "assemble medium gear" in str(call["prompt"])
    assert "The medium gear is assembled as requested." in str(call["prompt"])
    composition_input = _composition_input_from_prompt(call["prompt"])
    projection = composition_input["ontology_projection"]
    final_view_path = sorted((tmp_path / "products/grounding/product_context").glob("view_*.json"))[
        -1
    ]
    final_view = _read_json(final_view_path)
    assert set(projection) == {"tbox_fingerprint", "abox_fingerprint", "assertions"}
    assert projection["tbox_fingerprint"] == completion["tbox_fingerprint"]
    assert projection["abox_fingerprint"] == completion["abox_fingerprint"]
    assert projection["assertions"] == final_view["assertions"]
    serialized_projection = json.dumps(projection, sort_keys=True)
    for excluded_field in (
        "typed_bindings",
        "uncertainty",
        "attempted_evidence",
        "translated_location_m",
    ):
        assert excluded_field not in serialized_projection
    response_schema = call["response_format"]
    assert response_schema["schema"]["type"] == "object"
    assert response_schema["schema"]["properties"]["primitive_symbols"]["items"]["enum"] == (
        _EXPECTED_XARM6_SYNTHESIS_SYMBOLS
    )


def test_phase_5_2_rejects_assignment_inconsistent_ontology_projection(
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

    asyncio.run(activate_selected_ra_context(_AssignedContextRuntime(), tmp_path))
    runtime = _StructuralDraftRuntime(
        {
            "draft_status": "proposed",
            "primitive_symbols": ["detect_parts"],
            "unsupported_reason": None,
        }
    )

    with pytest.raises(PrimitiveDraftError, match="runsOnResource"):
        asyncio.run(author_primitive_program_draft(runtime, tmp_path))

    assert runtime.calls == []
    assert not (tmp_path / "composition/primitive_program_drafts").exists()


def test_phase_5_2_persists_one_unbound_draft_per_context_pair(
    tmp_path: Path,
) -> None:
    persist_native_completion_fixture(tmp_path)
    context_runtime = _AssignedContextRuntime()
    first_context = asyncio.run(activate_selected_ra_context(context_runtime, tmp_path))
    phase_4_path = tmp_path / "interaction_record/context_completion_0001.json"
    phase_4_bytes = phase_4_path.read_bytes()
    state_bytes = first_context.robot_state_path.read_bytes()
    catalog_bytes = first_context.primitive_catalog_path.read_bytes()
    draft_runtime = _StructuralDraftRuntime(
        {
            "draft_status": "proposed",
            "primitive_symbols": ["detect_parts", "move_pose", "detect_parts"],
            "unsupported_reason": None,
        }
    )

    first = asyncio.run(author_primitive_program_draft(draft_runtime, tmp_path))
    diagnostic = read_phase_5_2_diagnostic(tmp_path).to_view()

    assert first.path.relative_to(tmp_path).as_posix() == (
        "composition/primitive_program_drafts/draft_0001.json"
    )
    assert diagnostic["status"] == "draft_authored"
    assert diagnostic["primitive_symbols"] == [
        "detect_parts",
        "move_pose",
        "detect_parts",
    ]
    delivered_input = _composition_input_from_prompt(draft_runtime.calls[0]["prompt"])
    composition_input = diagnostic["composition_input"]
    assert composition_input == delivered_input
    assert set(composition_input) == {
        "target_feature",
        "selected_resource",
        "ontology_projection",
        "robot_state",
        "primitive_catalog",
        "grounded_context",
    }
    assert composition_input["robot_state"] == first_context.robot_state.robot_state
    assert composition_input["primitive_catalog"] == list(
        first_context.primitive_catalog.primitive_catalog
    )
    projection = composition_input["ontology_projection"]
    assert isinstance(projection, dict)
    assert projection["assertions"]
    grounded_context = composition_input["grounded_context"]
    assert isinstance(grounded_context, dict)
    assert set(grounded_context) == {"typed_records"}
    target_feature = composition_input["target_feature"]
    assert target_feature["product_requirement"] == "assemble medium gear"
    assert target_feature["desired_state"]["statement"]["text"] == (
        "The medium gear is assembled as requested."
    )
    assert target_feature["current_state"]["state_values"] == []
    assert target_feature["desired_state"]["state_values"] == []
    assert target_feature["resolved_state_values"] == []
    assert "task" not in composition_input
    assert all(
        set(record) == {"record_type", "record_ref"} for record in grounded_context["typed_records"]
    )
    serialized_input = json.dumps(composition_input, sort_keys=True)
    for excluded_field in (
        "raw_turtle",
        "typed_bindings",
        "attempted_evidence",
        "parameter_bindings",
    ):
        assert excluded_field not in serialized_input
    serialized = json.dumps(first.to_record())
    assert "primitive_steps" not in serialized
    assert "typed_parameters" not in serialized
    assert "parameter_bindings" not in serialized
    assert "composition_input" not in serialized
    assert "ontology_projection" not in serialized
    assert phase_4_path.read_bytes() == phase_4_bytes
    assert first_context.robot_state_path.read_bytes() == state_bytes
    assert first_context.primitive_catalog_path.read_bytes() == catalog_bytes
    with pytest.raises(PrimitiveDraftError, match="already has"):
        asyncio.run(author_primitive_program_draft(draft_runtime, tmp_path))

    asyncio.run(activate_selected_ra_context(context_runtime, tmp_path))
    assert read_phase_5_2_diagnostic(tmp_path).status == "ready_for_draft"
    second = asyncio.run(author_primitive_program_draft(draft_runtime, tmp_path))
    assert second.path.name == "draft_0002.json"
    assert (
        first.path.read_bytes()
        == json.dumps(first.to_record(), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def test_ra_receives_resolved_target_state_values_without_persisting_them(
    tmp_path: Path,
) -> None:
    persist_native_completion_fixture(
        tmp_path,
        include_target_state_values=True,
    )
    asyncio.run(activate_selected_ra_context(_AssignedContextRuntime(), tmp_path))
    runtime = _StructuralDraftRuntime(
        {
            "draft_status": "proposed",
            "primitive_symbols": ["detect_parts", "move_pose"],
            "unsupported_reason": None,
        }
    )

    draft = asyncio.run(author_primitive_program_draft(runtime, tmp_path))

    composition_input = _composition_input_from_prompt(runtime.calls[0]["prompt"])
    assert "task" not in composition_input
    target_feature = composition_input["target_feature"]
    assert [item["name"] for item in target_feature["desired_state"]["state_values"]] == [
        "specified_finish",
        "specified_coating",
    ]
    resolved_by_name = {item["name"]: item for item in target_feature["resolved_state_values"]}
    assert resolved_by_name["specified_finish"]["resolved_value"] == "matte"
    assert resolved_by_name["specified_coating"]["resolved_value"] == "primer"
    assert resolved_by_name["specified_finish"]["value_ref"]["field_path"] == ("/overview/summary")
    assert resolved_by_name["specified_coating"]["value_ref"]["field_path"] == (
        "/overview/observations/0"
    )
    serialized_draft = json.dumps(draft.to_record(), sort_keys=True)
    assert "target_feature" not in serialized_draft
    assert "specified_finish" not in serialized_draft


def test_phase_5_2_diagnostic_blocks_mismatched_draft_evidence_ref(
    tmp_path: Path,
) -> None:
    persist_native_completion_fixture(tmp_path)
    context = asyncio.run(activate_selected_ra_context(_AssignedContextRuntime(), tmp_path))
    runtime = _StructuralDraftRuntime(
        {
            "draft_status": "proposed",
            "primitive_symbols": ["detect_parts"],
            "unsupported_reason": None,
        }
    )
    draft = asyncio.run(author_primitive_program_draft(runtime, tmp_path))
    assignment_record = _read_json(context.assignment_path)
    altered = draft.to_record()
    altered["robot_state_ref"] = context.assignment_path.relative_to(tmp_path).as_posix()
    altered["robot_state_sha256"] = hashlib.sha256(context.assignment_path.read_bytes()).hexdigest()
    altered["robot_state_fingerprint"] = assignment_record["fingerprint"]
    altered["fingerprint"] = _record_fingerprint(altered)
    _write_json(draft.path, altered)

    diagnostic = read_phase_5_2_diagnostic(tmp_path).to_view()

    assert diagnostic["status"] == "blocked"
    assert diagnostic["draft"] is None
    assert diagnostic["composition_input"] is None
    assert diagnostic["failure"] == (
        "PrimitiveProgramDraft robot_state_ref does not match its active context."
    )
    assert len(runtime.calls) == 1


@pytest.mark.parametrize(
    ("response", "expected_status"),
    [
        (
            {
                "draft_status": "unsupported",
                "primitive_symbols": [],
                "unsupported_reason": "The catalog has no insertion behavior.",
            },
            "unsupported",
        ),
        (
            {
                "draft_status": "proposed",
                "primitive_symbols": ["invented_primitive"],
                "unsupported_reason": None,
            },
            None,
        ),
    ],
)
def test_phase_5_2_handles_unsupported_and_unknown_symbols(
    tmp_path: Path,
    response: dict[str, object],
    expected_status: str | None,
) -> None:
    persist_native_completion_fixture(tmp_path)
    asyncio.run(activate_selected_ra_context(_AssignedContextRuntime(), tmp_path))
    runtime = _StructuralDraftRuntime(response)

    if expected_status is None:
        with pytest.raises(PrimitiveDraftError, match="unknown primitive"):
            asyncio.run(author_primitive_program_draft(runtime, tmp_path))
        assert not (tmp_path / "composition/primitive_program_drafts").exists()
        return

    draft = asyncio.run(author_primitive_program_draft(runtime, tmp_path))
    diagnostic = read_phase_5_2_diagnostic(tmp_path).to_view()
    assert draft.record["structural_steps"] == []
    assert diagnostic["status"] == expected_status
    assert diagnostic["unsupported_reason"] == ("The catalog has no insertion behavior.")
    assert isinstance(diagnostic["composition_input"], dict)


def test_default_xarm6_robot_agent_owns_eight_synthesis_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cais_spade_llm import agent_creator

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
    from cais_spade_llm import agent_creator
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    def _forbidden_controller(_agent: RobotAgent) -> object:
        raise AssertionError("context-only RobotAgent constructed a controller")

    monkeypatch.setattr(agent_creator, "ROBOT_ENV", "gazebo")
    monkeypatch.setattr(agent_creator, "_EXECUTION_MODE_OVERRIDE", "simulation")
    monkeypatch.setattr(RobotAgent, "_build_controller", _forbidden_controller)
    monkeypatch.setitem(agent_creator.ALLOWED_FUNCS, resource_name, set())
    caplog.set_level("INFO", logger=f"agent:{resource_name}")

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
    assert (resource_root / "validation").is_dir()
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
    assert assignment_record["schema_version"] == 3
    assert assignment_record["process_symbol"] == "assembly"
    assert assignment_record["selected_resource_symbol"] == "xarm6"
    assert assignment_record["allocation_label"] == (
        "validated endpoint-motion allocation"
    )
    assert assignment_record["validation_scope"] == "endpoint_motion"
    assert assignment_record["motion_executed"] is False
    assert assignment_record["current_state_evidence"]["evidence_handle"]
    assert assignment_record["desired_state_evidence"]["evidence_handle"]
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
    assert len(validation_records) == 1
    assert _read_json(validation_records[0])["motion_executed"] is False

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
def test_assignment_envelope_v3_rejects_altered_authority_lineage(
    tmp_path: Path,
    variant: str,
) -> None:
    persist_native_completion_fixture(tmp_path)
    context = asyncio.run(activate_selected_ra_context(_AssignedContextRuntime(), tmp_path))
    assignment = _read_json(context.assignment_path)
    if variant == "process":
        assignment["process_symbol"] = "altered_process"
    elif variant == "state":
        assignment["current_state_evidence"]["evidence_handle"] = "altered_evidence"
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
    assert (resource_root / "validation").is_dir()
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
    marker = "\n\nCOMPOSITION_INPUT\n"
    _prefix, separator, serialized = str(value).partition(marker)
    assert separator == marker
    composition_input = json.loads(serialized)
    assert isinstance(composition_input, dict)
    return composition_input
