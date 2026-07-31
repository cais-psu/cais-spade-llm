"""Focused tests for one-step neurosymbolic recovery selection."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.agents.central_controller.outline_macro_safety import (
    project_outline_macro_recovery_aps,
    validate_outline_macro_recovery_safety,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (
    recovery_artifacts,
    recovery_validation_service,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
    multi_turn,
    multi_turn_outline_generation,
    multi_turn_prompts,
)
from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.resources.capability_engine import (
    bind_configured_capabilities,
)
from cais_spade_llm.resources.resource_profile import (
    ResourceProfile,
    register_resource_profile,
)
from cais_spade_llm.resources.robot.robot_tasks import robot_task_registry


def _condition(
    condition_id: str,
    *,
    part_name: str,
    expected: str,
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "kind": "recovery_part_goal",
        "entity_kind": "part",
        "entity": part_name,
        "field": "state",
        "expected": expected,
    }


def _prepared(*conditions: dict[str, Any]) -> dict[str, Any]:
    return {
        "llm_input": {
            "goal_conditions": [deepcopy(row) for row in conditions],
        },
        "tools_catalog": [],
        "recovery_resources": {},
    }


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
    _capability_atomic_transition = ResourceAgent._capability_atomic_transition
    _bind_capabilities = ResourceAgent._bind_capabilities
    _evaluate_capability_transition = (
        ResourceAgent._evaluate_capability_transition
    )
    _enabled_capability_instances = ResourceAgent._enabled_capability_instances
    public_capability_state_projections = (
        ResourceAgent.public_capability_state_projections
    )
    capability_runtime_fact_names = RobotAgent.capability_runtime_fact_names
    capability_runtime_facts = RobotAgent.capability_runtime_facts
    capability_state_valuation = RobotAgent.capability_state_valuation
    _resolve_generated_capability = RobotAgent._resolve_generated_capability
    _generated_successor_from_action_target = (
        RobotAgent._generated_successor_from_action_target
    )

    def __init__(self, *, resource_jid: str, capabilities: dict[str, Any]) -> None:
        self.jid = resource_jid
        self.static_capabilities = deepcopy(capabilities)
        self._configured_capability_declarations = deepcopy(capabilities)
        self.executables = {
            str(event.get("event_name") or ""): (
                lambda **_kwargs: {"success": True}
            )
            for event in capabilities.get("events") or []
            if isinstance(event, dict)
        }


def _enabled_capability_instances(
    harness: _RobotCapabilityHarness,
    *,
    resource_snapshot: dict[str, Any],
    part_contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_snapshot = {
        "resource_jid": str(
            resource_snapshot.get("resource_jid") or harness.jid
        ),
        "gripper_state": (
            "closed" if resource_snapshot.get("held_part") else "open"
        ),
        "carried_entity_location": f"{harness.jid}_gripper",
        **deepcopy(resource_snapshot),
    }
    return harness._enabled_capability_instances(
        resource_snapshot=normalized_snapshot,
        part_contexts=part_contexts,
    )


def _robot_manifest_descriptor(
    *,
    resource_jid: str,
    snapshot: dict[str, Any],
) -> _RobotCapabilityHarness:
    manifest = json.loads(
        Path(
            "cais_spade_llm/initialization/resources/robot_ur5e.json"
        ).read_text(encoding="utf-8")
    )
    configured_capabilities = manifest["ur5e"]["gazebo"][
        "static_capabilities"
    ]
    del snapshot
    return _RobotCapabilityHarness(
        resource_jid=resource_jid,
        capabilities=configured_capabilities,
    )


def _session(*, parts: dict[str, str], candidate_bound: int = 5) -> dict[str, Any]:
    return {
        "recovery_selection_mode": "neurosymbolic",
        "outline_mode": "incremental_candidates_validated",
        "action_horizon": "1",
        "action_horizon_steps": 1,
        "action_horizon_k": 1,
        "candidate_count": "auto",
        "candidate_bound": candidate_bound,
        "candidate_proposal_budget": candidate_bound,
        "accepted_outline_prefix": [],
        "candidate_rejection_feedback": [],
        "candidate_prune_history": {},
        "candidate_revision_targets": [],
        "candidate_revision_state_fingerprint": "",
        "outline_validation_findings": [],
        "pruned_actions": [],
        "selection_revision_count": 0,
        "selection_revision_fingerprint": "",
        "selection_revision_limit": 6,
        "selection_repeated_failure_count": 0,
        "selection_repeated_failure_fingerprint": "",
        "selection_repeated_failure_limit": 3,
        "symbolic_resources": {
            "resource@localhost": {
                "resource_jid": "resource@localhost",
                "resource_state": "idle",
                "current_state": "idle",
                "held_part": None,
            }
        },
        "symbolic_parts": {
            part_name: {
                "part_name": part_name,
                "part_state": state,
                "current_state": state,
                "part_location": "station",
                "current_location": "station",
                "part_holder_resource_jid": None,
                "current_holder_resource_jid": None,
            }
            for part_name, state in parts.items()
        },
        "projected_safety_dfa_states": {},
    }


def _event(
    event_name: str,
    *,
    part_name: str,
    start_state: str,
    end_state: str,
) -> dict[str, Any]:
    return {
        "outline_id": f"outline_{event_name}",
        "event_name": event_name,
        "resource_jid": "resource@localhost",
        "part_name": part_name,
        "expected_start_state": {
            "resource_state": "idle",
            "held_part": None,
            "part_state": start_state,
            "part_location": "station",
        },
        "expected_end_state": {
            "resource_state": "idle",
            "held_part": None,
            "part_state": end_state,
            "part_location": "station",
        },
        "rationale": "candidate rationale",
    }


def _evaluation(
    candidate_index: int,
    *,
    session_state: dict[str, Any],
    part_name: str,
    end_state: str,
    valid: bool = True,
) -> dict[str, Any]:
    projected_parts = deepcopy(session_state["symbolic_parts"])
    projected_parts[part_name]["part_state"] = end_state
    projected_parts[part_name]["current_state"] = end_state
    return {
        "candidate_index": candidate_index,
        "valid": valid,
        "validation_findings": [],
        "validation_stages": [
            {"validator_role": "PA", "status": "passed"},
            {"validator_role": "RA", "status": "passed"},
            {"validator_role": "CCA", "status": "passed"},
        ],
        "projected_symbolic_resources": deepcopy(
            session_state["symbolic_resources"]
        ),
        "projected_symbolic_parts": projected_parts,
        "safety_dfa_states_before": {},
        "safety_dfa_states_after": {},
    }


def test_mode_specific_candidate_response_schemas() -> None:
    pure_schema = multi_turn_prompts._outline_candidates_response_schema(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
        candidate_bound=5,
    )["schema"]
    neuro_schema = multi_turn_prompts._outline_candidates_response_schema(
        recovery_selection_mode="neurosymbolic",
        action_horizon="1",
        candidate_bound=5,
    )["schema"]

    assert pure_schema["properties"]["candidate_events"]["minItems"] == 3
    assert pure_schema["properties"]["candidate_events"]["maxItems"] == 3
    assert "selected_candidate_index" in pure_schema["required"]
    assert neuro_schema["properties"]["candidate_events"]["minItems"] == 1
    assert neuro_schema["properties"]["candidate_events"]["maxItems"] == 5
    assert "selected_candidate_index" not in neuro_schema["properties"]
    assert "selected_candidate_index" not in neuro_schema["required"]


def test_robot_task_program_backward_relevance_and_forward_enabledness() -> None:
    resource_jid = "ur5e@localhost"
    part_name = "LG"
    destination = "assembly_board-v1"
    carried_location = f"{resource_jid}_gripper"
    descriptor = _robot_manifest_descriptor(
        resource_jid=resource_jid,
        snapshot={
            "current_state": "picked",
            "current_location": "prusa-mk4-2",
            "held_part": part_name,
        },
    )
    goal_conditions = (
        {
            "condition_id": "goal_lg_state",
            "kind": "goal_part_state",
            "entity_kind": "part",
            "entity": part_name,
            "field": "state",
            "expected": "assembled",
        },
        {
            "condition_id": "goal_lg_location",
            "kind": "goal_part_location",
            "entity_kind": "part",
            "entity": part_name,
            "field": "location",
            "expected": destination,
        },
    )
    prepared = _prepared(*goal_conditions)
    session_state = _session(parts={})
    session_state["symbolic_resources"] = {
        resource_jid: {
            "resource_jid": resource_jid,
            "resource_type": "robot",
            "resource_state": "picked",
            "current_state": "picked",
            "resource_location": "prusa-mk4-2",
            "current_location": "prusa-mk4-2",
            "held_part": part_name,
            "carried_entity_location": carried_location,
        }
    }
    session_state["symbolic_parts"] = {
        part_name: {
            "part_name": part_name,
            "part_state": "in_gripper",
            "current_state": "in_gripper",
            "part_location": carried_location,
            "current_location": carried_location,
            "part_holder_resource_jid": resource_jid,
            "current_holder_resource_jid": resource_jid,
            "goal_location": destination,
        }
    }
    session_state["latest_enabled_capability_results"] = [
        {
            **row,
            "allowed": True,
            "physical_feasibility": {"allowed": True},
        }
        for row in _enabled_capability_instances(
            descriptor,
            resource_snapshot=session_state["symbolic_resources"][
                resource_jid
            ],
            part_contexts=[
                session_state["symbolic_parts"][part_name]
            ],
        )
    ]

    unresolved = multi_turn_outline_generation._unresolved_condition_ids(
        session_state=session_state,
        prepared_recovery_request=prepared,
    )
    enabled = (
        multi_turn_outline_generation._symbolically_enabled_recovery_event_instances(
            session_state=session_state,
            prepared_recovery_request=prepared,
            unresolved_condition_ids=unresolved,
        )
    )

    assert [row["task"]["event_name"] for row in enabled] == [
        "place_approach"
    ]
    approach = deepcopy(enabled[0]["task"])
    assert approach["expected_end_state"]["resource_state"]["location"] == (
        destination
    )

    multi_turn._apply_task_effects_to_symbolic_state(approach, session_state)
    session_state["latest_enabled_capability_results"] = [
        {
            **row,
            "allowed": True,
            "physical_feasibility": {"allowed": True},
        }
        for row in _enabled_capability_instances(
            descriptor,
            resource_snapshot=session_state["symbolic_resources"][
                resource_jid
            ],
            part_contexts=[
                session_state["symbolic_parts"][part_name]
            ],
        )
    ]
    unresolved = multi_turn_outline_generation._unresolved_condition_ids(
        session_state=session_state,
        prepared_recovery_request=prepared,
    )
    enabled_after_approach = (
        multi_turn_outline_generation._symbolically_enabled_recovery_event_instances(
            session_state=session_state,
            prepared_recovery_request=prepared,
            unresolved_condition_ids=unresolved,
        )
    )
    assert "place_insert" in {
        row["task"]["event_name"] for row in enabled_after_approach
    }


def test_arbitrary_bridge_labels_progress_only_through_structural_enabledness() -> None:
    resource_jid = "ur5e@localhost"
    part_name = "LG"
    destination = "assembly_board-v1"
    carried_location = f"{resource_jid}_gripper"
    prepared = _prepared(
        {
            "condition_id": "goal_lg_state",
            "kind": "goal_part_state",
            "entity_kind": "part",
            "entity": part_name,
            "field": "state",
            "expected": "assembled",
        },
        {
            "condition_id": "goal_lg_location",
            "kind": "goal_part_location",
            "entity_kind": "part",
            "entity": part_name,
            "field": "location",
            "expected": destination,
        },
    )
    session_state = _session(parts={})
    session_state["symbolic_resources"] = {
        resource_jid: {
            "resource_jid": resource_jid,
            "resource_type": "robot",
            "resource_state": "idle",
            "current_state": "idle",
            "resource_location": "prusa-mk4-2",
            "current_location": "prusa-mk4-2",
            "held_part": None,
        }
    }
    session_state["symbolic_parts"] = {
        part_name: {
            "part_name": part_name,
            "part_state": "misplaced",
            "current_state": "misplaced",
            "part_location": "prusa-mk4-2",
            "current_location": "prusa-mk4-2",
            "part_holder_resource_jid": None,
            "current_holder_resource_jid": None,
            "goal_location": destination,
        }
    }
    bridge = {
        "outline_id": "opaque_bridge",
        "event_name": "unrelated_authored_token",
        "resource_jid": resource_jid,
        "part_name": part_name,
        "expected_start_state": {},
        "expected_end_state": {
            "resource_state": "picked",
            "resource_location": "prusa-mk4-2",
            "held_part": part_name,
            "part_state": "in_gripper",
            "part_location": carried_location,
        },
        "rationale": "Apply the exact proposed state facts.",
    }
    projected = deepcopy(session_state)
    multi_turn._apply_task_effects_to_symbolic_state(bridge, projected)
    evaluation = {
        "candidate_index": 0,
        "valid": True,
        "projected_symbolic_resources": deepcopy(
            projected["symbolic_resources"]
        ),
        "projected_symbolic_parts": deepcopy(projected["symbolic_parts"]),
            "safety_dfa_states_before": {},
            "safety_dfa_states_after": {},
            "agent_filtered_enabledness": True,
            "admissible_recovery_enabled_event_ids_before": [
                "pick_approach"
            ],
            "admissible_recovery_enabled_event_ids_after": [
                "place_approach"
            ],
        }

    selected = multi_turn_outline_generation._apply_neurosymbolic_comparison(
        candidate_sequences=[
            {"candidate_index": 0, "surface_events": [bridge]}
        ],
        candidate_evaluations=[evaluation],
        session_state=session_state,
        prepared_recovery_request=prepared,
    )

    assert selected == [evaluation["candidate_id"]]
    assert projected["symbolic_resources"][resource_jid][
        "resource_state"
    ] == "picked"
    assert projected["symbolic_parts"][part_name]["part_state"] == "in_gripper"
    assert len(evaluation["selection_evidence"][
        "newly_enabled_recovery_event_ids"
    ]) == 1
    assert "place_approach" in evaluation["selection_evidence"][
        "newly_enabled_recovery_event_ids"
    ][0]


def test_modeled_continuation_bypasses_outline_llm_and_keeps_program_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_state = _session(parts={"LG": "staged"}, candidate_bound=3)
    session_state["candidate_count"] = 3
    session_state["modeled_continuation_binding"] = {
        "resource_jid": "resource@localhost",
        "part_name": "LG",
    }
    prepared = _prepared(
        _condition("goal_LG", part_name="LG", expected="assembled")
    )
    modeled_task = {
        "outline_id": "enabledness_private",
        "event_name": "registered_transition",
        "resource_jid": "resource@localhost",
        "part_name": "LG",
        "expected_start_state": {
            "resource_state": "holding",
            "held_part": "LG",
            "part_state": "staged",
            "part_location": "resource@localhost_gripper",
        },
        "expected_end_state": {
            "resource_state": "positioned",
            "held_part": "LG",
            "part_state": "in_transit",
            "part_location": "resource@localhost_gripper",
        },
        "rationale": "Modeled continuation.",
    }
    modeled_steps = [
        {
            "primitive": "compute_place_targets",
            "params": {
                "part_name": "<PART>",
                "destination_location": "<DESTINATION_LOCATION>",
            },
        }
    ]

    async def _enabledness(**_kwargs: Any) -> dict[str, Any]:
        return {
            "admissible_event_ids": ["private_event_id"],
            "enabled_capability_results": [
                {
                    "event_id": "private_event_id",
                    "resource_jid": "resource@localhost",
                    "part_name": "LG",
                    "task": deepcopy(modeled_task),
                    "recovery_visible_steps": deepcopy(modeled_steps),
                }
            ],
        }

    monkeypatch.setattr(
        multi_turn_outline_generation,
        "_agent_filtered_recovery_enabledness",
        _enabledness,
    )

    async def accept_modeled_candidate(**kwargs: Any) -> tuple[str, dict[str, Any]]:
        working_state = kwargs["session_state"]
        parsed_response = kwargs["parsed_response"]
        candidate = deepcopy(parsed_response["candidate_events"][0])
        assert "expected_start_state" not in candidate
        committed = {
            **candidate,
            "outline_id": "RECOVERY_SEQ1",
            "llm_outline_id": candidate["outline_id"],
        }
        working_state["accepted_outline_prefix"] = [deepcopy(committed)]
        return "need_next_task", {
            "selected_candidate_task": candidate,
            "selected_transition": deepcopy(committed),
            "selected_transition_sequence": [deepcopy(committed)],
            "next_transition": deepcopy(committed),
            "nondominated_candidate_ids": ["configured_continuation"],
            "tie_representative_evidence": {"used": False},
            "selection_evidence": {
                "newly_admissible_goal_recovery_event_ids": [
                    "private_goal_event"
                ]
            },
        }

    monkeypatch.setattr(
        multi_turn_outline_generation,
        "_handle_outline_phase",
        accept_modeled_candidate,
    )

    result = asyncio.run(
        multi_turn_outline_generation._try_handle_modeled_continuation(
            session_state=session_state,
            prepared_recovery_request=prepared,
            planner=object(),
        )
    )

    assert result is not None
    decision, turn_entry, _parsed_response = result
    assert decision == "need_next_task"
    assert turn_entry["candidate_source"] == "robot_task_program"
    assert turn_entry["llm_called"] is False
    assert session_state["candidate_count"] == 3
    accepted = session_state["accepted_outline_prefix"][0]
    assert accepted["candidate_source"] == "robot_task_program"
    assert accepted["modeled_task_steps"] == modeled_steps


def test_modeled_continuation_returns_to_llm_without_unique_goal_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(
        _condition("goal_LG", part_name="LG", expected="assembled")
    )
    base_session = _session(parts={"LG": "staged"}, candidate_bound=3)
    base_session["modeled_continuation_binding"] = {
        "resource_jid": "resource@localhost",
        "part_name": "LG",
    }

    async def no_enabledness(**_kwargs: Any) -> dict[str, Any]:
        return {
            "admissible_event_ids": [],
            "enabled_capability_results": [],
        }

    monkeypatch.setattr(
        multi_turn_outline_generation,
        "_agent_filtered_recovery_enabledness",
        no_enabledness,
    )
    no_choice_session = deepcopy(base_session)
    no_choice = asyncio.run(
        multi_turn_outline_generation._try_handle_modeled_continuation(
            session_state=no_choice_session,
            prepared_recovery_request=prepared,
            planner=object(),
        )
    )
    assert no_choice is None
    assert no_choice_session["modeled_continuation_binding"] == {}

    tasks = [
        {
            "event_id": f"private_event_{index}",
            "resource_jid": "resource@localhost",
            "part_name": "LG",
            "task": {
                "event_name": f"configured_transition_{index}",
                "resource_jid": "resource@localhost",
                "part_name": "LG",
                "expected_end_state": {
                    "resource_state": "positioned",
                    "part_state": "in_transit",
                },
            },
        }
        for index in range(2)
    ]

    async def tied_enabledness(**_kwargs: Any) -> dict[str, Any]:
        return {
            "admissible_event_ids": [
                str(row["event_id"]) for row in tasks
            ],
            "enabled_capability_results": deepcopy(tasks),
        }

    async def tied_selection(**kwargs: Any) -> tuple[str, dict[str, Any]]:
        assert len(kwargs["parsed_response"]["candidate_events"]) == 2
        return "need_next_task", {
            "nondominated_candidate_ids": ["candidate_a", "candidate_b"],
            "tie_representative_evidence": {"used": True},
            "selection_evidence": {
                "newly_admissible_goal_recovery_event_ids": [
                    "private_goal_event"
                ]
            },
        }

    monkeypatch.setattr(
        multi_turn_outline_generation,
        "_agent_filtered_recovery_enabledness",
        tied_enabledness,
    )
    monkeypatch.setattr(
        multi_turn_outline_generation,
        "_handle_outline_phase",
        tied_selection,
    )
    tied_session = deepcopy(base_session)
    tied_choice = asyncio.run(
        multi_turn_outline_generation._try_handle_modeled_continuation(
            session_state=tied_session,
            prepared_recovery_request=prepared,
            planner=object(),
        )
    )
    assert tied_choice is None
    assert tied_session["modeled_continuation_binding"] == {}
    assert tied_session.get("accepted_outline_prefix") in (None, [])


def test_registered_modeled_steps_ground_exact_part_destination_and_refs() -> None:
    resource_jid = "ur5e@localhost"
    part_name = "LG"
    destination = "assembly_board-v1"
    carried_location = f"{resource_jid}_gripper"
    outline_event = {
        "outline_id": "RECOVERY_SEQ1",
        "event_name": "place_approach",
        "resource_jid": resource_jid,
        "part_name": part_name,
        "candidate_source": "robot_task_program",
        "expected_start_state": {
            "resource_state": "picked",
            "resource_location": "prusa-mk4-2",
            "held_part": part_name,
            "part_state": "in_gripper",
            "part_location": carried_location,
        },
        "expected_end_state": {
            "resource_state": "positioned",
            "resource_location": destination,
            "held_part": part_name,
            "part_state": "in_transit",
            "part_location": carried_location,
        },
        "modeled_task_steps": (
            robot_task_registry()["place_approach"].program.render_recovery_steps()
        ),
    }
    session_state = {
        "accepted_outline_prefix": [deepcopy(outline_event)],
        "accepted_primitive_program": [],
        "symbolic_resources": {
            resource_jid: {
                "resource_jid": resource_jid,
                "resource_state": "positioned",
                "current_state": "positioned",
                "resource_location": destination,
                "current_location": destination,
                "held_part": part_name,
            }
        },
        "symbolic_parts": {
            part_name: {
                "part_name": part_name,
                "part_state": "in_transit",
                "current_state": "in_transit",
                "part_location": carried_location,
                "current_location": carried_location,
                "part_holder_resource_jid": resource_jid,
                "current_holder_resource_jid": resource_jid,
            }
        },
        "observation_store": {},
    }
    prepared = {
        "recovery_resources": {
            resource_jid: {
                "recovery_snapshot": {
                    "resource_jid": resource_jid,
                    "resource_type": "robot",
                    "current_state": "picked",
                    "current_location": "prusa-mk4-2",
                    "held_part": part_name,
                },
                "static_capabilities": {},
            }
        },
        "grounding_context": {
            "parts": {
                part_name: {
                    "target": {
                        "location": destination,
                        "model_name": "gear_large",
                    }
                }
            }
        },
    }

    primitive_steps, unresolved = multi_turn._ground_modeled_task_steps(
        session_state=session_state,
        prepared_recovery_request=prepared,
        outline_event=outline_event,
    )

    assert unresolved == []
    assert primitive_steps[0] == {
        "primitive": "compute_place_targets",
        "params": {
            "part_name": part_name,
            "destination_location": destination,
        },
    }
    assert primitive_steps[1]["params"]["x"] == {
        "context_ref": "event_facts.place_targets.LG.approach_pose.x"
    }

    batch_result = asyncio.run(
        multi_turn._run_resource_primitive_batch(
            planner=object(),
            prepared_recovery_request=prepared,
            session_state=session_state,
            resource_jid=resource_jid,
            assigned_outline_events=[outline_event],
        )
    )
    assert batch_result["decision"] == "draft_ready"
    assert batch_result["primitive_events"] == [
        {
            "outline_id": "RECOVERY_SEQ1",
            "resource_jid": resource_jid,
            "primitive_steps": primitive_steps,
        }
    ]


def test_capability_binding_uses_direct_static_declarations() -> None:
    class _ConfiguredResource:
        jid = "configured@localhost"
        agent_name = "configured@localhost"
        static_capabilities = {
            "state_variables": {
                "job_state": {
                    "scope": "resource",
                    "domain": ["paused", "running"],
                }
            },
            "events": [
                {
                    "event_name": "continue_job",
                    "controllable": True,
                    "observable": True,
                    "guards": {"job_state": {"equals": "paused"}},
                    "updates": {"job_state": {"set": "running"}},
                }
            ],
        }

    configured = _ConfiguredResource()
    descriptor = bind_configured_capabilities(
        configured,
    )
    assert [
        event["event_name"] for event in descriptor["events"]
    ] == ["continue_job"]
    assert "current_valuation" not in descriptor
    assert "configured_capabilities_fingerprint" not in descriptor

    configured.static_capabilities = {}
    assert bind_configured_capabilities(
        configured,
    ) == {}


def test_printer_transition_enables_ra_declared_continuation_event() -> None:
    session_state = _session(parts={"P": "faulted"})
    session_state["symbolic_resources"] = {
        "printer@localhost": {
            "resource_jid": "printer@localhost",
            "resource_state": "printing",
            "current_state": "printing",
            "job_state": "printing",
        }
    }
    prepared = _prepared(_condition("goal_P", part_name="P", expected="restored"))
    event = {
        "outline_id": "printer_transition",
        "event_name": "authored_printer_symbol",
        "resource_jid": "printer@localhost",
        "expected_start_state": {
            "resource_state": "printing",
            "current_state": "printing",
            "job_state": "printing",
        },
        "expected_end_state": {
            "resource_state": "paused",
            "current_state": "paused",
            "job_state": "paused",
        },
        "rationale": "apply exact printer variables",
    }
    projected = deepcopy(session_state)
    multi_turn._apply_task_effects_to_symbolic_state(event, projected)
    evaluation = {
        "candidate_index": 0,
        "valid": True,
        "projected_symbolic_resources": deepcopy(projected["symbolic_resources"]),
        "projected_symbolic_parts": deepcopy(projected["symbolic_parts"]),
        "safety_dfa_states_before": {},
        "safety_dfa_states_after": {},
        "agent_filtered_enabledness": True,
        "admissible_recovery_enabled_event_ids_before": ["pause_job"],
        "admissible_recovery_enabled_event_ids_after": ["resume_job"],
    }

    selected = multi_turn_outline_generation._apply_neurosymbolic_comparison(
        candidate_sequences=[{"candidate_index": 0, "surface_events": [event]}],
        candidate_evaluations=[evaluation],
        session_state=session_state,
        prepared_recovery_request=prepared,
    )

    assert selected == [evaluation["candidate_id"]]
    assert evaluation["selection_evidence"][
        "newly_enabled_recovery_event_ids"
    ] == ["resume_job"]


def test_cca_returns_admissible_nominal_reentry_event_id() -> None:
    event_id = (
        '{"function_name":"resume_job","part_name":"",'
        '"resource_jid":"printer@localhost","task_id":"PRINT_T2"}'
    )
    resources = {
        "printer@localhost": {
            "resource_jid": "printer@localhost",
            "current_state": "paused",
        }
    }

    safety_result = validate_outline_macro_recovery_safety(
        task={
            "outline_id": "candidate",
            "resource_jid": "printer@localhost",
            "expected_end_state": {"resource_state": "paused"},
        },
        signature={"task_kind": "resource_action", "changes_part_world": False},
        pre_resources=resources,
        pre_parts={},
        projected_resources=resources,
        projected_parts={},
        llm_input={
            "loaded_safety_rules": [],
            "nominal_reentry_events": [
                {
                    "event_id": event_id,
                    "task": {"outline_id": "PRINT_T2"},
                    "signature": {"task_kind": "resource_action"},
                }
            ],
        },
        safety_dfa_states_before=None,
    )
    assert safety_result["admissible_nominal_reentry_event_ids"] == [event_id]


def test_cca_classifies_typed_locations_from_configured_area_geometry() -> None:
    rule = {
        "id": "SAFE_AREA",
        "ap_scope": "recovery",
        "dfa_dot": (
            'digraph { init -> 1; 1 -> 1 [label="ap_area"]; }'
        ),
        "recovery_aps": [
            {
                "label": "ap_area",
                "full": (
                    "ap_state/process/any/resource/positioned/"
                    "destination=protected_area"
                ),
                "selector": {
                    "mode": "resource_in_destination",
                    "resource": "resource",
                    "part": "any",
                    "destination": "protected_area",
                },
            }
        ],
    }
    llm_input = {
        "loaded_safety_rules": [rule],
        "public_locations": [
            {
                "resource_jid": "resource@localhost",
                "locations": [
                    {
                        "location": "protected_area",
                        "area": {
                            "frame": "map",
                            "units": "grid",
                            "bounds": {
                                "u": {"min": -1.0, "max": 1.0},
                                "v": {"min": -2.0, "max": 2.0},
                            },
                        },
                    }
                ],
            }
        ],
    }
    inside = {
        "resource_jid": "resource@localhost",
        "current_state": "idle",
        "current_location": {
            "frame": "map",
            "units": "grid",
            "u": 0.25,
            "v": 0.5,
        },
    }
    outside = {
        **inside,
        "current_location": {
            "frame": "map",
            "units": "grid",
            "u": 2.0,
            "v": 0.5,
        },
    }

    projection = project_outline_macro_recovery_aps(
        task={
            "outline_id": "leave_area",
            "event_name": "generated_event",
            "resource_jid": "resource@localhost",
        },
        signature={
            "task_kind": "resource_action",
            "changes_part_world": False,
        },
        pre_resources={"resource@localhost": inside},
        pre_parts={},
        projected_resources={"resource@localhost": outside},
        projected_parts={},
        llm_input=llm_input,
    )

    assert projection["current_state_aps"] == ["ap_area"]
    assert projection["predicted_state_aps"] == []


def test_strict_nominal_reentry_expansion_counts_as_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_state = _session(parts={"P": "waiting", "Q": "faulted"})
    session_state["symbolic_parts"]["P"]["current_location"] = "buffer_a"
    prepared = _prepared(
        _condition("goal_Q", part_name="Q", expected="restored")
    )
    event = {
        "outline_id": "move_P_to_reentry_location",
        "event_name": "opaque_authored_event",
        "resource_jid": "resource@localhost",
        "part_name": "P",
        "expected_start_state": {
            "resource_state": "idle",
            "held_part": None,
            "part_state": "waiting",
            "part_location": "buffer_a",
        },
        "expected_end_state": {
            "resource_state": "idle",
            "held_part": None,
            "part_state": "waiting",
            "part_location": "buffer_b",
        },
        "rationale": "change one exact successor fact",
    }
    projected = deepcopy(session_state)
    multi_turn._apply_task_effects_to_symbolic_state(event, projected)
    evaluation = {
        "candidate_index": 0,
        "valid": True,
        "projected_symbolic_resources": deepcopy(projected["symbolic_resources"]),
        "projected_symbolic_parts": deepcopy(projected["symbolic_parts"]),
        "safety_dfa_states_before": {},
        "safety_dfa_states_after": {},
        "agent_filtered_enabledness": True,
        "admissible_recovery_enabled_event_ids_before": ["recovery_event"],
        "admissible_recovery_enabled_event_ids_after": ["recovery_event"],
        "admissible_nominal_reentry_event_ids_before": [],
        "admissible_nominal_reentry_event_ids_after": ["nominal_reentry_event"],
    }

    monkeypatch.setattr(
        multi_turn_outline_generation,
        "_admissible_nominal_reentry_event_ids",
        lambda **kwargs: (
            {"nominal_reentry_event"}
            if kwargs["session_state"]["symbolic_parts"]["P"].get(
                "current_location"
            )
            == "buffer_b"
            else set()
        ),
    )

    selected = multi_turn_outline_generation._apply_neurosymbolic_comparison(
        candidate_sequences=[{"candidate_index": 0, "surface_events": [event]}],
        candidate_evaluations=[evaluation],
        session_state=session_state,
        prepared_recovery_request=prepared,
    )

    assert selected == [evaluation["candidate_id"]]
    assert evaluation["progressing"] is True
    evidence = evaluation["selection_evidence"]
    assert evidence["newly_enabled_recovery_event_ids"] == []
    assert evidence["newly_enabled_nominal_reentry_event_ids"] == [
        "nominal_reentry_event"
    ]


def test_product_candidate_schema_does_not_embed_resource_domains() -> None:
    schema = multi_turn_prompts._outline_candidates_response_schema(
        recovery_selection_mode="neurosymbolic",
    )["schema"]
    state_schema = schema["$defs"]["outline_state"]
    assert state_schema["additionalProperties"] == {
        "type": ["string", "number", "boolean", "null"],
    }
    assert set(state_schema["properties"]) == {
        "resource_state",
        "part_state",
    }
    assert "enum" not in json.dumps(state_schema)
    assert (
        schema["$defs"]["outline_event"]["properties"]["event_name"]["minLength"]
        == 1
    )
    assert (
        "expected_start_state"
        not in schema["$defs"]["outline_event"]["properties"]
    )

    prepared = {
        "llm_input": {
            "observed_runtime_state": {
                "resources": [
                    {
                        "resource_jid": "printer@localhost",
                        "resource_state": "printing",
                        "current_state": "printing",
                        "job_state": "printing",
                    }
                ]
            },
            "part_facts": [],
        },
        "recovery_resources": {},
    }
    session_state = {
        "symbolic_resources": {
            "printer@localhost": {
                "resource_jid": "printer@localhost",
                "resource_state": "printing",
                "current_state": "printing",
                "job_state": "printing",
            }
        },
        "symbolic_parts": {},
        "public_capability_state_projections": [
                {
                    "resource_jid": "printer@localhost",
                    "state": {
                        "resource_state": {
                            "condition": "printing",
                            "location": None,
                        },
                        "job_state": "printing",
                    },
                }
        ],
        "public_state_values": [
            {
                    "resource_jid": "printer@localhost",
                    "state_values": {
                        "resource_state": {
                            "condition": ["printing", "paused"],
                            "location": [None],
                        },
                        "job_state": ["printing", "paused"],
                    },
                }
        ],
    }
    llm_task = {
        "outline_id": "printer_pause",
        "event_name": "authored_event",
            "resource_jid": "printer@localhost",
            "expected_end_state": {
                "resource_state": {
                    "condition": "paused",
                    "location": None,
                },
                "job_state": "paused",
            },
        "rationale": "exact declared variables",
    }
    valid_task, schema_findings = multi_turn._derive_candidate_outline_task(
        candidate_task=llm_task,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )
    assert schema_findings == []
    assert valid_task is not None
    assert valid_task["expected_start_state"] == {
        "job_state": "printing",
        "resource_state": {
            "condition": "printing",
            "location": None,
        },
    }
    authored_start_task, authored_start_findings = (
        multi_turn._derive_candidate_outline_task(
            candidate_task={
                **llm_task,
                "expected_start_state": {
                    "resource_state": "paused",
                    "job_state": {"not": "a scalar"},
                },
            },
            session_state=session_state,
            prepared_recovery_request=prepared,
        )
    )
    assert authored_start_findings == []
    assert authored_start_task is not None
    assert authored_start_task["expected_start_state"] == {
        "job_state": "printing",
        "resource_state": {
            "condition": "printing",
            "location": None,
        },
    }
    findings, _grounded = multi_turn.validate_recovery_outline_task(
        planner=object(),
        task=valid_task,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )
    assert findings == []

    sequential_state = deepcopy(session_state)
    multi_turn._apply_task_effects_to_symbolic_state(valid_task, sequential_state)
    resume_task, schema_findings = multi_turn._derive_candidate_outline_task(
        candidate_task={
            **llm_task,
            "outline_id": "printer_resume",
            "expected_end_state": {
                "resource_state": {
                    "condition": "printing",
                    "location": None,
                },
                "job_state": "printing",
            },
        },
        session_state=sequential_state,
        prepared_recovery_request=prepared,
        )
    assert schema_findings == []
    assert resume_task is not None
    assert resume_task["expected_start_state"] == {
        "job_state": "paused",
        "resource_state": {
            "condition": "paused",
            "location": None,
        },
    }

    outside_domain = deepcopy(valid_task)
    outside_domain["expected_end_state"]["job_state"] = "maintenance"
    findings, _grounded = multi_turn.validate_recovery_outline_task(
        planner=object(),
        task=outside_domain,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )
    assert findings == []

    invalid_task = deepcopy(valid_task)
    invalid_task["expected_end_state"]["spindle_speed"] = 1
    findings, _grounded = multi_turn.validate_recovery_outline_task(
        planner=object(),
        task=invalid_task,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )
    assert findings == []

    prepared["llm_input"]["observed_runtime_state"]["resources"].append(
        {
            "resource_jid": "robot@localhost",
            "resource_state": "idle",
            "current_state": "idle",
            "gripper_state": "open",
        }
    )
    session_state["symbolic_resources"]["robot@localhost"] = {
        "resource_jid": "robot@localhost",
        "resource_state": "idle",
        "current_state": "idle",
        "gripper_state": "open",
    }
    wrong_resource_field = {
        "outline_id": "robot_job_state",
        "event_name": "authored_event",
        "resource_jid": "robot@localhost",
        "expected_start_state": {
            "resource_state": "idle",
            "job_state": "printing",
        },
        "expected_end_state": {
            "resource_state": "ready",
            "job_state": "paused",
        },
        "rationale": "job_state belongs to another RA descriptor",
    }
    findings, _grounded = multi_turn.validate_recovery_outline_task(
        planner=object(),
        task=wrong_resource_field,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )
    assert findings == []

    private_gripper_field = {
        "outline_id": "robot_private_gripper_field",
        "event_name": "authored_event",
        "resource_jid": "robot@localhost",
        "expected_start_state": {
            "resource_state": "idle",
            "gripper_state": "open",
        },
        "expected_end_state": {
            "resource_state": "ready",
            "gripper_state": "closed",
        },
        "rationale": "gripper_state is private RobotAgent evidence",
    }
    findings, _grounded = multi_turn.validate_recovery_outline_task(
        planner=object(),
        task=private_gripper_field,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )
    assert findings == []

    missing_part_binding = deepcopy(valid_task)
    missing_part_binding["expected_start_state"]["part_quality"] = "unknown"
    missing_part_binding["expected_end_state"]["part_quality"] = "accepted"
    findings, _grounded = multi_turn.validate_recovery_outline_task(
        planner=object(),
        task=missing_part_binding,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )
    assert findings == []


def test_exact_goal_state_label_is_allowed_without_other_state_delta() -> None:
    prepared = _prepared(_condition("goal_P", part_name="P", expected="restored"))
    session_state = _session(parts={"P": "faulted"})
    task = {
        "outline_id": "restore_goal_state",
        "event_name": "authored_exact_symbol",
        "resource_jid": "resource@localhost",
        "part_name": "P",
        "expected_start_state": {
            "resource_state": "idle",
            "held_part": None,
            "part_state": "faulted",
            "part_location": "station",
        },
        "expected_end_state": {
            "resource_state": "idle",
            "held_part": None,
            "part_state": "restored",
            "part_location": "station",
        },
        "rationale": "Satisfy the exact supplied part-state goal.",
    }

    findings, grounded = multi_turn.validate_recovery_outline_task(
        planner=object(),
        task=task,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )

    assert findings == []
    assert grounded is not None
    assert grounded["resource_jid"] == "resource@localhost"
    assert grounded["part_name"] == "P"

    empty_state_label = deepcopy(task)
    empty_state_label["expected_end_state"]["part_state"] = ""
    findings, _grounded = multi_turn.validate_recovery_outline_task(
        planner=object(),
        task=empty_state_label,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )
    assert findings == []


def test_neurosymbolic_prompt_assigns_only_candidate_generation_to_llm() -> None:
    prompt = multi_turn_prompts.render_multi_turn_phase_prompt(
        multi_turn_prompts.build_multi_turn_phase_prompt_input(
            phase="outline",
            llm_input={
                "observed_runtime_state": {"resources": []},
                "part_facts": [],
                "goal_conditions": [],
                "loaded_safety_rules": [],
            },
            session_state=_session(parts={}),
            recovery_resources={},
        )
    )

    assert "Do not select or rank the candidates" in prompt
    assert "do not pad the list" in prompt
    assert "You must choose the best candidate" not in prompt
    assert "selected_candidate_index" not in prompt


def test_no_progress_feedback_omits_declared_effects_and_event_names() -> None:
    selection_evidence = {
        "open_recovery_obligation_ids_before": ["condition_open"],
        "cleared_recovery_obligation_ids": [],
        "open_recovery_obligation_ids_after": ["condition_open"],
        "introduced_recovery_obligation_ids": [],
        "newly_enabled_recovery_event_ids": ["private_new_recovery_event"],
        "newly_enabled_nominal_reentry_event_ids": ["private_new_nominal_event"],
        "safety_dfa_states_before": {"SAFE_1": "1"},
        "safety_dfa_states_after": {"SAFE_1": "1"},
        "admissible_recovery_enabled_event_ids_before": ["private_event"],
        "admissible_recovery_enabled_event_ids_after": ["private_event"],
        "cca_admissible_goal_recovery_event_ids_before": [
            "private_goal_event"
        ],
        "cca_admissible_goal_recovery_event_ids_after": [
            "private_goal_event"
        ],
    }
    finding = {
        "constraint_code": "no_progressing_candidate",
        "evidence": {
            "candidate_comparison": [
                {
                    "candidate_id": "candidate_part",
                    "selection_status": "excluded_no_progress",
                    "resource_jid": "ur5e@localhost",
                    "part_name": "LG",
                    "event_name": "place_insert",
                    "rationale": "recommended transition",
                    "expected_end_state": {
                        "resource_state": "acquired",
                        "held_part": "LG",
                        "part_state": "misplaced",
                        "part_location": "ur5e@localhost_gripper",
                    },
                    "selection_evidence": deepcopy(selection_evidence),
                },
                {
                    "candidate_id": "candidate_resource",
                    "selection_status": "excluded_no_progress",
                    "resource_jid": "xarm6@localhost",
                    "expected_end_state": {
                        "resource_state": "failed",
                        "resource_location": "assembly_board-v1",
                    },
                    "selection_evidence": deepcopy(selection_evidence),
                },
                {
                    "candidate_id": "candidate_declared_fields",
                    "selection_status": "excluded_no_progress",
                    "resource_jid": "printer@localhost",
                    "part_name": "P",
                    "expected_end_state": {
                        "resource_state": "paused",
                        "job_state": "paused",
                        "part_quality": "unknown",
                    },
                    "selection_evidence": deepcopy(selection_evidence),
                },
            ]
        },
    }

    compact = multi_turn_prompts._compact_selection_feedback_evidence(finding)
    rows = compact["candidate_comparison"]

    assert "expected_end_state" not in rows[0]
    assert "part_name" not in rows[1]
    assert "expected_end_state" not in rows[1]
    assert "expected_end_state" not in rows[2]
    assert rows[0]["safety_dfa_states_before"] == {"SAFE_1": "1"}
    assert rows[0]["safety_dfa_states_after"] == {"SAFE_1": "1"}
    assert rows[0]["open_recovery_obligation_ids_before"] == ["condition_open"]
    assert rows[0]["open_recovery_obligation_ids_after"] == ["condition_open"]
    assert "place_insert" not in str(compact)
    assert "recommended transition" not in str(compact)
    assert "admissible_recovery_enabled_event_ids" not in str(compact)
    assert "cca_admissible_goal_recovery_event_ids" not in str(compact)
    assert "private_goal_event" not in str(compact)
    assert "private_new_recovery_event" not in str(compact)
    assert "private_new_nominal_event" not in str(compact)

    ambiguous = deepcopy(finding)
    ambiguous["constraint_code"] = "selection_ambiguous"
    ambiguous_row = multi_turn_prompts._compact_selection_feedback_evidence(
        ambiguous
    )["candidate_comparison"][0]
    assert "resource_jid" not in ambiguous_row
    assert "part_name" not in ambiguous_row
    assert "expected_end_state" not in ambiguous_row
    assert "safety_dfa_states_before" not in ambiguous_row


def test_no_progress_prompt_renders_goal_and_material_effect_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_jid = "ur5e@localhost"
    part_name = "LG"
    carried_location = f"{resource_jid}_gripper"
    session_state = _session(parts={part_name: "misplaced"})
    session_state["symbolic_resources"] = {
        resource_jid: {
            "resource_jid": resource_jid,
            "resource_state": "acquired",
            "current_state": "acquired",
            "resource_location": "prusa-mk4-2",
            "current_location": "prusa-mk4-2",
            "held_part": part_name,
            "held_part_location": carried_location,
        }
    }
    session_state["symbolic_parts"][part_name].update(
        {
            "part_location": carried_location,
            "current_location": carried_location,
            "part_holder_resource_jid": resource_jid,
            "current_holder_resource_jid": resource_jid,
            "goal_location": "assembly_board-v1",
        }
    )
    session_state["projected_safety_dfa_states"] = {"SAFE_1": "1"}
    condition = {
        "condition_id": "goal_lg_location",
        "kind": "goal_part_location",
        "condition_family": "goal",
        "entity_kind": "part",
        "entity": part_name,
        "field": "location",
        "expected": "assembly_board-v1",
        "actual": carried_location,
    }
    prepared = _prepared(condition)
    event = {
        "outline_id": "candidate_move_lg",
        "event_name": "move_with_part_to_assembly_board-v1",
        "resource_jid": resource_jid,
        "part_name": part_name,
        "expected_start_state": {
            "resource_state": "acquired",
            "resource_location": "prusa-mk4-2",
            "held_part": part_name,
            "part_state": "misplaced",
            "part_location": carried_location,
        },
        "expected_end_state": {
            "resource_state": "acquired",
            "resource_location": "assembly_board-v1",
            "held_part": part_name,
            "part_state": "misplaced",
            "part_location": carried_location,
        },
        "rationale": "old candidate rationale",
    }

    async def validate_candidate(**kwargs: Any) -> dict[str, Any]:
        evaluation = _evaluation(
            0,
            session_state=session_state,
            part_name=part_name,
            end_state="misplaced",
        )
        evaluation.update(
            {
                "task": deepcopy(event),
                "validated_task": deepcopy(event),
                "surface_events": [deepcopy(event)],
                "safety_dfa_states_before": {"SAFE_1": "1"},
                "safety_dfa_states_after": {"SAFE_1": "1"},
            }
        )
        return evaluation

    monkeypatch.setattr(
        multi_turn_outline_generation,
        "_validate_candidate_sequence",
        validate_candidate,
    )

    decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={"thought": "move while holding", "candidate_events": [event]},
            prepared_recovery_request=prepared,
            planner=object(),
        )
    )

    assert decision == "need_revision"
    model_finding = next(
        row
        for row in turn_entry["candidate_rejection_feedback"]
        if row.get("constraint_code") == "no_progressing_candidate"
    )
    comparison = model_finding["evidence"]["candidate_comparison"][0]
    assert comparison["resource_jid"] == resource_jid
    assert comparison["part_name"] == part_name
    assert comparison["expected_end_state"] == event["expected_end_state"]

    prompt = multi_turn_prompts.render_multi_turn_phase_prompt(
        multi_turn_prompts.build_multi_turn_phase_prompt_input(
            phase="outline",
            llm_input=deepcopy(prepared["llm_input"]),
            session_state=session_state,
            recovery_resources={},
            current_recovery_blockers=[deepcopy(condition)],
        )
    )

    assert "Modeled Continuation Gap" not in prompt
    assert "Recovery Goals" in prompt
    assert "restore LG to assembly_board-v1" in prompt
    assert '"expected_end_state"' not in prompt
    assert f'"part_location": "{carried_location}"' not in prompt
    assert "revise its symbolic expected_end_state" in prompt
    assert "Changing only event_name, outline_id, or rationale" in prompt
    assert "move_with_part_to_assembly_board-v1" not in prompt
    assert "old candidate rationale" not in prompt
    assert "place_insert" not in prompt
    assert "admissible_recovery_enabled_event_ids" not in prompt
    assert "future_goal_query" not in prompt
    assert "future_goal_capability_results" not in prompt


def test_unique_condition_clearing_candidate_is_selected_independent_of_names_and_order() -> None:
    session_state = _session(parts={"P": "faulted"})
    prepared = _prepared(_condition("goal_P", part_name="P", expected="restored"))
    clearing = _event(
        "arbitrary_symbol_alpha",
        part_name="P",
        start_state="faulted",
        end_state="restored",
    )
    no_progress = _event(
        "arbitrary_symbol_beta",
        part_name="P",
        start_state="faulted",
        end_state="other_state",
    )

    def select(rows: list[dict[str, Any]]) -> list[str]:
        evaluations = [
            _evaluation(
                index,
                session_state=session_state,
                part_name="P",
                end_state=str(row["expected_end_state"]["part_state"]),
            )
            for index, row in enumerate(rows)
        ]
        return multi_turn_outline_generation._apply_neurosymbolic_comparison(
            candidate_sequences=[
                {"candidate_index": index, "surface_events": [deepcopy(row)]}
                for index, row in enumerate(rows)
            ],
            candidate_evaluations=evaluations,
            session_state=session_state,
            prepared_recovery_request=prepared,
        )

    forward = select([clearing, no_progress])
    reverse = select([no_progress, clearing])
    renamed = deepcopy(clearing)
    renamed["event_name"] = "unrelated_authored_token"

    assert len(forward) == 1
    assert reverse == forward
    assert select([renamed, no_progress]) == forward


def test_selection_codes_distinguish_no_op_and_nonprogressing_label_effects() -> None:
    session_state = _session(parts={"P": "faulted"})
    prepared = _prepared(_condition("goal_P", part_name="P", expected="restored"))
    no_op = _event(
        "no_op_symbol",
        part_name="P",
        start_state="faulted",
        end_state="faulted",
    )
    restoring = _event(
        "restoring_symbol",
        part_name="P",
        start_state="faulted",
        end_state="restored",
    )
    other_label = _event(
        "other_label_symbol",
        part_name="P",
        start_state="faulted",
        end_state="other_state",
    )
    events = (no_op, restoring, other_label)
    evaluations = [
        _evaluation(
            index,
            session_state=session_state,
            part_name="P",
            end_state=str(event["expected_end_state"]["part_state"]),
        )
        for index, event in enumerate(events)
    ]

    selected = multi_turn_outline_generation._apply_neurosymbolic_comparison(
        candidate_sequences=[
            {"candidate_index": index, "surface_events": [deepcopy(event)]}
            for index, event in enumerate(events)
        ],
        candidate_evaluations=evaluations,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )

    assert selected == [evaluations[1]["candidate_id"]]
    assert evaluations[0]["valid"] is True
    assert evaluations[0]["validation_findings"] == []
    assert evaluations[0]["selection_status"] == "excluded_no_progress"
    assert evaluations[0]["selection_constraint_codes"] == ["no_state_change"]
    assert evaluations[1]["selection_constraint_codes"] == []
    assert evaluations[2]["selection_status"] == "excluded_no_progress"
    assert evaluations[2]["selection_constraint_codes"] == [
        "label_only_state_change"
    ]

    audit_rows = multi_turn._compact_artifact_candidate_evaluations(
        [evaluations[0]]
    )
    assert audit_rows[0]["selection_constraint_codes"] == ["no_state_change"]
    assert audit_rows[0]["constraint_codes"] == ["no_state_change"]
    assert "findings" not in audit_rows[0]
    concise = recovery_artifacts._outline_result_payload(
        {
            "turn_index": 1,
            "phase": "outline",
            "decision": "need_revision",
            "candidate_evaluation_summary": audit_rows,
            "artifact_paths": {},
        }
    )
    assert concise["constraint_codes"] == ["no_state_change"]
    assert concise["candidate_evaluation_summary"][0]["constraint_codes"] == [
        "no_state_change"
    ]
    assert "selection_constraint_codes" not in concise[
        "candidate_evaluation_summary"
    ][0]
    assert all(
        "constraint_codes" not in stage
        for stage in concise["candidate_evaluation_summary"][0][
            "validation_stages"
        ]
    )


def test_incomplete_primitive_program_cannot_build_executable_proposal() -> None:
    result = multi_turn.build_multi_turn_recovery_proposal(
        final_output_payload={
            "final_output_stage": "ready_for_primitive_generation",
            "transition_trace": [
                {
                    "outline_id": "recovery_1",
                    "resource_jid": "resource@localhost",
                    "event_name": "authored_symbol",
                }
            ],
            "accepted_primitive_program": [],
            "primitive_program_complete": False,
        },
        prepared_recovery_request={"llm_input": {}},
    )

    assert result["accepted"] is False
    assert result["recovery_proposal"] is None
    assert result["reason"] == "final output did not include accepted_primitive_program"


def test_ra_declared_guard_enabling_candidate_progresses() -> None:
    session_state = _session(parts={"P": "faulted", "AUX": "held"})
    prepared = _prepared(_condition("goal_P", part_name="P", expected="restored"))
    session_state["symbolic_resources"]["resource@localhost"]["held_part"] = "AUX"
    event = {
        "outline_id": "release",
        "event_name": "no_action_verb_required",
        "resource_jid": "resource@localhost",
        "expected_start_state": {"resource_state": "idle", "held_part": "AUX"},
        "expected_end_state": {"resource_state": "idle", "held_part": None},
        "rationale": "exact declared effect",
    }
    projected = deepcopy(session_state)
    multi_turn._apply_task_effects_to_symbolic_state(event, projected)
    evaluation = {
        "candidate_index": 0,
        "valid": True,
        "projected_symbolic_resources": deepcopy(projected["symbolic_resources"]),
        "projected_symbolic_parts": deepcopy(projected["symbolic_parts"]),
        "safety_dfa_states_before": {},
        "safety_dfa_states_after": {},
        "agent_filtered_enabledness": True,
        "admissible_recovery_enabled_event_ids_before": ["release_entity"],
        "admissible_recovery_enabled_event_ids_after": ["acquire_entity"],
    }

    selected = multi_turn_outline_generation._apply_neurosymbolic_comparison(
        candidate_sequences=[{"candidate_index": 0, "surface_events": [event]}],
        candidate_evaluations=[evaluation],
        session_state=session_state,
        prepared_recovery_request=prepared,
    )

    assert selected == [evaluation["candidate_id"]]
    assert evaluation["selection_evidence"]["cleared_recovery_obligation_ids"] == []
    assert evaluation["selection_evidence"]["newly_enabled_recovery_event_ids"]

    renamed_event = deepcopy(event)
    renamed_event["event_name"] = "another_exact_authored_name"
    renamed_event["expected_end_state"]["resource_state"] = (
        "another_exact_intermediate_state"
    )
    renamed_projected = deepcopy(session_state)
    multi_turn._apply_task_effects_to_symbolic_state(
        renamed_event,
        renamed_projected,
    )
    renamed_evaluation = {
        "candidate_index": 0,
        "valid": True,
        "projected_symbolic_resources": deepcopy(
            renamed_projected["symbolic_resources"]
        ),
            "projected_symbolic_parts": deepcopy(renamed_projected["symbolic_parts"]),
            "safety_dfa_states_before": {},
            "safety_dfa_states_after": {},
            "agent_filtered_enabledness": True,
            "admissible_recovery_enabled_event_ids_before": [
                "release_entity"
            ],
            "admissible_recovery_enabled_event_ids_after": [
                "acquire_entity"
            ],
        }

    renamed_selected = (
        multi_turn_outline_generation._apply_neurosymbolic_comparison(
            candidate_sequences=[
                {"candidate_index": 0, "surface_events": [renamed_event]}
            ],
            candidate_evaluations=[renamed_evaluation],
            session_state=session_state,
            prepared_recovery_request=prepared,
        )
    )

    assert renamed_selected == [renamed_evaluation["candidate_id"]]
    assert (
        renamed_evaluation["selection_evidence"][
            "newly_enabled_recovery_event_ids"
        ]
        == evaluation["selection_evidence"]["newly_enabled_recovery_event_ids"]
    )


def test_incomparable_candidates_use_stable_effect_representative() -> None:
    session_state = _session(parts={"P": "faulted", "Q": "faulted"})
    prepared = _prepared(
        _condition("goal_P", part_name="P", expected="restored"),
        _condition("goal_Q", part_name="Q", expected="restored"),
    )
    events = [
        _event("restore_P", part_name="P", start_state="faulted", end_state="restored"),
        _event("restore_Q", part_name="Q", start_state="faulted", end_state="restored"),
    ]
    evaluations = [
        _evaluation(
            index,
            session_state=session_state,
            part_name=part_name,
            end_state="restored",
        )
        for index, part_name in enumerate(("P", "Q"))
    ]

    nondominated = multi_turn_outline_generation._apply_neurosymbolic_comparison(
        candidate_sequences=[
            {"candidate_index": index, "surface_events": [event]}
            for index, event in enumerate(events)
        ],
        candidate_evaluations=evaluations,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )

    assert len(nondominated) == 1
    assert nondominated == [min(row["candidate_id"] for row in evaluations)]
    assert {row["selection_status"] for row in evaluations} == {
        "nondominated",
        "stable_representative_not_selected",
    }
    selected = next(row for row in evaluations if row["selection_status"] == "nondominated")
    assert selected["tie_representative_evidence"] == {
        "used": True,
        "rule": "smallest_exact_effect_candidate_id",
        "representative_candidate_id": selected["candidate_id"],
        "symbolically_tied_candidate_ids": sorted(
            {row["candidate_id"] for row in evaluations}
        ),
        "operational_superiority_claimed": False,
    }
    assert session_state["accepted_outline_prefix"] == []


def test_equivalent_exact_successors_select_one_reproducible_representative() -> None:
    session_state = _session(parts={"P": "faulted"})
    prepared = _prepared(_condition("goal_P", part_name="P", expected="restored"))
    first = _event(
        "first_name",
        part_name="P",
        start_state="faulted",
        end_state="restored",
    )
    second = deepcopy(first)
    second["outline_id"] = "other_outline"
    second["event_name"] = "second_name"
    second["rationale"] = "other rationale"
    evaluations = [
        _evaluation(
            index,
            session_state=session_state,
            part_name="P",
            end_state="restored",
        )
        for index in range(2)
    ]

    nondominated = multi_turn_outline_generation._apply_neurosymbolic_comparison(
        candidate_sequences=[
            {"candidate_index": 0, "surface_events": [first]},
            {"candidate_index": 1, "surface_events": [second]},
        ],
        candidate_evaluations=evaluations,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )

    assert len(nondominated) == 1
    assert evaluations[0]["candidate_id"] == evaluations[1]["candidate_id"]
    assert [row["selection_status"] for row in evaluations].count("nondominated") == 1
    assert [
        row["selection_status"] for row in evaluations
    ].count("stable_representative_not_selected") == 1


def test_general_mocked_sequence_uses_cca_admissibility_then_converges() -> None:
    goal_condition = _condition("affected_part_goal", part_name="P", expected="restored")
    prepared = _prepared(goal_condition)
    session_state = _session(parts={"P": "faulted", "AUX": "in_gripper"})
    session_state["symbolic_resources"] = {
        "blocker@localhost": {
            "resource_jid": "blocker@localhost",
            "resource_state": "occupied",
            "current_state": "occupied",
            "resource_location": "station",
            "current_location": "station",
            "held_part": None,
        },
        "handler@localhost": {
            "resource_jid": "handler@localhost",
            "resource_state": "loaded",
            "current_state": "loaded",
            "held_part": "AUX",
        },
    }
    session_state["symbolic_parts"]["AUX"].update(
        {
            "part_holder_resource_jid": "handler@localhost",
            "current_holder_resource_jid": "handler@localhost",
            "part_location": "handler@localhost_gripper",
            "current_location": "handler@localhost_gripper",
        }
    )
    events = [
        {
            "outline_id": "clear_occupancy",
            "event_name": "authored_a",
            "resource_jid": "blocker@localhost",
            "expected_start_state": {
                "resource_state": "occupied",
                "resource_location": "station",
                "held_part": None,
            },
            "expected_end_state": {
                "resource_state": "clear",
                "resource_location": "buffer",
                "held_part": None,
            },
            "rationale": "clear exact occupancy condition",
        },
        {
            "outline_id": "free_guard",
            "event_name": "authored_b",
            "resource_jid": "handler@localhost",
            "part_name": "AUX",
            "expected_start_state": {
                "resource_state": "loaded",
                "held_part": "AUX",
                "part_state": "in_gripper",
                "part_location": "handler@localhost_gripper",
            },
            "expected_end_state": {
                "resource_state": "idle",
                "held_part": None,
                "part_state": "staged",
                "part_location": "buffer",
            },
            "rationale": "enable an exact capability guard",
        },
        {
            "outline_id": "restore_affected_part",
            "event_name": "authored_c",
            "resource_jid": "handler@localhost",
            "part_name": "P",
            "expected_start_state": {
                "resource_state": "idle",
                "held_part": None,
                "part_state": "faulted",
                "part_location": "station",
            },
            "expected_end_state": {
                "resource_state": "completed",
                "held_part": None,
                "part_state": "restored",
                "part_location": "station",
            },
            "rationale": "clear exact affected part goal",
        },
    ]

    evidence_rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        projected = deepcopy(session_state)
        multi_turn._apply_task_effects_to_symbolic_state(event, projected)
        evaluation = {
            "candidate_index": 0,
            "valid": True,
            "projected_symbolic_resources": deepcopy(
                projected["symbolic_resources"]
            ),
            "projected_symbolic_parts": deepcopy(projected["symbolic_parts"]),
            "safety_dfa_states_before": {},
            "safety_dfa_states_after": {},
            "cca_admissible_goal_recovery_event_ids_before": (
                [] if event_index == 0 else ["private_goal_event"]
            ),
            "cca_admissible_goal_recovery_event_ids_after": [
                "private_goal_event"
            ],
            "future_goal_evaluation_complete": True,
            "agent_filtered_enabledness": True,
            "admissible_recovery_enabled_event_ids_before": [
                "release_entity"
            ],
            "admissible_recovery_enabled_event_ids_after": (
                ["acquire_entity"]
                if event_index >= 1
                else ["release_entity"]
            ),
        }
        selected = multi_turn_outline_generation._apply_neurosymbolic_comparison(
            candidate_sequences=[
                {"candidate_index": 0, "surface_events": [deepcopy(event)]}
            ],
            candidate_evaluations=[evaluation],
            session_state=session_state,
            prepared_recovery_request=prepared,
        )
        assert selected == [evaluation["candidate_id"]]
        evidence_rows.append(deepcopy(evaluation["selection_evidence"]))
        multi_turn._apply_task_effects_to_symbolic_state(event, session_state)

    assert evidence_rows[0]["cleared_recovery_obligation_ids"] == []
    assert evidence_rows[0]["newly_cca_admissible_goal_recovery_event_ids"] == [
        "private_goal_event"
    ]
    assert evidence_rows[1]["cleared_recovery_obligation_ids"] == []
    assert evidence_rows[1]["newly_enabled_recovery_event_ids"]
    assert evidence_rows[2]["cleared_recovery_obligation_ids"] == ["affected_part_goal"]
    assert not multi_turn_outline_generation._unresolved_condition_ids(
        session_state=session_state,
        prepared_recovery_request=prepared,
    )


def test_cca_admissibility_does_not_prefer_home_over_another_clear_location() -> None:
    prepared = _prepared(
        _condition("affected_part_goal", part_name="P", expected="restored")
    )
    session_state = _session(parts={"P": "faulted"})
    session_state["symbolic_resources"] = {
        "blocker@localhost": {
            "resource_jid": "blocker@localhost",
            "resource_state": "occupied",
            "current_state": "occupied",
            "resource_location": "station",
            "current_location": "station",
            "held_part": None,
        }
    }
    events = [
        {
            "outline_id": f"clear_to_{location}",
            "event_name": f"authored_{location}",
            "resource_jid": "blocker@localhost",
            "expected_start_state": {
                "resource_state": "occupied",
                "resource_location": "station",
                "held_part": None,
            },
            "expected_end_state": {
                "resource_state": "clear",
                "resource_location": location,
                "held_part": None,
            },
            "rationale": "declare a clear successor",
        }
        for location in ("home", "buffer")
    ]
    evaluations: list[dict[str, Any]] = []
    for candidate_index, event in enumerate(events):
        projected = deepcopy(session_state)
        multi_turn._apply_task_effects_to_symbolic_state(event, projected)
        evaluations.append(
            {
                "candidate_index": candidate_index,
                "valid": True,
                "projected_symbolic_resources": deepcopy(
                    projected["symbolic_resources"]
                ),
                "projected_symbolic_parts": deepcopy(projected["symbolic_parts"]),
                "safety_dfa_states_before": {},
                "safety_dfa_states_after": {},
                "cca_admissible_goal_recovery_event_ids_before": [],
                "cca_admissible_goal_recovery_event_ids_after": [
                    "private_goal_event"
                ],
                "future_goal_evaluation_complete": True,
            }
        )

    nondominated = multi_turn_outline_generation._apply_neurosymbolic_comparison(
        candidate_sequences=[
            {"candidate_index": index, "surface_events": [deepcopy(event)]}
            for index, event in enumerate(events)
        ],
        candidate_evaluations=evaluations,
        session_state=session_state,
        prepared_recovery_request=prepared,
    )

    assert len(nondominated) == 1
    assert {row["selection_status"] for row in evaluations} == {
        "nondominated",
        "stable_representative_not_selected",
    }
    selected = next(
        row for row in evaluations if row["selection_status"] == "nondominated"
    )
    tie_evidence = selected["tie_representative_evidence"]
    assert tie_evidence["used"] is True
    assert tie_evidence["operational_superiority_claimed"] is False
    assert len(tie_evidence["symbolically_tied_candidate_ids"]) == 2


def test_empty_and_budget_overflow_generation_fail_before_agent_validation() -> None:
    session_state = _session(parts={}, candidate_bound=2)

    empty_decision, _ = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=deepcopy(session_state),
            parsed_response={"thought": "none", "candidate_events": []},
            prepared_recovery_request={},
            planner=object(),
        )
    )
    overflow_decision, _ = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=deepcopy(session_state),
            parsed_response={
                "thought": "too many",
                "candidate_events": [
                    _event(
                        f"event_{index}",
                        part_name="P",
                        start_state="faulted",
                        end_state="restored",
                    )
                    for index in range(3)
                ],
            },
            prepared_recovery_request={},
            planner=object(),
        )
    )
    selected_index_decision, selected_index_turn = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=deepcopy(session_state),
            parsed_response={
                "thought": "invalid ownership",
                "selected_candidate_index": 0,
                "candidate_events": [
                    _event(
                        "event",
                        part_name="P",
                        start_state="faulted",
                        end_state="restored",
                    )
                ],
            },
            prepared_recovery_request={},
            planner=object(),
        )
    )

    assert empty_decision == "need_revision"
    assert overflow_decision == "need_revision"
    assert selected_index_decision == "need_revision"
    assert (
        selected_index_turn["validation_findings"][0]["constraint_code"]
        == "candidate_schema_violation"
    )


def test_neurosymbolic_handler_commits_unique_successor_without_selected_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_state = _session(parts={"P": "faulted"})
    session_state.update(
        {
            "selection_revision_count": 4,
            "selection_revision_fingerprint": "old-state",
            "selection_repeated_failure_count": 2,
            "selection_repeated_failure_fingerprint": "old-failure",
            "candidate_revision_targets": [{"candidate_id": "old-candidate"}],
            "candidate_revision_state_fingerprint": "old-state",
        }
    )
    prepared = _prepared(_condition("goal_P", part_name="P", expected="restored"))
    event = _event(
        "authored_transition",
        part_name="P",
        start_state="faulted",
        end_state="restored",
    )

    async def validate_candidate(**kwargs: Any) -> dict[str, Any]:
        candidate = dict(kwargs["candidate"])
        candidate_index = int(candidate.get("candidate_index") or 0)
        evaluation = _evaluation(
            candidate_index,
            session_state=session_state,
            part_name="P",
            end_state="restored",
        )
        evaluation.update(
            {
                "task": deepcopy(event),
                "validated_task": deepcopy(event),
                "surface_events": [deepcopy(event)],
                "validated_events": [deepcopy(event)],
                "committed_events": [
                    multi_turn._commit_selected_candidate_task(
                        task=event,
                        sequence_index=int(kwargs["sequence_index"]),
                    )
                ],
                "grounded_actions": [],
                "pa_state_fingerprint": (
                    multi_turn_outline_generation._pa_state_fingerprint(
                        session_state=session_state,
                        prepared_recovery_request=prepared,
                    )
                ),
                "safety_rule_fingerprint": "rules",
                "live_safety_dfa_state_fingerprint": "live",
            }
        )
        return evaluation

    monkeypatch.setattr(
        multi_turn_outline_generation,
        "_validate_candidate_sequence",
        validate_candidate,
    )

    decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={"thought": "propose", "candidate_events": [event]},
            prepared_recovery_request=prepared,
            planner=object(),
        )
    )
    turn_entry["decision"] = decision
    artifact = multi_turn._artifact_response_payload(
        phase="outline",
        session_state=session_state,
        parsed_response={"thought": "propose", "candidate_events": [event]},
        turn_entry=turn_entry,
    )

    assert decision == "outline_ready"
    assert turn_entry["selected_by"] == "neurosymbolic"
    assert turn_entry["selection_status"] == "selected"
    assert "selected_candidate_index" not in turn_entry
    assert "selected_candidate_index" not in artifact
    assert artifact["selected_by"] == "neurosymbolic"
    assert artifact["selection_status"] == "selected"
    assert artifact["selection_evidence"]["cleared_recovery_obligation_ids"] == ["goal_P"]
    assert len(artifact["nondominated_candidate_ids"]) == 1
    assert "projected_successor" not in artifact["candidate_evaluation_summary"][0]
    assert len(session_state["accepted_outline_prefix"]) == 1
    assert session_state["symbolic_parts"]["P"]["current_state"] == "restored"
    assert session_state["selection_revision_count"] == 0
    assert session_state["selection_revision_fingerprint"] == ""
    assert session_state["selection_repeated_failure_count"] == 0
    assert session_state["selection_repeated_failure_fingerprint"] == ""
    assert session_state["candidate_revision_targets"] == []
    assert session_state["candidate_revision_state_fingerprint"] == ""
    roles = {
        stage["validator_role"]
        for stage in turn_entry["candidate_evaluations"][0]["validation_stages"]
    }
    assert roles == {"PA", "RA", "CCA"}


def test_neurosymbolic_handler_commits_stable_representative_without_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_state = _session(parts={"P": "faulted", "Q": "faulted"})
    prepared = _prepared(
        _condition("goal_P", part_name="P", expected="restored"),
        _condition("goal_Q", part_name="Q", expected="restored"),
    )
    events = [
        _event("restore_P", part_name="P", start_state="faulted", end_state="restored"),
        _event("restore_Q", part_name="Q", start_state="faulted", end_state="restored"),
    ]

    async def validate_candidate(**kwargs: Any) -> dict[str, Any]:
        candidate = dict(kwargs["candidate"])
        candidate_index = int(candidate.get("candidate_index") or 0)
        event = dict(candidate["surface_events"][0])
        part_name = str(event["part_name"])
        evaluation = _evaluation(
            candidate_index,
            session_state=session_state,
            part_name=part_name,
            end_state="restored",
        )
        evaluation.update(
            {
                "task": deepcopy(event),
                "validated_task": deepcopy(event),
                "surface_events": [deepcopy(event)],
                "committed_events": [
                    multi_turn._commit_selected_candidate_task(
                        task=event,
                        sequence_index=int(kwargs["sequence_index"]),
                    )
                ],
                "grounded_actions": [],
                "pa_state_fingerprint": (
                    multi_turn_outline_generation._pa_state_fingerprint(
                        session_state=session_state,
                        prepared_recovery_request=prepared,
                    )
                ),
            }
        )
        return evaluation

    monkeypatch.setattr(
        multi_turn_outline_generation,
        "_validate_candidate_sequence",
        validate_candidate,
    )

    decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={"thought": "propose", "candidate_events": events},
            prepared_recovery_request=prepared,
            planner=object(),
        )
    )

    assert decision == "need_next_task"
    assert len(turn_entry["nondominated_candidate_ids"]) == 1
    assert len(session_state["accepted_outline_prefix"]) == 1
    assert turn_entry["tie_representative_evidence"]["used"] is True
    assert turn_entry["tie_representative_evidence"][
        "operational_superiority_claimed"
    ] is False


def test_repeated_invalid_candidates_terminate_as_selection_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_state = _session(parts={"P": "faulted", "Q": "faulted"})
    prepared = _prepared(
        _condition("goal_P", part_name="P", expected="restored"),
        _condition("goal_Q", part_name="Q", expected="restored"),
    )
    events = [
        _event("restore_P", part_name="P", start_state="faulted", end_state="restored"),
        _event("restore_Q", part_name="Q", start_state="faulted", end_state="restored"),
    ]
    reject_representative = True

    async def validate_candidate(**kwargs: Any) -> dict[str, Any]:
        candidate = dict(kwargs["candidate"])
        candidate_index = int(candidate.get("candidate_index") or 0)
        event = dict(candidate["surface_events"][0])
        if reject_representative:
            finding = multi_turn.annotate_validation_finding(
                {
                    "validation_category": "syntax_and_grounding_validation",
                    "constraint_owner": "binding",
                    "constraint_family": "binding",
                    "constraint_code": "candidate_schema_violation",
                    "reason": "representative requires correction",
                    "task_id": str(event.get("outline_id") or ""),
                    "resource_jid": str(event.get("resource_jid") or ""),
                    "part_name": str(event.get("part_name") or ""),
                }
            )
            return {
                "candidate_index": candidate_index,
                "valid": False,
                "task": deepcopy(event),
                "surface_events": [deepcopy(event)],
                "validation_findings": [finding],
                "validation_stages": [],
                "pa_state_fingerprint": (
                    multi_turn_outline_generation._pa_state_fingerprint(
                        session_state=session_state,
                        prepared_recovery_request=prepared,
                    )
                ),
            }

        part_name = str(event["part_name"])
        evaluation = _evaluation(
            candidate_index,
            session_state=session_state,
            part_name=part_name,
            end_state="restored",
        )
        evaluation.update(
            {
                "task": deepcopy(event),
                "validated_task": deepcopy(event),
                "surface_events": [deepcopy(event)],
                "committed_events": [
                    multi_turn._commit_selected_candidate_task(
                        task=event,
                        sequence_index=int(kwargs["sequence_index"]),
                    )
                ],
                "grounded_actions": [],
                "pa_state_fingerprint": (
                    multi_turn_outline_generation._pa_state_fingerprint(
                        session_state=session_state,
                        prepared_recovery_request=prepared,
                    )
                ),
            }
        )
        return evaluation

    monkeypatch.setattr(
        multi_turn_outline_generation,
        "_validate_candidate_sequence",
        validate_candidate,
    )

    decision, _turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={"thought": "compare", "candidate_events": events},
            prepared_recovery_request=prepared,
            planner=object(),
        )
    )
    assert decision == "need_revision"
    assert session_state["selection_revision_count"] == 1

    decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={"thought": "revise", "candidate_events": events},
            prepared_recovery_request=prepared,
            planner=object(),
        )
    )
    assert decision == "need_revision"
    assert session_state["selection_revision_count"] == 2
    feedback_codes = {
        str(finding.get("constraint_code") or "")
        for row in turn_entry["candidate_rejection_feedback"]
        for finding in (
            [row]
            if str(row.get("constraint_code") or "")
            else list(row.get("validation_findings") or [])
        )
    }
    assert feedback_codes == {"candidate_schema_violation"}
    prompt = multi_turn_prompts.render_multi_turn_phase_prompt(
        multi_turn_prompts.build_multi_turn_phase_prompt_input(
            phase="outline",
            llm_input=deepcopy(prepared.get("llm_input") or {}),
            session_state=session_state,
            recovery_resources=deepcopy(prepared.get("recovery_resources") or {}),
        )
    )
    assert "representative requires correction" in prompt
    assert "selection_ambiguous" not in prompt

    decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={"thought": "revise", "candidate_events": events},
            prepared_recovery_request=prepared,
            planner=object(),
        )
    )
    assert decision == "selection_unresolved"
    assert turn_entry["selection_status"] == "selection_unresolved"
    assert session_state["selection_revision_count"] == 3
    assert session_state["selection_revision_limit"] == 6
    assert session_state["selection_repeated_failure_count"] == 3
    assert session_state["selection_repeated_failure_limit"] == 3
    assert session_state["accepted_outline_prefix"] == []


def test_materially_different_failures_reach_the_total_revision_limit() -> None:
    session_state = _session(parts={"P": "faulted"})
    prepared = _prepared(_condition("goal_P", part_name="P", expected="restored"))

    for attempt_index in range(6):
        (
            _state_fingerprint,
            total_count,
            total_limit,
            repeated_count,
            repeated_limit,
        ) = multi_turn_outline_generation._register_selection_revision(
            session_state=session_state,
            prepared_recovery_request=prepared,
            candidate_evaluations=[
                {
                    "candidate_id": f"candidate_effect_{attempt_index}",
                    "valid": False,
                    "selection_status": "excluded_invalid",
                    "validation_findings": [
                        {"constraint_code": "candidate_schema_violation"}
                    ],
                }
            ],
        )

        assert total_count == attempt_index + 1
        assert total_limit == 6
        assert repeated_count == 1
        assert repeated_limit == 3

    assert session_state["selection_revision_count"] == 6
    assert session_state["selection_repeated_failure_count"] == 1


def test_six_materially_different_handler_attempts_are_selection_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_state = _session(parts={"P": "faulted"})
    prepared = _prepared(_condition("goal_P", part_name="P", expected="restored"))

    async def validate_candidate(**kwargs: Any) -> dict[str, Any]:
        candidate = dict(kwargs["candidate"])
        candidate_index = int(candidate.get("candidate_index") or 0)
        event = dict(candidate["surface_events"][0])
        evaluation = _evaluation(
            candidate_index,
            session_state=session_state,
            part_name="P",
            end_state="faulted",
        )
        evaluation.update(
            {
                "task": deepcopy(event),
                "validated_task": deepcopy(event),
                "surface_events": [deepcopy(event)],
                "pa_state_fingerprint": (
                    multi_turn_outline_generation._pa_state_fingerprint(
                        session_state=session_state,
                        prepared_recovery_request=prepared,
                    )
                ),
            }
        )
        return evaluation

    monkeypatch.setattr(
        multi_turn_outline_generation,
        "_validate_candidate_sequence",
        validate_candidate,
    )

    decisions: list[str] = []
    for attempt_index in range(6):
        event = _event(
            f"attempt_{attempt_index}",
            part_name="P",
            start_state="faulted",
            end_state="faulted",
        )
        event["expected_end_state"]["attempt_marker"] = attempt_index
        decision, _turn_entry = asyncio.run(
            multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
                session_state=session_state,
                parsed_response={
                    "thought": "try another material effect",
                    "candidate_events": [event],
                },
                prepared_recovery_request=prepared,
                planner=object(),
            )
        )
        decisions.append(decision)

    assert decisions == ["need_revision"] * 5 + ["selection_unresolved"]
    assert session_state["selection_revision_count"] == 6
    assert session_state["selection_repeated_failure_count"] == 1


def test_pa_revision_targets_exclude_ra_cca_and_no_progress_candidates() -> None:
    def _evaluation_row(
        *,
        candidate_index: int,
        validator_role: str,
        valid: bool = False,
        selection_status: str = "excluded_invalid",
    ) -> dict[str, Any]:
        return {
            "candidate_index": candidate_index,
            "candidate_id": f"candidate_{candidate_index}",
            "valid": valid,
            "selection_status": selection_status,
            "task": {
                "resource_jid": "resource@localhost",
                "part_name": f"P{candidate_index}",
                "expected_end_state": {
                    "resource_state": "idle",
                    "held_part": f"P{candidate_index}",
                    "part_state": "in_gripper",
                    "part_location": "resource@localhost_gripper",
                },
            },
            "validation_findings": [
                {"constraint_code": f"constraint_{candidate_index}"}
            ],
            "validation_stages": [
                {
                    "validator_role": validator_role,
                    "status": "rejected",
                }
            ],
        }

    evaluations = [
        _evaluation_row(candidate_index=0, validator_role="PA"),
        _evaluation_row(candidate_index=1, validator_role="RA"),
        _evaluation_row(candidate_index=2, validator_role="CCA"),
        _evaluation_row(
            candidate_index=3,
            validator_role="PA",
            valid=True,
            selection_status="excluded_no_progress",
        ),
    ]

    targets = multi_turn_outline_generation._pa_candidate_revision_targets(
        candidate_evaluations=evaluations,
        candidate_bound=5,
    )

    assert targets == [
        {
            "candidate_id": "candidate_0",
            "candidate_index": 0,
            "resource_jid": "resource@localhost",
            "part_name": "P0",
            "expected_end_state": {
                "resource_state": "idle",
                "held_part": "P0",
                "part_state": "in_gripper",
                "part_location": "resource@localhost_gripper",
            },
            "constraint_codes": ["constraint_0"],
        }
    ]


def test_plant_state_change_clears_revision_targets_and_both_counters() -> None:
    session_state = _session(parts={"P": "faulted"})
    prepared = _prepared(_condition("goal_P", part_name="P", expected="restored"))
    session_state.update(
        {
            "candidate_revision_targets": [
                {
                    "candidate_id": "candidate_p",
                    "resource_jid": "resource@localhost",
                    "part_name": "P",
                    "expected_end_state": {"part_state": "restored"},
                    "constraint_codes": ["expected_start_state_mismatch"],
                }
            ],
            "candidate_revision_state_fingerprint": (
                multi_turn_outline_generation._pa_state_fingerprint(
                    session_state=session_state,
                    prepared_recovery_request=prepared,
                )
            ),
            "selection_revision_count": 4,
            "selection_revision_fingerprint": "old-state",
            "selection_repeated_failure_count": 2,
            "selection_repeated_failure_fingerprint": "old-failure",
        }
    )
    session_state["symbolic_parts"]["P"].update(
        {"part_state": "observed_elsewhere", "current_state": "observed_elsewhere"}
    )

    targets = multi_turn_outline_generation._active_candidate_revision_targets(
        session_state=session_state,
        prepared_recovery_request=prepared,
    )

    assert targets == []
    assert session_state["candidate_revision_targets"] == []
    assert session_state["candidate_revision_state_fingerprint"] == ""
    assert session_state["selection_revision_count"] == 0
    assert session_state["selection_revision_fingerprint"] == ""
    assert session_state["selection_repeated_failure_count"] == 0
    assert session_state["selection_repeated_failure_fingerprint"] == ""


def test_pa_revision_targets_require_changed_effects_for_each_identity() -> None:
    targets = [
        {
            "candidate_id": "candidate_lg",
            "resource_jid": "ur5e@localhost",
            "part_name": "LG",
            "expected_end_state": {
                "held_part": "LG",
                "part_state": "in_gripper",
                "part_location": "prusa-mk4-2",
            },
            "constraint_codes": ["held_part_location_mismatch"],
        },
        {
            "candidate_id": "candidate_mcp",
            "resource_jid": "ur5e@localhost",
            "part_name": "MCP",
            "expected_end_state": {
                "held_part": None,
                "part_state": "placed",
                "part_location": "prusa-mk4-2",
            },
            "constraint_codes": ["held_part_location_mismatch"],
        },
    ]
    candidate_sequences = [
        {
            "candidate_index": 0,
            "surface_events": [
                {
                    "outline_id": "renamed_mcp",
                    "event_name": "renamed_mcp_event",
                    "resource_jid": "ur5e@localhost",
                    "part_name": "MCP",
                    "expected_end_state": {
                        "held_part": None,
                        "part_state": "waiting",
                        "part_location": "prusa-mk4-2",
                    },
                }
            ],
        }
    ]

    findings = (
        multi_turn_outline_generation._candidate_revision_requirement_findings(
            candidate_sequences=candidate_sequences,
            revision_targets=targets,
        )
    )

    assert len(findings) == 1
    assert findings[0]["constraint_code"] == "candidate_schema_violation"
    assert findings[0]["resource_jid"] == "ur5e@localhost"
    assert findings[0]["part_name"] == "LG"
    assert findings[0]["evidence"]["candidate_id"] == "candidate_lg"

    corrected_sequences = [
        *candidate_sequences,
        {
            "candidate_index": 1,
            "surface_events": [
                {
                    "outline_id": "corrected_lg",
                    "event_name": "corrected_lg_event",
                    "resource_jid": "ur5e@localhost",
                    "part_name": "LG",
                    "expected_end_state": {
                        "held_part": "LG",
                        "part_state": "in_gripper",
                        "part_location": "ur5e@localhost_gripper",
                    },
                }
            ],
        },
    ]
    assert (
        multi_turn_outline_generation._candidate_revision_requirement_findings(
            candidate_sequences=corrected_sequences,
            revision_targets=targets,
        )
        == []
    )


def test_omitted_pa_revision_target_is_rejected_before_ra_cca(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_state = _session(parts={"P": "faulted", "Q": "faulted"})
    prepared = _prepared(_condition("goal_P", part_name="P", expected="restored"))
    session_state["candidate_revision_targets"] = [
        {
            "candidate_id": "candidate_p",
            "resource_jid": "resource@localhost",
            "part_name": "P",
            "expected_end_state": {
                "resource_state": "idle",
                "held_part": None,
                "part_state": "restored",
                "part_location": "station",
            },
            "constraint_codes": ["expected_start_state_mismatch"],
        }
    ]
    session_state["candidate_revision_state_fingerprint"] = (
        multi_turn_outline_generation._pa_state_fingerprint(
            session_state=session_state,
            prepared_recovery_request=prepared,
        )
    )

    async def validation_must_not_run(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("candidate validation ran before revision continuity")

    monkeypatch.setattr(
        multi_turn_outline_generation,
        "_validate_candidate_sequence",
        validation_must_not_run,
    )
    decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={
                "thought": "omit the required P correction",
                "candidate_events": [
                    _event(
                        "restore_Q",
                        part_name="Q",
                        start_state="faulted",
                        end_state="restored",
                    )
                ],
            },
            prepared_recovery_request=prepared,
            planner=object(),
        )
    )

    assert decision == "need_revision"
    evaluation = turn_entry["candidate_evaluations"][0]
    assert evaluation["valid"] is False
    assert {
        finding["constraint_code"]
        for finding in evaluation["validation_findings"]
    } == {"candidate_schema_violation"}
    assert [
        (stage["validator_role"], stage["status"])
        for stage in evaluation["validation_stages"]
    ] == [
        ("PA", "rejected"),
        ("RA", "skipped"),
        ("CCA", "skipped"),
    ]
    assert session_state["candidate_revision_targets"][0]["part_name"] == "P"


def test_candidate_prompt_requires_pa_revision_targets_without_action_hints() -> None:
    session_state = _session(parts={"LG": "misplaced", "MCP": "in_gripper"})
    session_state["candidate_revision_targets"] = [
        {
            "candidate_id": "candidate_lg",
            "resource_jid": "ur5e@localhost",
            "part_name": "LG",
            "expected_end_state": {
                "held_part": "LG",
                "part_state": "in_gripper",
                "part_location": "prusa-mk4-2",
            },
            "constraint_codes": ["held_part_location_mismatch"],
        },
        {
            "candidate_id": "candidate_mcp",
            "resource_jid": "ur5e@localhost",
            "part_name": "MCP",
            "expected_end_state": {
                "held_part": None,
                "part_state": "placed",
                "part_location": "prusa-mk4-2",
            },
            "constraint_codes": ["held_part_location_mismatch"],
        },
    ]
    prompt = multi_turn_prompts.render_multi_turn_phase_prompt(
        multi_turn_prompts.build_multi_turn_phase_prompt_input(
            phase="outline",
            llm_input={
                "observed_runtime_state": {"resources": []},
                "part_facts": [],
                "goal_conditions": [],
                "loaded_safety_rules": [],
            },
            session_state=session_state,
            recovery_resources={},
        )
    )

    assert "Candidate Revision Targets" in prompt
    assert "candidate_lg" in prompt
    assert "candidate_mcp" in prompt
    assert "one materially revised candidate for every listed target" in prompt
    assert "Changing only event_name, outline_id, or rationale" in prompt
    assert '"expected_end_state"' not in prompt
    assert "prusa-mk4-2" not in prompt
    assert "place_insert" not in prompt
    assert "move_home" not in prompt


def _carrier_rejection(
    *,
    candidate_index: int,
    resource_jid: str,
    part_name: str,
    destination: str,
) -> dict[str, Any]:
    return {
        "candidate_index": candidate_index,
        "task": {
            "outline_id": f"candidate_{candidate_index}",
            "event_name": f"rejected_event_{candidate_index}",
            "resource_jid": resource_jid,
            "part_name": part_name,
            "expected_end_state": {
                "resource_state": "idle",
                "held_part": None,
                "part_state": "restored",
                "part_location": destination,
            },
            "rationale": f"rejected rationale {candidate_index}",
        },
        "validation_findings": [
            {
                "validation_category": "transition_feasibility",
                "constraint_owner": "product",
                "constraint_code": "part_relocation_without_carrier",
                "resource_jid": resource_jid,
                "part_name": part_name,
                "reason": "direct relocation hides acquisition",
                "evidence": {
                    "field": "part_motion",
                    "changed_fields": ["part_location"],
                },
            }
        ],
    }


def test_atomic_custody_feedback_uses_only_the_responsible_ra_token() -> None:
    resource_jid = "ur5e@localhost"
    part_name = "LG"
    carried_location = "ur5e@localhost_gripper"
    session_state = {
        "symbolic_resources": {
            resource_jid: {
                "resource_jid": resource_jid,
                "resource_state": "idle",
                "held_part": None,
            },
            "xarm6@localhost": {
                "resource_jid": "xarm6@localhost",
                "resource_state": "idle",
                "held_part": None,
                "workspace_bounds": {"y": [-0.8, 0.1]},
            },
        },
        "symbolic_parts": {
            part_name: {
                "part_name": part_name,
                "part_state": "misplaced",
                "part_location": None,
                "current_location": None,
                "current_holder_resource_jid": None,
                "observed_pose": {"x": 0.1, "y": 0.2, "z": 0.3},
            }
        },
        "candidate_rejection_feedback": [
            {
                "candidate_index": 0,
                "task": {
                    "event_name": "rejected_xarm_acquisition",
                    "resource_jid": "xarm6@localhost",
                    "part_name": part_name,
                    "rationale": "old rejected rationale",
                },
                "validation_findings": [
                    {
                        "validation_category": "physical_feasibility",
                        "constraint_owner": "resource",
                        "constraint_code": "workspace_unreachable",
                        "resource_jid": "xarm6@localhost",
                        "part_name": part_name,
                        "reason": "LG pose is outside the xarm6 workspace.",
                        "durable": True,
                        "evidence": {
                            "checked_pose": {"x": 0.1, "y": 0.2, "z": 0.3},
                            "workspace_bounds": {"y": [-0.8, 0.1]},
                        },
                    }
                ],
            }
        ],
    }
    prepared = {
        "recovery_resources": {
            resource_jid: {
                "resource_type": "robot",
                "recovery_snapshot": {
                    "resource_type": "robot",
                    "held_part": None,
                },
            }
        }
    }
    current_rows = [
        _carrier_rejection(
            candidate_index=index,
            resource_jid=resource_jid,
            part_name=part_name,
            destination=destination,
        )
        for index, destination in enumerate(
            ["assembly_board-v1", "prusa-mk4-1", "prusa-mk4-2"]
        )
    ]

    feedback = multi_turn._merge_applicable_candidate_feedback(
        session_state=session_state,
        prepared_recovery_request=prepared,
        current_feedback_rows=current_rows,
        prune_current=False,
    )
    codes = [
        finding["constraint_code"]
        for row in feedback
        for finding in row.get("validation_findings") or []
    ]
    assert codes.count("part_relocation_without_carrier") == 1
    assert codes.count("workspace_unreachable") == 1
    summary = multi_turn_prompts._candidate_rejection_learning_summary(
        history=[],
        feedback_rows=feedback,
        feedback_render_style="des_event_diagnostic",
        resources_by_jid=deepcopy(session_state["symbolic_resources"]),
        parts_by_name=deepcopy(session_state["symbolic_parts"]),
    )
    assert "part_relocation_without_carrier" in summary
    assert carried_location not in summary
    assert "direct relocation hides acquisition" in summary
    assert "xarm6@localhost_gripper" not in summary
    assert "rejected_event_" not in summary
    assert "rejected rationale" not in summary

    session_state["candidate_rejection_feedback"] = deepcopy(feedback)
    no_progress = {
        "validation_category": "model_based_selection",
        "constraint_owner": "product",
        "constraint_code": "no_progressing_candidate",
        "reason": "current turn made no symbolic progress",
    }
    feedback = multi_turn._merge_applicable_candidate_feedback(
        session_state=session_state,
        prepared_recovery_request=prepared,
        current_feedback_rows=[no_progress],
        prune_current=False,
    )
    assert {
        finding["constraint_code"]
        for row in feedback
        for finding in row.get("validation_findings") or []
    } == {
        "workspace_unreachable",
        "part_relocation_without_carrier",
        "no_progressing_candidate",
    }

    session_state["candidate_rejection_feedback"] = deepcopy(feedback)
    session_state["symbolic_resources"][resource_jid]["held_part"] = part_name
    session_state["symbolic_parts"][part_name].update(
        {
            "part_location": carried_location,
            "current_location": carried_location,
            "current_holder_resource_jid": resource_jid,
        }
    )
    assert multi_turn._merge_applicable_candidate_feedback(
        session_state=session_state,
        prepared_recovery_request=prepared,
        current_feedback_rows=[],
        prune_current=True,
    ) == []


def test_custody_correction_fails_closed_without_a_declared_location() -> None:
    row = _carrier_rejection(
        candidate_index=0,
        resource_jid="printer@localhost",
        part_name="P",
        destination="station",
    )
    session_state = {
        "symbolic_resources": {
            "printer@localhost": {
                "resource_jid": "printer@localhost",
                "resource_state": "paused",
                "held_part": None,
            }
        },
        "symbolic_parts": {
            "P": {
                "part_name": "P",
                "part_location": None,
                "current_holder_resource_jid": None,
            }
        },
        "candidate_rejection_feedback": [],
    }
    prepared = {
        "recovery_resources": {
            "printer@localhost": {
                "resource_type": "printing",
                "recovery_snapshot": {"resource_type": "printing"},
            }
        }
    }
    feedback = multi_turn._merge_applicable_candidate_feedback(
        session_state=session_state,
        prepared_recovery_request=prepared,
        current_feedback_rows=[row],
        prune_current=False,
    )
    finding = feedback[0]["validation_findings"][0]
    assert "carried_part_location" not in finding["evidence"]
    summary = multi_turn_prompts._candidate_rejection_learning_summary(
        history=[],
        feedback_rows=feedback,
        feedback_render_style="des_event_diagnostic",
    )
    assert "First establish custody in a separate transition" not in summary


def test_non_robot_custody_correction_uses_its_exact_declared_token() -> None:
    resource_type = "fixture_handler_test"
    resource_jid = "fixture@localhost"
    carried_location = "fixture@localhost_fixture_slot"
    register_resource_profile(
        ResourceProfile(
            resource_type=resource_type,
            carried_entity_field="payload",
            carried_entity_location_builder=(
                lambda jid, _snapshot: f"{jid}_fixture_slot"
            ),
        )
    )
    session_state = {
        "symbolic_resources": {
            resource_jid: {
                "resource_jid": resource_jid,
                "resource_state": "idle",
                "payload": None,
            }
        },
        "symbolic_parts": {
            "P": {
                "part_name": "P",
                "part_location": None,
                "current_holder_resource_jid": None,
            }
        },
            "candidate_rejection_feedback": [],
    }
    prepared = {
        "recovery_resources": {
            resource_jid: {
                "resource_type": resource_type,
                "recovery_snapshot": {"resource_type": resource_type},
            }
        }
    }
    feedback = multi_turn._merge_applicable_candidate_feedback(
        session_state=session_state,
        prepared_recovery_request=prepared,
        current_feedback_rows=[
            _carrier_rejection(
                candidate_index=0,
                resource_jid=resource_jid,
                part_name="P",
                destination="station",
            )
        ],
        prune_current=False,
    )
    finding = feedback[0]["validation_findings"][0]
    assert finding["evidence"]["carried_part_location"] == carried_location


def test_non_robot_custody_validation_uses_its_declared_carried_location() -> None:
    resource_type = "fixture_custody_validation"
    resource_jid = "fixture_validation@localhost"
    carried_location = "fixture_validation@localhost_fixture_slot"
    register_resource_profile(
        ResourceProfile(
            resource_type=resource_type,
            carried_entity_field="payload",
            carried_entity_location_builder=(
                lambda jid, _snapshot: f"{jid}_fixture_slot"
            ),
        )
    )
    session_state = {
        "symbolic_resources": {
            resource_jid: {
                "resource_jid": resource_jid,
                "resource_state": "idle",
                "held_part": None,
                "payload": None,
            }
        },
        "symbolic_parts": {
            "P": {
                "part_name": "P",
                "part_state": "loose",
                "part_location": None,
                "part_holder_resource_jid": None,
                "observed_pose": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
        },
    }
    prepared = {
        "llm_input": {"part_facts": []},
        "recovery_resources": {
            resource_jid: {
                "resource_type": resource_type,
                "recovery_snapshot": {"resource_type": resource_type},
            }
        },
    }

    def _validate(part_location: str) -> list[dict[str, Any]]:
        findings, _grounded_action = (
            recovery_validation_service.validate_recovery_outline_task(
                planner=object(),
                task={
                    "outline_id": "fixture_acquisition",
                    "event_name": "opaque_fixture_event",
                    "resource_jid": resource_jid,
                    "part_name": "P",
                    "expected_start_state": {
                        "resource_state": "idle",
                        "held_part": None,
                        "part_state": "loose",
                        "part_location": None,
                    },
                    "expected_end_state": {
                        "resource_state": "idle",
                        "held_part": "P",
                        "part_state": "loose",
                        "part_location": part_location,
                    },
                    "rationale": "Exercise the declared custody token.",
                },
                session_state=session_state,
                prepared_recovery_request=prepared,
            )
        )
        return findings

    assert _validate(carried_location) == []
    assert _validate("station") == []

    unavailable_type = "fixture_custody_validation_unavailable"
    register_resource_profile(ResourceProfile(resource_type=unavailable_type))
    prepared["recovery_resources"][resource_jid]["resource_type"] = unavailable_type
    prepared["recovery_resources"][resource_jid]["recovery_snapshot"][
        "resource_type"
    ] = unavailable_type
    assert _validate(carried_location) == []


def test_cca_projects_dfa_state_without_mutating_a_live_monitor() -> None:
    rule = {
        "id": "SAFE_TEST",
        "ap_scope": "recovery",
        "dfa_dot": (
            'digraph { init -> 1; 1 -> 2 [label="ap001"]; '
            '1 -> 1 [label="!ap001"]; 2 -> 2 [label="!ap001"]; }'
        ),
        "recovery_aps": [
            {
                "label": "ap001",
                "full": "ap_event/assembly/any/any/authored_event/destination=station",
                "selector": {
                    "mode": "resource_move_to_destination",
                    "resource": "any",
                    "part": "any",
                    "destination": "station",
                },
            }
        ],
    }
    validation_input = {
        "task": {
            "outline_id": "candidate",
            "event_name": "uninterpreted_name",
            "resource_jid": "resource@localhost",
            "expected_start_state": {"resource_state": "idle"},
            "expected_end_state": {
                "resource_state": "new_state",
                "resource_location": "station",
            },
        },
        "signature": {"task_kind": "resource_only"},
        "pre_resources": {
            "resource@localhost": {
                "current_state": "idle",
                "current_location": "elsewhere",
            }
        },
        "pre_parts": {},
        "projected_resources": {
            "resource@localhost": {
                "current_state": "new_state",
                "current_location": "station",
            }
        },
        "projected_parts": {},
        "llm_input": {"loaded_safety_rules": [rule]},
    }

    first = validate_outline_macro_recovery_safety(**validation_input)
    second = validate_outline_macro_recovery_safety(
        **validation_input,
        safety_dfa_states_before=deepcopy(first["safety_dfa_states_after"]),
    )

    assert first["safety_dfa_states_before"] == {"SAFE_TEST": "1"}
    assert first["safety_dfa_states_after"] == {"SAFE_TEST": "2"}
    assert second["safety_dfa_states_before"] == {"SAFE_TEST": "2"}
    assert second["safety_dfa_states_after"] == {"SAFE_TEST": "2"}
    with pytest.raises(ValueError, match="invalid"):
        validate_outline_macro_recovery_safety(
            **validation_input,
            safety_dfa_states_before={"SAFE_TEST": "stale_state"},
        )


def test_session_seed_preserves_pure_llm_and_enables_neurosymbolic_budget() -> None:
    pure = multi_turn.build_multi_turn_session_seed(
        {"recovery_session": {"recovery_selection_mode": "pure_llm"}}
    )
    neuro = multi_turn.build_multi_turn_session_seed(
        {
            "recovery_session": {
                "recovery_selection_mode": "neurosymbolic",
                "action_horizon": "full",
                "candidate_count": 3,
                "candidate_proposal_budget": 7,
            }
        }
    )

    assert pure["candidate_count"] == 3
    assert pure["recovery_selection_mode"] == "pure_llm"
    assert neuro["recovery_selection_mode"] == "neurosymbolic"
    assert neuro["action_horizon"] == "1"
    assert neuro["candidate_count"] == "adaptive"
    assert neuro["candidate_bound"] == 7
