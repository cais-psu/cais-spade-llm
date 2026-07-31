"""Focused tests for action_target binding authority in recovery outlines.

Product validates action_target shape only; grounding an action_target against
authored end-state locations is Resource Agent authority. These tests pin that
split so a held part's gripper location is never rejected by Product before the
owning Resource Agent can adjudicate it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
    multi_turn,
    multi_turn_prompts,
)
from cais_spade_llm.agents.resource_agent.printing_agent import PrintingAgent
from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.resources.capability_engine import compound_recovery_state

UR5E_JID = "ur5e@localhost"
UR5E_GRIPPER = "ur5e@localhost_gripper"

_CARTESIAN_CONTRACT = {
    "frame": "world",
    "units": "m",
    "fields": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
    },
    "required": ["x", "y", "z"],
}


def _session_state() -> dict[str, Any]:
    return {
        "public_action_targets": [
            {
                "resource_jid": UR5E_JID,
                "action_target": deepcopy(_CARTESIAN_CONTRACT),
            }
        ],
        "public_locations": [
            {
                "resource_jid": UR5E_JID,
                "locations": [
                    {"location": "home"},
                    {"location": "prusa-mk4-2"},
                    {"location": "assembly_board-v1"},
                ],
            }
        ],
    }


def _held_part_candidate(**overrides: Any) -> dict[str, Any]:
    candidate = {
        "outline_id": "recovery_candidate_1",
        "event_name": "move_with_held_part",
        "resource_jid": UR5E_JID,
        "part_name": "MCP",
        "action_target": {"x": 0.4, "y": 0.3, "z": 1.2},
        "expected_end_state": compound_recovery_state({
            "resource_state": "picked",
            "part_state": "in_gripper",
            "part_location": UR5E_GRIPPER,
        }),
        "rationale": "Lift MCP clear of the board while keeping it grasped.",
    }
    candidate.update(deepcopy(overrides))
    return candidate


def _findings(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    return multi_turn._candidate_action_target_findings(
        candidate_task=candidate,
        session_state=_session_state(),
        resource_jid=UR5E_JID,
        part_name=str(candidate.get("part_name") or ""),
    )


def test_held_part_gripper_location_passes_product_binding() -> None:
    """A held part's gripper token is a holder, not a competing destination."""
    assert _findings(_held_part_candidate()) == []


def test_cartesian_target_allows_authored_resource_location() -> None:
    """Product no longer adjudicates location semantics against a pose."""
    candidate = _held_part_candidate(
        expected_end_state=compound_recovery_state({
            "resource_state": "picked",
            "resource_location": "prusa-mk4-2",
            "part_state": "in_gripper",
            "part_location": UR5E_GRIPPER,
        })
    )
    assert _findings(candidate) == []


def test_null_authored_location_passes_product_binding() -> None:
    candidate = _held_part_candidate(
        expected_end_state=compound_recovery_state({
            "resource_state": "picked",
            "resource_location": None,
            "part_state": "in_gripper",
            "part_location": None,
        })
    )
    assert _findings(candidate) == []


def test_product_still_rejects_action_target_shape_violations() -> None:
    """Contract-driven shape checks stay with Product and remain generic."""
    unknown_field = _held_part_candidate(
        action_target={"x": 0.4, "y": 0.3, "z": 1.2, "yaw": 0.0}
    )
    findings = _findings(unknown_field)
    assert findings
    assert findings[0]["constraint_code"] == "candidate_schema_violation"
    assert findings[0]["evidence"]["field"] == "action_target"

    missing_required = _held_part_candidate(action_target={"x": 0.4, "y": 0.3})
    findings = _findings(missing_required)
    assert findings
    assert findings[0]["constraint_code"] == "candidate_schema_violation"

    wrong_type = _held_part_candidate(
        action_target={"x": 0.4, "y": 0.3, "z": "high"}
    )
    findings = _findings(wrong_type)
    assert findings
    assert findings[0]["evidence"]["field"] == "action_target.z"

    unknown_location = _held_part_candidate(action_target={"location": "nowhere"})
    findings = _findings(unknown_location)
    assert findings
    assert findings[0]["evidence"]["field"] == "action_target.location"


def test_known_named_location_still_passes() -> None:
    candidate = _held_part_candidate(action_target={"location": "prusa-mk4-2"})
    assert _findings(candidate) == []


def test_non_robot_resource_refuses_action_target_from_the_resource_agent() -> None:
    """Generalization: refusal is Resource Agent authority, not Product's."""
    session_state = _session_state()
    session_state["public_action_targets"] = [
        {
            "resource_jid": "prusa-mk4-2@localhost",
            "action_target": {
                "fields": {"material": {"type": "string"}},
                "required": ["material"],
            },
        }
    ]
    candidate = {
        "outline_id": "recovery_candidate_1",
        "event_name": "reload_filament",
        "resource_jid": "prusa-mk4-2@localhost",
        "action_target": {"material": "PLA"},
        "expected_end_state": compound_recovery_state({
            "resource_state": "idle",
            "resource_location": "prusa-mk4-2",
        }),
        "rationale": "Reload the printer before resuming the job.",
    }
    # Product accepts the declared shape; no Cartesian vocabulary is involved.
    assert (
        multi_turn._candidate_action_target_findings(
            candidate_task=candidate,
            session_state=session_state,
            resource_jid="prusa-mk4-2@localhost",
            part_name="",
        )
        == []
    )

    printing_agent = PrintingAgent.__new__(PrintingAgent)
    printing_agent.jid = "prusa-mk4-2@localhost"
    _successor, conflict = ResourceAgent._generated_successor_from_action_target(
        printing_agent,
        task=candidate,
        initial_valuation={},
        successor={},
        runtime_fact_tables=[],
    )
    assert conflict is not None
    assert conflict["constraint_code"] == "unsupported_resource_target"


def test_recovery_goals_render_goal_lines_only() -> None:
    """Recovery Goals states the goal lines and nothing more."""
    llm_input = {
        "goal_conditions": [
            {
                "entity_kind": "part",
                "entity": "LG",
                "field": "location",
                "expected": "assembly_board-v1",
            },
            {
                "entity_kind": "part",
                "entity": "LG",
                "field": "state",
                "expected": "assembled",
            },
        ]
    }
    goals = multi_turn_prompts._compact_recovery_goals(
        llm_input,
        projected_parts=[],
    )
    assert goals == "- restore LG to assembly_board-v1 with part_state=assembled"
