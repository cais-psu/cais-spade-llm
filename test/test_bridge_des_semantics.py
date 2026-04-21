from __future__ import annotations

from copy import deepcopy

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_des_semantics import (
    bridge_process_schema_registry,
    build_bridge_validation_context,
    normalize_surface_bridge_proposal,
    parse_bridge_event_instance,
    parse_surface_bridge_proposal,
    validate_bridge_event_instance,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    multi_turn_prompts,
)


def _prepared_bridge_request() -> dict[str, object]:
    return {
        "bridge_resources": {
            "ur5e@localhost": {
                "resource_type": "robot",
                "bridge_snapshot": {
                    "resource_type": "robot",
                    "reachable_locations": ["assembly_board-v1", "prusa-mk4-2"],
                    "available_named_poses": ["home"],
                },
                "static_capabilities": {
                    "resource_type": "robot",
                    "reachability": ["assembly_board-v1", "prusa-mk4-2"],
                },
            },
            "xarm6@localhost": {
                "resource_type": "robot",
                "bridge_snapshot": {
                    "resource_type": "robot",
                    "reachable_locations": ["home", "assembly_board-v1"],
                },
            },
        },
        "grounding_context": {
            "parts": {
                "LG": {
                    "target": {"location": "assembly_board-v1"},
                }
            }
        },
        "llm_input": {
            "fault_event": {"affected_part_names": ["LG"]},
        },
    }


def _turn6_context() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, object]]:
    resources_by_jid = {
        "ur5e@localhost": {
            "resource_jid": "ur5e@localhost",
            "resource_type": "robot",
            "current_state": "idle",
            "held_part": None,
            "gripper_state": "open",
        },
        "xarm6@localhost": {
            "resource_jid": "xarm6@localhost",
            "resource_type": "robot",
            "current_state": "idle",
            "held_part": None,
            "gripper_state": "open",
        },
    }
    parts_by_name = {
        "LG": {
            "part_name": "LG",
            "current_state": "misplaced",
            "current_location": None,
            "observed_pose": {"x": 0.0, "y": 0.2, "z": 1.035},
            "current_holder_resource_jid": None,
            "goal_location": "assembly_board-v1",
        }
    }
    prepared_bridge_request = _prepared_bridge_request()
    return resources_by_jid, parts_by_name, prepared_bridge_request


def test_bridge_schema_registry_includes_required_schemas() -> None:
    registry = bridge_process_schema_registry()

    assert {
        "recover_resource_idle",
        "pick_part",
        "place_part",
        "stage_part",
        "resume_nominal_task",
    }.issubset(registry)


def test_turn6_place_part_fails_at_plant_enabledness_without_control() -> None:
    resources_by_jid, parts_by_name, prepared_bridge_request = _turn6_context()
    context = build_bridge_validation_context(
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        prepared_bridge_request=prepared_bridge_request,
    )
    instance = parse_bridge_event_instance(
        {
            "event_schema_id": "place_part",
            "resource_binding": "ur5e@localhost",
            "object_bindings": {
                "part": "LG",
                "target_location": "assembly_board-v1",
            },
            "parameters": {},
            "depends_on": [],
            "rationale": "Place LG back on the assembly board.",
        },
        outline_id="RECOVERY_SEQ3_1",
    )

    result = validate_bridge_event_instance(instance, context=context)

    assert not result.ok
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.stage == "plant_enabledness"
    assert finding.code == "unsatisfied_guard_predicate"
    assert finding.unsatisfied_predicates == ["holds(ur5e@localhost, LG)"]
    assert "establish control" in finding.retry_hint


def test_pick_part_then_place_part_becomes_enabled_in_projected_state() -> None:
    resources_by_jid, parts_by_name, prepared_bridge_request = _turn6_context()
    context = build_bridge_validation_context(
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        prepared_bridge_request=prepared_bridge_request,
    )
    pick_instance = parse_bridge_event_instance(
        {
            "event_schema_id": "pick_part",
            "resource_binding": "ur5e@localhost",
            "object_bindings": {
                "part": "LG",
                "source_location": "observed_pose",
            },
            "parameters": {},
            "depends_on": [],
            "rationale": "Pick LG from the observed pose.",
        },
        outline_id="RECOVERY_SEQ3",
    )

    pick_result = validate_bridge_event_instance(pick_instance, context=context)

    assert pick_result.ok
    assert pick_result.normalized_task is not None
    assert pick_result.projected_resources["ur5e@localhost"]["held_part"] == "LG"
    assert pick_result.projected_parts["LG"]["current_holder_resource_jid"] == "ur5e@localhost"

    place_context = build_bridge_validation_context(
        resources_by_jid=deepcopy(pick_result.projected_resources),
        parts_by_name=deepcopy(pick_result.projected_parts),
        prepared_bridge_request=prepared_bridge_request,
    )
    place_instance = parse_bridge_event_instance(
        {
            "event_schema_id": "place_part",
            "resource_binding": "ur5e@localhost",
            "object_bindings": {
                "part": "LG",
                "target_location": "assembly_board-v1",
            },
            "parameters": {},
            "depends_on": ["RECOVERY_SEQ3"],
            "rationale": "Place LG on the assembly board.",
        },
        outline_id="RECOVERY_SEQ4",
    )

    place_result = validate_bridge_event_instance(place_instance, context=place_context)

    assert place_result.ok
    assert place_result.normalized_task is not None
    assert place_result.normalized_task["expected_end_state"]["part_location"] == "assembly_board-v1"
    assert place_result.normalized_task["expected_end_state"]["part_state"] == "assembled"


def test_surface_return_to_board_normalizes_to_place_part() -> None:
    resources_by_jid, parts_by_name, prepared_bridge_request = _turn6_context()
    context = build_bridge_validation_context(
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        prepared_bridge_request=prepared_bridge_request,
    )
    proposal = parse_surface_bridge_proposal(
        {
            "surface_event_name": "return_lg_to_board",
            "surface_description": "return LG back to the assembly board",
            "resource_binding": "ur5e@localhost",
            "object_bindings": {
                "part": "LG",
                "target_location": "assembly_board-v1",
            },
            "parameters": {},
            "depends_on": [],
            "rationale": "Return LG to the board.",
        },
        outline_id="RECOVERY_SEQ3_1",
    )

    normalized_event, findings = normalize_surface_bridge_proposal(proposal, context=context)

    assert findings == []
    assert normalized_event is not None
    assert normalized_event.event_schema_id == "place_part"


def test_compound_surface_proposal_is_rejected_for_atomic_split() -> None:
    resources_by_jid, parts_by_name, prepared_bridge_request = _turn6_context()
    context = build_bridge_validation_context(
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        prepared_bridge_request=prepared_bridge_request,
    )
    proposal = parse_surface_bridge_proposal(
        {
            "surface_event_name": "pick_and_place_lg",
            "surface_description": "pick LG from observed_pose and place it on the assembly board",
            "resource_binding": "ur5e@localhost",
            "object_bindings": {
                "part": "LG",
                "source_location": "observed_pose",
                "target_location": "assembly_board-v1",
            },
            "parameters": {},
            "depends_on": [],
            "rationale": "Do the whole pick and place in one event.",
        },
        outline_id="RECOVERY_SEQ3_1",
    )

    normalized_event, findings = normalize_surface_bridge_proposal(proposal, context=context)

    assert normalized_event is None
    assert len(findings) == 1
    finding = findings[0]
    assert finding.stage == "surface_normalization"
    assert finding.code == "compound_surface_event"
    assert "split" in finding.retry_hint


def test_outline_candidate_schema_uses_event_instances_not_expected_state_deltas() -> None:
    schema = multi_turn_prompts.multi_turn_phase_response_schema(
        "outline",
        outline_mode="incremental_candidates_validated",
        candidate_bound=3,
    )
    event_schema = schema["schema"]["properties"]["candidate_events"]["items"]

    assert set(event_schema["required"]) == {
        "resource_binding",
        "object_bindings",
        "parameters",
        "rationale",
    }
    assert "surface_event_name" in event_schema["properties"]
    assert "surface_description" in event_schema["properties"]
    assert "expected_start_state" not in event_schema["properties"]
    assert "expected_end_state" not in event_schema["properties"]
