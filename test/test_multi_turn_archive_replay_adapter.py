from __future__ import annotations

from copy import deepcopy

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn import (
    build_multi_turn_bridge_proposal,
)


def _prepared_bridge_request() -> dict:
    return {
        "bridge_resources": {
            "xarm6@localhost": {
                "bridge_snapshot": {
                    "resource_jid": "xarm6@localhost",
                    "resource_type": "robot",
                    "resource_state": "failed",
                    "current_state": "failed",
                    "current_location": None,
                    "held_part": None,
                    "gripper_state": "open",
                }
            }
        },
        "llm_input": {
            "observed_runtime_state": {
                "resources": [
                    {
                        "resource_jid": "xarm6@localhost",
                        "current_state": "failed",
                    }
                ]
            },
            "part_facts": [],
        },
        "grounding_context": {"parts": {}},
    }


def _final_output_payload() -> dict:
    root_event = {
        "outline_id": "RECOVERY_SEQ1",
        "event_name": "recover_to_home_idle",
        "resource_jid": "xarm6@localhost",
        "description": "Recover xarm6 to home idle.",
        "expected_start_state": {
            "resource_state": "failed",
        },
        "expected_end_state": {
            "resource_state": "idle",
            "resource_location": "home",
        },
    }
    primitive_row = {
        "outline_id": "RECOVERY_SEQ1",
        "des_event_id": "RECOVERY_SEQ1",
        "resource_jid": "xarm6@localhost",
        "event_name": "recover_to_home_idle",
        "description": "Recover xarm6 to home idle.",
        "predecessors": [],
        "primitive_steps": [
            {
                "primitive": "move_to_named_pose",
                "params": {"pose_name": "home"},
            }
        ],
        "projected_snapshot": {
            "resource_jid": "xarm6@localhost",
            "resource_type": "robot",
            "resource_state": "failed",
            "current_state": "idle",
            "current_location": None,
            "held_part": None,
            "gripper_state": "open",
            "resource_core": {
                "resource_type": "robot",
                "current_state": "idle",
                "occupancy": {"location": "home"},
            },
            "resource_facets": {
                "manipulator": {
                    "current_pose_ref": "home",
                    "current_pose": None,
                }
            },
            "occupancy": {"location": "home"},
            "current_pose_ref": "home",
            "current_pose": None,
        },
        "projected_outline_state": {
            "resource_state": "idle",
            "resource_location": "home",
        },
    }
    return {
        "engine": "multi_turn",
        "decision": "final_output_ready",
        "final_output_stage": "primitive_program_ready",
        "primitive_program_complete": True,
        "transition_trace": [deepcopy(root_event)],
        "executable_recovery_trace": [
            {
                **deepcopy(root_event),
                "predecessors": [],
                "des_event_id": "RECOVERY_SEQ1",
                "part_name": None,
                "target_ref": None,
                "primitive_steps": deepcopy(primitive_row["primitive_steps"]),
            }
        ],
        "accepted_primitive_program": [deepcopy(primitive_row)],
    }


def test_build_multi_turn_bridge_proposal_prefers_executable_recovery_trace() -> None:
    result = build_multi_turn_bridge_proposal(
        final_output_payload=_final_output_payload(),
        prepared_bridge_request=_prepared_bridge_request(),
    )

    assert result["accepted"] is True
    assert result["reason"] == ""
    assert result["bridge_proposal"]["macro_tasks"][0]["predecessors"] == []


def test_build_multi_turn_bridge_proposal_uses_primitive_predecessors_fallback() -> None:
    payload = _final_output_payload()
    payload.pop("executable_recovery_trace", None)

    result = build_multi_turn_bridge_proposal(
        final_output_payload=payload,
        prepared_bridge_request=_prepared_bridge_request(),
    )

    assert result["accepted"] is True
    assert result["reason"] == ""
    assert result["bridge_proposal"]["macro_tasks"][0]["predecessors"] == []


def test_build_multi_turn_bridge_proposal_merges_grounding_part_state_for_observed_pose() -> None:
    payload = {
        "engine": "multi_turn",
        "decision": "final_output_ready",
        "final_output_stage": "primitive_program_ready",
        "primitive_program_complete": True,
        "executable_recovery_trace": [
            {
                "outline_id": "RECOVERY_SEQ3",
                "des_event_id": "RECOVERY_SEQ3",
                "event_name": "recover_pick_LG_from_observed_pose",
                "resource_jid": "ur5e@localhost",
                "part_name": "LG",
                "predecessors": [],
                "description": "Pick LG from observed pose.",
                "expected_start_state": {
                    "resource_state": "idle",
                    "held_part": None,
                    "part_state": "misplaced",
                },
                "expected_end_state": {
                    "resource_state": "idle",
                    "held_part": "LG",
                    "part_state": "held",
                },
                "primitive_steps": [
                    {
                        "primitive": "compute_pick_targets",
                        "params": {"part_name": "LG", "target_pose_source": "observed_pose"},
                    }
                ],
            }
        ],
        "accepted_primitive_program": [
            {
                "outline_id": "RECOVERY_SEQ3",
                "des_event_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "part_name": "LG",
                "event_name": "recover_pick_LG_from_observed_pose",
                "description": "Pick LG from observed pose.",
                "predecessors": [],
                "primitive_steps": [
                    {
                        "primitive": "compute_pick_targets",
                        "params": {"part_name": "LG", "target_pose_source": "observed_pose"},
                    }
                ],
                "projected_snapshot": {
                    "resource_jid": "ur5e@localhost",
                    "resource_type": "robot",
                    "resource_state": "idle",
                    "current_state": "idle",
                    "current_location": None,
                    "held_part": "LG",
                    "gripper_state": "closed",
                    "resource_core": {
                        "resource_type": "robot",
                        "current_state": "idle",
                    },
                    "resource_facets": {},
                },
                "projected_outline_state": {
                    "resource_state": "idle",
                    "held_part": "LG",
                    "part_state": "held",
                },
            }
        ],
    }
    prepared_bridge_request = {
        "bridge_resources": {
            "ur5e@localhost": {
                "bridge_snapshot": {
                    "resource_jid": "ur5e@localhost",
                    "resource_type": "robot",
                    "resource_state": "idle",
                    "current_state": "idle",
                    "current_location": None,
                    "held_part": None,
                    "gripper_state": "open",
                }
            }
        },
        "llm_input": {
            "observed_runtime_state": {
                "resources": [
                    {
                        "resource_jid": "ur5e@localhost",
                        "current_state": "idle",
                        "held_part": None,
                    }
                ]
            },
            "part_facts": [
                {
                    "part_name": "LG",
                    "current_state": "unknown",
                    "current_location": None,
                    "current_holder_resource_jid": None,
                    "observed_pose": None,
                }
            ],
        },
        "grounding_context": {
            "parts": {
                "LG": {
                    "state": "misplaced",
                    "location": None,
                    "observed_pose": {"x": 0.0, "y": 0.2, "z": 1.035},
                }
            }
        },
    }

    result = build_multi_turn_bridge_proposal(
        final_output_payload=payload,
        prepared_bridge_request=prepared_bridge_request,
    )

    assert result["accepted"] is True
    assert result["reason"] == ""
