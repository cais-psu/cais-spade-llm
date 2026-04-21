from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    multi_turn_prompts,
)


_TOP_FEEDBACK_LABEL = "Previous Validation Feedback (read first)"


def _feedback_block(prompt: str) -> str:
    start = prompt.index(_TOP_FEEDBACK_LABEL)
    block = prompt[start:]
    sentinel = "\n\n"
    next_break = block.find(sentinel, len(_TOP_FEEDBACK_LABEL))
    if next_break == -1:
        return block
    return block[:next_break]


def test_primitive_prompt_moves_previous_validation_feedback_to_top() -> None:
    prompt = multi_turn_prompts._render_primitive_generation_prompt(
        {
            "session_state": {
                "primitive_rejection_feedback": [
                    {
                        "outline_id": "RECOVERY_SEQ3",
                        "resource_jid": "ur5e@localhost",
                        "constraint_code": "primitive_projection_mismatch",
                        "reason": "projected current_state expected='idle' actual='picked'",
                    }
                ]
            },
            "primitive_active_outline_event_token": {
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
            },
            "primitive_active_outline_event": {
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "event_name": "ur5e_recover_lg_observed_pose_pick",
                "expected_start_state": {"resource_state": "idle"},
                "expected_end_state": {"resource_state": "idle", "held_part": "LG"},
            },
        }
    )

    assert _TOP_FEEDBACK_LABEL in prompt
    assert prompt.index(_TOP_FEEDBACK_LABEL) < prompt.index("Active DES Transition (token)")
    assert "primitive_projection_mismatch" in _feedback_block(prompt)
    assert "Validator Feedback (previous turn)" not in prompt


def test_outline_prompt_prioritizes_active_validation_findings_at_top() -> None:
    prompt = multi_turn_prompts._render_outline_prompt(
        {
            "llm_input": {},
            "bridge_resources": {},
            "session_state": {
                "outline_validation_findings": [
                    {
                        "task_id": "RECOVERY_SEQ3_1",
                        "resource_jid": "ur5e@localhost",
                        "part_name": "LG",
                        "constraint_code": "expected_start_state_mismatch",
                        "reason": "expected_start_state does not match the projected current state",
                    }
                ],
                "candidate_rejection_feedback": [
                    {
                        "task": {
                            "resource_jid": "xarm6@localhost",
                            "part_name": "SG",
                            "target_ref": "assembly_board-v1",
                        },
                        "validation_findings": [
                            {
                                "constraint_code": "workspace_unreachable",
                                "reason": "target pose is outside workspace",
                            }
                        ],
                    }
                ],
                "primitive_escalation_diagnostics": [
                    {
                        "outline_id": "RECOVERY_SEQ3",
                        "resource_jid": "ur5e@localhost",
                        "part_name": "LG",
                        "trigger": "no_progress_turns",
                        "reason": "primitive authoring stalled on the active event",
                    }
                ],
            },
        }
    )

    block = _feedback_block(prompt)
    assert _TOP_FEEDBACK_LABEL in prompt
    assert prompt.index(_TOP_FEEDBACK_LABEL) < prompt.index("Current DES State")
    assert "RECOVERY_SEQ3_1/ur5e@localhost/LG" in block
    assert "xarm6@localhost/SG" not in block
    assert "primitive authoring stalled" not in block


def test_outline_prompt_regression_surfaces_stuck_session_feedback_near_top() -> None:
    candidate_paths = [
        ROOT
        / "cais_spade_llm"
        / "monitor"
        / "debug"
        / "multi_turn_session_session_20260420T160129.txt",
        ROOT
        / "cais_spade_llm"
        / "monitor"
        / "debug"
        / "multi_turn_session_session_20260421T133838.txt",
        ROOT
        / "cais_spade_llm"
        / "monitor"
        / "debug"
        / "worked"
        / "2"
        / "multi_turn_session_session_20260418T232908.txt",
    ]
    session_path = next(path for path in candidate_paths if path.exists())
    session_state = json.loads(session_path.read_text(encoding="utf-8"))

    prompt = multi_turn_prompts._render_outline_prompt(
        {
            "llm_input": {},
            "bridge_resources": {},
            "session_state": session_state,
        }
    )
    if _TOP_FEEDBACK_LABEL not in prompt:
        prompt = multi_turn_prompts._render_outline_prompt(
            {
                "llm_input": {},
                "bridge_resources": {},
                "session_state": {
                    "primitive_escalation_diagnostics": [
                        {
                            "outline_id": "RECOVERY_SEQ3",
                            "resource_jid": "ur5e@localhost",
                            "part_name": "LG",
                            "trigger": "no_progress_turns",
                            "reason": "primitive authoring stalled on the active event",
                        }
                    ]
                },
            }
        )

    block = _feedback_block(prompt)
    assert _TOP_FEEDBACK_LABEL in prompt
    assert prompt.index(_TOP_FEEDBACK_LABEL) < prompt.index("Current DES State")
    assert "primitive authoring stalled on the active event" in block
    assert "Primitive Escalation Diagnostics" in prompt
