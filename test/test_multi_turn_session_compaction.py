from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bootstrap_repo_site_packages(root: Path) -> None:
    venv_lib = root / ".venv" / "lib"
    if not venv_lib.exists():
        return
    for site_packages in sorted(venv_lib.glob("python*/site-packages")):
        site_path = str(site_packages.resolve())
        if site_path not in sys.path:
            sys.path.insert(0, site_path)


_bootstrap_repo_site_packages(ROOT)

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (
    compact_multi_turn_runtime_session,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_prompts import (
    _candidate_rejection_history,
    _outline_rejection_history,
)


def test_compact_multi_turn_runtime_session_drops_heavy_old_turn_fields() -> None:
    session_state = {
        "turns": [
            {
                "turn_index": 1,
                "phase": "primitive_generation",
                "decision": "need_context",
                "outline_id": "RECOVERY_SEQ4",
                "resource_jid": "ur5e@localhost",
                "primitive_local_turn_index": 1,
                "prompt_text": "X" * 2048,
                "prompt_input": {"large": ["payload"] * 64},
                "llm_raw_response": {"thought": "heavy"},
                "raw_response": {
                    "decision": "need_context",
                    "primitive_steps": [],
                    "notes": ["kept compact"],
                },
                "primitive_substream_turns": [
                    {
                        "outline_id": "RECOVERY_SEQ4",
                        "resource_jid": "ur5e@localhost",
                        "primitive_local_turn_index": 1,
                        "decision": "need_context",
                        "prompt_text": "subturn heavy",
                    }
                ],
            },
            {
                "turn_index": 2,
                "phase": "outline",
                "decision": "outline_ready",
                "prompt_text": "latest prompt stays available",
                "prompt_input": {"latest": True},
                "next_transition": {"outline_id": "RECOVERY_SEQ5"},
            },
        ]
    }

    compact_multi_turn_runtime_session(session_state, keep_full_turns=1)

    older_turn = session_state["turns"][0]
    latest_turn = session_state["turns"][1]

    assert "prompt_text" not in older_turn
    assert "prompt_input" not in older_turn
    assert "llm_raw_response" not in older_turn
    assert older_turn["primitive_substream_turns"] == [
        {
            "outline_id": "RECOVERY_SEQ4",
            "resource_jid": "ur5e@localhost",
            "primitive_local_turn_index": 1,
            "decision": "need_context",
        }
    ]
    assert older_turn["raw_response"]["decision"] == "need_context"
    assert latest_turn["prompt_text"] == "latest prompt stays available"


def test_compaction_preserves_outline_and_candidate_rejection_history() -> None:
    session_state = {
        "turns": [
            {
                "turn_index": 1,
                "phase": "outline",
                "decision": "need_revision",
                "next_transition": {
                    "outline_id": "RECOVERY_SEQ1",
                    "resource_jid": "xarm6@localhost",
                    "event_name": "recover_to_home_idle",
                },
                "validation_findings": [
                    {
                        "constraint_code": "unknown_location_binding",
                        "reason": "home is not grounded",
                    }
                ],
                "prompt_text": "old outline prompt",
            },
            {
                "turn_index": 2,
                "phase": "outline",
                "decision": "need_revision",
                "candidate_evaluations": [
                    {
                        "candidate_index": 0,
                        "task": {
                            "resource_jid": "ur5e@localhost",
                            "part_name": "LG",
                            "target_ref": "assembly_board-v1",
                        },
                        "validation_findings": [
                            {
                                "constraint_code": "workspace_unreachable",
                                "reason": "target is outside workspace",
                            }
                        ],
                        "validated_task": {"ignored": True},
                        "surface_task": {"ignored": True},
                        "progress_detail": {"remaining": 1},
                    }
                ],
                "prompt_text": "candidate rejection prompt",
            },
            {
                "turn_index": 3,
                "phase": "outline",
                "decision": "need_next_task",
                "selected_transition": {
                    "outline_id": "RECOVERY_SEQ2",
                    "resource_jid": "xarm6@localhost",
                },
                "next_transition": {
                    "outline_id": "RECOVERY_SEQ2",
                    "resource_jid": "xarm6@localhost",
                },
                "prompt_text": "accepted outline prompt",
            },
            {
                "turn_index": 4,
                "phase": "outline",
                "decision": "need_revision",
                "candidate_evaluations": [
                    {
                        "candidate_index": 1,
                        "task": {
                            "resource_jid": "ur5e@localhost",
                            "part_name": "MCP",
                            "target_ref": "assembly_board-v1",
                        },
                        "validation_findings": [
                            {
                                "constraint_code": "dependency_unsatisfied",
                                "reason": "REQ_2_T3 is incomplete",
                            }
                        ],
                    }
                ],
                "prompt_text": "fresh candidate rejection prompt",
            },
        ]
    }

    outline_history_before = _outline_rejection_history(session_state)
    candidate_history_before = _candidate_rejection_history(session_state)

    compact_multi_turn_runtime_session(session_state, keep_full_turns=0)

    assert _outline_rejection_history(session_state) == outline_history_before
    assert _candidate_rejection_history(session_state) == candidate_history_before
    assert len(candidate_history_before) == 1
    assert candidate_history_before[0]["turn_index"] == 4
