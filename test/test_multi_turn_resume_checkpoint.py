from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DIR = ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))


def _bootstrap_repo_site_packages(root: Path) -> None:
    venv_lib = root / ".venv" / "lib"
    if not venv_lib.exists():
        return
    for site_packages in sorted(venv_lib.glob("python*/site-packages")):
        site_path = str(site_packages.resolve())
        if site_path not in sys.path:
            sys.path.insert(0, site_path)


_bootstrap_repo_site_packages(ROOT)

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (  # noqa: E402
    write_bridge_artifacts,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn import (  # noqa: E402
    _build_final_output_payload,
)
import test_case3_bridge_dryrun as case3_dryrun  # noqa: E402


def test_write_bridge_artifacts_writes_multi_turn_resume_checkpoint(
    tmp_path: Path,
) -> None:
    session_state = {
        "session_id": "CASE3",
        "turn_index": 2,
        "current_phase": "outline",
        "turns": [
            {
                "turn_index": 2,
                "phase": "outline",
                "prompt_text": "resume me",
                "raw_response": {"decision": "need_next_task"},
            }
        ],
    }
    payload = {
        "reasoning_mode": "multi_turn",
        "write_resume_checkpoints": True,
        "prepared_bridge_request": {
            "bridge_session": {"session_id": "CASE3"},
            "multi_turn_session_state": session_state,
            "bridge_debug": {"multi_turn_session": session_state},
        },
        "multi_turn_session_result": session_state,
        "multi_turn_current_turn": {
            "turn_index": 2,
            "phase": "outline",
            "prompt_text": "resume me",
            "raw_response": {"decision": "need_next_task"},
        },
    }

    artifact_paths = write_bridge_artifacts(
        payload,
        phase_label="multi_turn",
        debug_dir=tmp_path,
        write_latest=False,
    )

    checkpoint_path = Path(artifact_paths["resume_checkpoint_artifact_path"])
    latest_checkpoint_path = Path(
        artifact_paths["latest_resume_checkpoint_artifact_path"]
    )

    assert checkpoint_path.exists()
    assert latest_checkpoint_path.exists()

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["kind"] == "multi_turn_resume_checkpoint"
    assert checkpoint_payload["session_state"]["turn_index"] == 2
    assert checkpoint_payload["current_turn"]["phase"] == "outline"


def test_write_bridge_artifacts_writes_primitive_batch_resume_checkpoint(
    tmp_path: Path,
) -> None:
    payload = {
        "reasoning_mode": "multi_turn",
        "write_resume_checkpoints": True,
        "prepared_bridge_request": {
            "bridge_session": {"session_id": "CASE3"},
            "bridge_debug": {
                "multi_turn_session": {
                    "session_id": "CASE3",
                    "turn_index": 4,
                    "current_phase": "primitive_generation",
                    "turns": [],
                }
            },
        },
        "multi_turn_current_turn": {
            "turn_index": 4,
            "phase": "primitive_generation",
            "primitive_substream_turns": [
                {
                    "outline_id": "RECOVERY_SEQ4",
                    "resource_jid": "ur5e@localhost",
                    "primitive_local_turn_index": 2,
                    "decision": "need_context",
                    "prompt_text": "primitive turn",
                    "raw_response": {"decision": "need_context"},
                }
            ],
        },
        "primitive_batch_resume_checkpoint": {
            "prepared_bridge_request": {
                "bridge_session": {"session_id": "CASE3"},
            },
            "assigned_outline_events": [
                {
                    "outline_id": "RECOVERY_SEQ4",
                    "resource_jid": "ur5e@localhost",
                }
            ],
            "session_state": {
                "turn_index": 2,
                "final_output_turn_base": 4,
                "turns": [],
            },
            "bridge_session_id": "CASE3",
            "resource_jid": "ur5e@localhost",
            "current_turn": {
                "outline_id": "RECOVERY_SEQ4",
                "primitive_local_turn_index": 2,
            },
            "final_output_turn_base": 4,
        },
    }

    artifact_paths = write_bridge_artifacts(
        payload,
        phase_label="multi_turn",
        debug_dir=tmp_path,
        write_latest=False,
    )

    checkpoint_path = Path(artifact_paths["primitive_resume_checkpoint_artifact_path"])
    latest_checkpoint_path = Path(
        artifact_paths["latest_primitive_resume_checkpoint_artifact_path"]
    )

    assert checkpoint_path.exists()
    assert latest_checkpoint_path.exists()

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["kind"] == "primitive_batch_resume_checkpoint"
    assert checkpoint_payload["resource_jid"] == "ur5e@localhost"
    assert checkpoint_payload["session_state"]["turn_index"] == 2


def test_write_bridge_artifacts_skips_resume_checkpoints_by_default(
    tmp_path: Path,
) -> None:
    session_state = {
        "session_id": "CASE3",
        "turn_index": 1,
        "current_phase": "outline",
        "turns": [],
    }
    payload = {
        "reasoning_mode": "multi_turn",
        "prepared_bridge_request": {
            "bridge_session": {"session_id": "CASE3"},
            "multi_turn_session_state": session_state,
            "bridge_debug": {"multi_turn_session": session_state},
        },
        "multi_turn_session_result": session_state,
        "multi_turn_current_turn": {
            "turn_index": 1,
            "phase": "outline",
            "prompt_text": "resume me",
            "raw_response": {"decision": "need_next_task"},
        },
    }

    artifact_paths = write_bridge_artifacts(
        payload,
        phase_label="multi_turn",
        debug_dir=tmp_path,
        write_latest=False,
    )

    assert "resume_checkpoint_artifact_path" not in artifact_paths
    assert "latest_resume_checkpoint_artifact_path" not in artifact_paths


def test_write_bridge_artifacts_can_skip_phase_prompt_response(
    tmp_path: Path,
) -> None:
    session_state = {
        "session_id": "CASE3",
        "turn_index": 9,
        "current_phase": "finalize",
        "status": "paused_after_primitive_generation",
        "turns": [
            {
                "turn_index": 9,
                "phase": "final_output",
                "prompt_text": "",
                "report_text": "",
                "raw_response": {"decision": "final_output_ready"},
            }
        ],
    }
    payload = {
        "reasoning_mode": "multi_turn",
        "prepared_bridge_request": {
            "bridge_session": {"session_id": "CASE3"},
            "bridge_debug": {"multi_turn_session": session_state},
        },
        "bridge_debug": {"multi_turn_session": session_state},
        "multi_turn_current_turn": {
            "turn_index": 9,
            "phase": "final_output",
            "raw_response": {"decision": "final_output_ready"},
        },
    }

    artifact_paths = write_bridge_artifacts(
        payload,
        phase_label="multi_turn",
        debug_dir=tmp_path,
        write_latest=False,
        write_session_transcript=True,
        write_phase_prompt_response=False,
    )

    assert "prompt_artifact_path" not in artifact_paths
    assert "response_artifact_path" not in artifact_paths
    assert "session_transcript_artifact_path" in artifact_paths


def test_resolve_resume_checkpoint_path_accepts_prompt_artifact(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / (
        "multi_turn_RECOVERY_SEQ4_primitive_generation_"
        "turn02_prompt_20260423T003408.txt"
    )
    checkpoint_path = prompt_path.with_name(
        "multi_turn_RECOVERY_SEQ4_primitive_generation_"
        "turn02_resume_checkpoint_20260423T003408.json"
    )
    prompt_path.write_text("prompt", encoding="utf-8")
    checkpoint_path.write_text('{"kind":"primitive_batch_resume_checkpoint"}', encoding="utf-8")
    try:
        assert case3_dryrun._resolve_resume_checkpoint_path(prompt_path) == checkpoint_path
    finally:
        prompt_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)


def test_run_case3_bridge_dryrun_resumes_primitive_batch_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "primitive_resume_checkpoint.json"
    checkpoint_payload = {
        "kind": "primitive_batch_resume_checkpoint",
        "prepared_bridge_request": {
            "bridge_session": {"session_id": "CASE3"},
            "bridge_debug": {},
            "llm_input": {},
            "context_summary": {},
        },
        "assigned_outline_events": [
            {
                "outline_id": "RECOVERY_SEQ4",
                "resource_jid": "ur5e@localhost",
            }
        ],
        "session_state": {
            "turn_index": 2,
            "turns": [],
        },
        "bridge_session_id": "CASE3",
        "resource_jid": "ur5e@localhost",
    }
    checkpoint_path.write_text(
        json.dumps(checkpoint_payload, ensure_ascii=True),
        encoding="utf-8",
    )

    class _FakeProductAgent:
        turn_log: list[dict[str, Any]] = []

    class _FakeResourceAgent:
        jid = "ur5e@localhost"

    class _FakePlanner:
        def __init__(self) -> None:
            self.resource_agents = [_FakeResourceAgent()]

    observed_call: dict[str, Any] = {}

    async def _fake_prepare_bridge_dryrun_harness(**_kwargs: Any) -> tuple[Any, Any, Any, Any]:
        return {}, _FakeProductAgent(), _FakePlanner(), {}

    async def _fake_generate_primitive_batch_with_llm_agent(**kwargs: Any) -> dict[str, Any]:
        observed_call.update(kwargs)
        return {
            "decision": "need_context",
            "bridge_proposal": {},
            "session_state": {
                "turn_index": 3,
                "turns": [
                    {
                        "prompt_artifact_path": "/tmp/prompt.txt",
                        "response_artifact_path": "/tmp/response.txt",
                        "primitive_resume_checkpoint_artifact_path": "/tmp/primitive.json",
                        "latest_primitive_resume_checkpoint_artifact_path": "/tmp/primitive_latest.json",
                    }
                ],
            },
        }

    with patch.object(
        case3_dryrun,
        "_prepare_bridge_dryrun_harness",
        _fake_prepare_bridge_dryrun_harness,
    ), patch.object(
        case3_dryrun,
        "generate_primitive_batch_with_llm_agent",
        _fake_generate_primitive_batch_with_llm_agent,
    ):
        result = asyncio.run(
            case3_dryrun.run_case3_bridge_dryrun(
                write_debug=False,
                resume_checkpoint=checkpoint_path,
            )
        )

    assert observed_call["bridge_session_id"] == "CASE3"
    assert observed_call["session_state"]["turn_index"] == 2
    assert result["resume_checkpoint_source_path"] == str(checkpoint_path)
    assert result["primitive_resume_checkpoint_artifact_path"] == "/tmp/primitive.json"


def test_final_output_payload_includes_predecessors_in_executable_recovery_trace() -> None:
    session_state = {
        "status": "paused_after_primitive_generation",
        "current_phase": "finalize",
        "accepted_outline_prefix": [
            {
                "outline_id": "RECOVERY_SEQ1",
                "event_name": "recover_to_home_idle",
                "resource_jid": "xarm6@localhost",
                "predecessors": [],
            },
            {
                "outline_id": "RECOVERY_SEQ2",
                "event_name": "stage_mcp_to_prusa_mk3",
                "resource_jid": "ur5e@localhost",
                "part_name": "MCP",
                "target_ref": "prusa-mk3",
                "predecessors": ["RECOVERY_SEQ1"],
            },
        ],
        "accepted_primitive_program": [
            {
                "outline_id": "RECOVERY_SEQ1",
                "des_event_id": "RECOVERY_SEQ1",
                "resource_jid": "xarm6@localhost",
                "predecessors": [],
                "primitive_steps": [{"primitive": "move_to_named_pose", "params": {}}],
            },
            {
                "outline_id": "RECOVERY_SEQ2",
                "des_event_id": "RECOVERY_SEQ2",
                "resource_jid": "ur5e@localhost",
                "part_name": "MCP",
                "predecessors": ["RECOVERY_SEQ1"],
                "primitive_steps": [{"primitive": "release_part", "params": {}}],
            },
        ],
    }

    payload = _build_final_output_payload(
        session_state,
        stage="primitive_program_ready",
        prepared_bridge_request={},
    )

    executable_trace = payload["executable_recovery_trace"]
    assert executable_trace[0]["predecessors"] == []
    assert executable_trace[1]["predecessors"] == ["RECOVERY_SEQ1"]
