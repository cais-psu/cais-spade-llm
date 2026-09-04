"""Tests for native ProductAgent clarification and cancellation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from cais_spade_llm.spec2primitives.agents.pa.context_assessment import (
    cancel_pa_context_interaction,
    continue_pa_context_interaction,
    submit_pa_clarification_reply,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    start_pa_context_interaction,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    ontology_config,
)
from cais_spade_llm.spec2primitives.tests.test_pa_context_interaction import (
    _GroundingRuntime,
    _UnusedProductAgent,
)


def test_exact_reply_resumes_same_interaction_with_native_history(
    tmp_path: Path,
) -> None:
    runtime = _GroundingRuntime(
        [
            {
                "grounding_status": "clarification_required",
                "clarification_question": "Which gear size should be assembled?",
                "tool_call_refs": [],
            },
            {
                "grounding_status": "incomplete",
                "grounding_stage": "target_feature",
                "insufficient_evidence": "The target feature cites invalid evidence.",
                "grounding_validation_code": "evidence_reference_invalid",
                "tool_call_refs": [],
            },
        ]
    )
    agent = _UnusedProductAgent()
    _start(tmp_path, agent, runtime)

    result = asyncio.run(
        submit_pa_clarification_reply(
            agent,
            tmp_path,
            "  Medium Gear  ",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )

    assert result["grounding_status"] == "incomplete"
    assert result["grounding_validation_code"] == "evidence_reference_invalid"
    assert "unmet_grounding_obligation" not in result
    record = _read_json(tmp_path / "interaction_record/clarification_0001.json")
    assert record["record_type"] == "PAClarification"
    assert record["question_turn"] == 1
    assert record["reply"] == "  Medium Gear  "
    payload = {key: value for key, value in record.items() if key != "fingerprint"}
    assert record["fingerprint"] == _fingerprint(payload)
    assert runtime.calls[-1]["clarification_history"] == (record,)
    turn = _read_json(tmp_path / "interaction_record/turn_0002.json")
    assert turn["PA_output"]["grounding_validation_code"] == (
        "evidence_reference_invalid"
    )
    assert turn["PA_input"]["mode"] == "native_tool_grounding"
    assert turn["PA_input"]["clarification_refs"] == [
        "interaction_record/clarification_0001.json"
    ]
    assert not (tmp_path / "products/grounding/session").exists()


def test_reply_cannot_complete_without_resource_assignment(
    tmp_path: Path,
) -> None:
    runtime = _GroundingRuntime(
        [
            {
                "grounding_status": "clarification_required",
                "clarification_question": "Which gear size should be assembled?",
                "tool_call_refs": [],
            },
            {
                "grounding_status": "complete",
                "resource_assignment_status": "deferred",
                "ontology_projection_ref": "proposal.json",
                "tool_call_refs": [],
            },
        ]
    )
    agent = _UnusedProductAgent()
    _start(tmp_path, agent, runtime)

    result = asyncio.run(
        submit_pa_clarification_reply(
            agent,
            tmp_path,
            "Medium Gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )

    assert result["failure"]["reason"] == "pa_call_failed"
    assert "resource-assignment status is invalid" in result["failure"]["message"]
    assert not (tmp_path / "products/grounding/session").exists()


def test_repeated_clarifications_use_append_only_turns(tmp_path: Path) -> None:
    runtime = _GroundingRuntime(
        [
            {
                "grounding_status": "clarification_required",
                "clarification_question": "Which gear size?",
                "tool_call_refs": [],
            },
            {
                "grounding_status": "clarification_required",
                "clarification_question": "Which assembly outcome?",
                "tool_call_refs": [],
            },
            {
                "grounding_status": "incomplete",
                "grounding_stage": "resource_assignment",
                "insufficient_evidence": "No valid state-location evidence is available.",
                "grounding_validation_code": "location_evidence_unavailable",
                "tool_call_refs": [],
            },
        ]
    )
    agent = _UnusedProductAgent()
    _start(tmp_path, agent, runtime)

    first = asyncio.run(
        submit_pa_clarification_reply(
            agent,
            tmp_path,
            "Medium Gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )
    second = asyncio.run(
        submit_pa_clarification_reply(
            agent,
            tmp_path,
            "assembled product",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )

    assert first["grounding_status"] == "clarification_required"
    assert second["grounding_status"] == "incomplete"
    assert (tmp_path / "interaction_record/clarification_0001.json").is_file()
    assert (tmp_path / "interaction_record/clarification_0002.json").is_file()
    assert (tmp_path / "interaction_record/turn_0003.json").is_file()
    assert len(runtime.calls[-1]["clarification_history"]) == 2
    assert _read_json(tmp_path / "interaction_record/turn_0003.json")["PA_input"][
        "clarification_refs"
    ] == [
        "interaction_record/clarification_0001.json",
        "interaction_record/clarification_0002.json",
    ]


def test_cancellation_is_terminal_without_product_agent_call(tmp_path: Path) -> None:
    runtime = _GroundingRuntime(
        [
            {
                "grounding_status": "clarification_required",
                "clarification_question": "Which gear size?",
                "tool_call_refs": [],
            }
        ]
    )
    agent = _UnusedProductAgent()
    _start(tmp_path, agent, runtime)
    call_count = len(runtime.calls)

    result = cancel_pa_context_interaction(tmp_path)

    assert result == {"status": "cancelled", "grounding_status": "incomplete"}
    assert len(runtime.calls) == call_count
    record = _read_json(tmp_path / "interaction_record/clarification_0001.json")
    assert record["action"] == "cancelled"
    assert record["reply"] is None
    assert not (tmp_path / "interaction_record/turn_0002.json").exists()


def test_blank_reply_and_unanswered_continue_are_rejected(tmp_path: Path) -> None:
    runtime = _GroundingRuntime(
        [
            {
                "grounding_status": "clarification_required",
                "clarification_question": "Which gear size?",
                "tool_call_refs": [],
            }
        ]
    )
    agent = _UnusedProductAgent()
    _start(tmp_path, agent, runtime)

    blank = asyncio.run(
        submit_pa_clarification_reply(
            agent,
            tmp_path,
            "   ",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )
    unanswered = asyncio.run(
        continue_pa_context_interaction(
            agent,
            tmp_path,
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )

    assert blank["failure"]["reason"] == "invalid_clarification"
    assert unanswered["failure"]["reason"] == "invalid_clarification"
    assert len(runtime.calls) == 1


def test_different_reply_cannot_replace_persisted_answer(tmp_path: Path) -> None:
    runtime = _GroundingRuntime(
        [
            {
                "grounding_status": "clarification_required",
                "clarification_question": "Which gear size?",
                "tool_call_refs": [],
            },
            RuntimeError("controlled interruption"),
        ]
    )
    agent = _UnusedProductAgent()
    _start(tmp_path, agent, runtime)
    first = asyncio.run(
        submit_pa_clarification_reply(
            agent,
            tmp_path,
            "Medium Gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )

    replacement = asyncio.run(
        submit_pa_clarification_reply(
            agent,
            tmp_path,
            "Small Gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )

    assert first["failure"]["reason"] == "pa_call_failed"
    assert replacement["failure"]["reason"] == "invalid_clarification"
    record = _read_json(tmp_path / "interaction_record/clarification_0001.json")
    assert record["reply"] == "Medium Gear"


def test_resume_rejects_unknown_grounding_validation_code(tmp_path: Path) -> None:
    runtime = _GroundingRuntime(
        [
            {
                "grounding_status": "clarification_required",
                "clarification_question": "Which gear size?",
                "tool_call_refs": [],
            },
            {
                "grounding_status": "incomplete",
                "insufficient_evidence": "Invalid controller result.",
                "grounding_validation_code": "unknown_code",
                "tool_call_refs": [],
            },
        ]
    )
    agent = _UnusedProductAgent()
    _start(tmp_path, agent, runtime)

    result = asyncio.run(
        submit_pa_clarification_reply(
            agent,
            tmp_path,
            "Medium Gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )

    assert result["failure"]["reason"] == "pa_call_failed"
    turn = _read_json(tmp_path / "interaction_record/turn_0002.json")
    assert turn["PA_output"] is None


def _start(
    root: Path,
    agent: _UnusedProductAgent,
    runtime: _GroundingRuntime,
) -> None:
    result = asyncio.run(
        start_pa_context_interaction(
            agent,
            root,
            "assemble gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )
    assert result["grounding_status"] == "clarification_required"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
