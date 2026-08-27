"""Tests for resumable Phase 3.4 ProductAgent clarification."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from cais_spade_llm.spec2primitives.agents.pa import context_assessment
from cais_spade_llm.spec2primitives.agents.pa.context_assessment import (
    cancel_pa_context_interaction,
    continue_pa_context_interaction,
    submit_pa_clarification_reply,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    start_pa_context_interaction,
)
from cais_spade_llm.spec2primitives.agents.pa.context_serving import (
    serve_pa_requested_context,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    ControlledGroundingRuntime,
    complete_context,
    ontology_config,
    request_clarification,
    request_context,
)
from cais_spade_llm.spec2primitives.tests.test_pa_context_assessment import (
    FakeProductAgent,
)


def test_exact_reply_resumes_same_interaction_and_rebuilds_view(
    tmp_path: Path,
) -> None:
    grounding = ControlledGroundingRuntime(
        assessments=[
            request_clarification("Which gear size should be assembled?"),
            complete_context(),
        ]
    )
    product_agent = _initialize(tmp_path, grounding, product_requirement="assemble gear")

    clarification = _continue(product_agent, tmp_path, grounding)
    assert clarification["context understanding complete"] is False

    result = asyncio.run(
        submit_pa_clarification_reply(
            product_agent,
            tmp_path,
            "  Medium Gear  ",
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )

    assert result == {
        "needed_context": None,
        "context understanding complete": True,
    }
    record = _read_json(tmp_path / "interaction_record/clarification_0002.json")
    assert record["reply"] == "  Medium Gear  "
    assert record["action"] == "answered"
    payload = {key: value for key, value in record.items() if key != "fingerprint"}
    assert record["fingerprint"] == _fingerprint(payload)
    assert grounding.assessment_calls[-1]["clarification_history"] == (record,)

    before = _read_json(tmp_path / "products/grounding/product_context/view_0000.json")
    after = _read_json(tmp_path / "products/grounding/product_context/view_0001.json")
    assert before["delta_count"] == after["delta_count"] == 1
    assert before["abox_fingerprint"] == after["abox_fingerprint"]
    assert "Medium Gear" not in json.dumps(after["assertions"])
    assert (tmp_path / "interaction_record/decision_0002.json").is_file()
    assert (tmp_path / "interaction_record/turn_0003.json").is_file()


def test_reply_can_request_evidence_then_complete_without_fixed_order(
    tmp_path: Path,
) -> None:
    grounding = ControlledGroundingRuntime(
        assessments=[
            request_clarification("Which gear size should be assembled?"),
            request_context("Gear_Medium.STL"),
            complete_context(),
        ]
    )
    product_agent = _initialize(tmp_path, grounding, product_requirement="assemble gear")
    _continue(product_agent, tmp_path, grounding)

    result = asyncio.run(
        submit_pa_clarification_reply(
            product_agent,
            tmp_path,
            "Medium Gear",
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )

    assert result["context understanding complete"] is True
    assert not (tmp_path / "interaction_record/retrieval_0002.json").exists()
    retrieval = _read_json(tmp_path / "interaction_record/retrieval_0003.json")
    assert retrieval["served_context"]["context_ref"] == "Gear_Medium.STL"
    assert (tmp_path / "interaction_record/interpretation_0003.json").is_file()


def test_repeated_clarification_and_explicit_cancellation(tmp_path: Path) -> None:
    repeat_root = tmp_path / "repeat"
    grounding = ControlledGroundingRuntime(
        assessments=[
            request_clarification("Which gear size?"),
            request_clarification("Which assembly outcome?"),
            complete_context(),
        ]
    )
    product_agent = _initialize(repeat_root, grounding, product_requirement="assemble gear")
    _continue(product_agent, repeat_root, grounding)
    first = asyncio.run(
        submit_pa_clarification_reply(
            product_agent,
            repeat_root,
            "Medium Gear",
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )
    assert first["needed_context"]["clarification_question"] == "Which assembly outcome?"
    second = asyncio.run(
        submit_pa_clarification_reply(
            product_agent,
            repeat_root,
            "assembled on its receiving feature",
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )
    assert second["context understanding complete"] is True
    assert (repeat_root / "interaction_record/clarification_0002.json").is_file()
    assert (repeat_root / "interaction_record/clarification_0003.json").is_file()

    cancel_root = tmp_path / "cancel"
    cancel_grounding = ControlledGroundingRuntime(
        assessments=[request_clarification("Which gear size?")]
    )
    cancel_agent = _initialize(
        cancel_root,
        cancel_grounding,
        product_requirement="assemble gear",
    )
    _continue(cancel_agent, cancel_root, cancel_grounding)
    calls_before = len(cancel_agent.calls)
    assert cancel_pa_context_interaction(cancel_root) == {
        "status": "cancelled",
        "context understanding complete": False,
    }
    assert len(cancel_agent.calls) == calls_before
    cancellation = _read_json(
        cancel_root / "interaction_record/clarification_0002.json"
    )
    assert cancellation["action"] == "cancelled"
    assert cancellation["reply"] is None
    assert not (cancel_root / "interaction_record/turn_0003.json").exists()


def test_invalid_or_system_clarification_reply_is_rejected(tmp_path: Path) -> None:
    grounding = ControlledGroundingRuntime(
        assessments=[request_clarification("Which gear size?")]
    )
    product_agent = _initialize(tmp_path, grounding, product_requirement="assemble gear")
    _continue(product_agent, tmp_path, grounding)

    blank = asyncio.run(
        submit_pa_clarification_reply(
            product_agent,
            tmp_path,
            "   ",
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )
    assert blank["failure"]["reason"] == "invalid_clarification"
    assert not (tmp_path / "interaction_record/clarification_0002.json").exists()

    decision_path = tmp_path / "interaction_record/decision_0001.json"
    decision = _read_json(decision_path)
    decision["Phase_4_3_output"]["unresolved_semantic_need"]["kind"] = (
        "typed_context_record"
    )
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    system_need = asyncio.run(
        submit_pa_clarification_reply(
            product_agent,
            tmp_path,
            "The gear is on the left",
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )
    assert system_need["failure"]["reason"] == "invalid_clarification"
    assert not (tmp_path / "interaction_record/clarification_0002.json").exists()


def test_interrupted_same_reply_recovers_and_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grounding = ControlledGroundingRuntime(
        assessments=[request_clarification("Which gear size?"), complete_context()]
    )
    product_agent = _initialize(tmp_path, grounding, product_requirement="assemble gear")
    _continue(product_agent, tmp_path, grounding)
    real_continue = context_assessment.continue_pa_context_interaction

    async def interrupted(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("controlled interruption")

    monkeypatch.setattr(context_assessment, "continue_pa_context_interaction", interrupted)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        asyncio.run(
            submit_pa_clarification_reply(
                product_agent,
                tmp_path,
                "Medium Gear",
                ontology_config=ontology_config(),
                grounding_runtime=grounding,
            )
        )

    replacement = asyncio.run(
        submit_pa_clarification_reply(
            product_agent,
            tmp_path,
            "Small Gear",
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )
    assert replacement["failure"]["reason"] == "clarification_exists"

    monkeypatch.setattr(context_assessment, "continue_pa_context_interaction", real_continue)
    recovered = asyncio.run(
        submit_pa_clarification_reply(
            product_agent,
            tmp_path,
            "Medium Gear",
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )
    assert recovered["context understanding complete"] is True


def _initialize(
    interaction_root: Path,
    grounding: ControlledGroundingRuntime,
    *,
    product_requirement: str,
) -> FakeProductAgent:
    product_agent = FakeProductAgent(
        responses=[
            {
                "needed_context": {
                    "context_ref": "NIST_assembly_instructions.pdf",
                    "request_live_observation": False,
                    "clarification_question": None,
                }
            }
        ]
    )
    initial = asyncio.run(
        start_pa_context_interaction(
            product_agent,
            interaction_root,
            product_requirement,
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )
    assert "needed_context" in initial
    assert "served_context" in serve_pa_requested_context(interaction_root)
    return product_agent


def _continue(
    product_agent: FakeProductAgent,
    interaction_root: Path,
    grounding: ControlledGroundingRuntime,
) -> dict[str, object]:
    return asyncio.run(
        continue_pa_context_interaction(
            product_agent,
            interaction_root,
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )


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
