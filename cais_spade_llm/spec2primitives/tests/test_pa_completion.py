"""Tests for the formal Phase 3.5 PA grounding-completion boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from cais_spade_llm.spec2primitives.agents.pa import (
    PAContextGroundingCompletionV2,
    load_pa_context_grounding_completion,
    submit_pa_clarification_reply,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingDecision,
    GroundingContractError,
    GroundingSession,
    InformationNeed,
    load_latest_grounding_session,
    persist_grounding_session,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    ControlledGroundingRuntime,
    complete_context,
    ontology_config,
    request_clarification,
)
from cais_spade_llm.spec2primitives.tests.test_pa_clarification import (
    _continue,
    _initialize,
    _read_json,
)


def test_completion_record_pins_generalized_grounding_bundle(
    tmp_path: Path,
) -> None:
    grounding = ControlledGroundingRuntime(assessments=[complete_context()])
    product_agent = _initialize(
        tmp_path,
        grounding,
        product_requirement="assemble Medium Gear",
    )

    result = _continue(product_agent, tmp_path, grounding)

    assert result["context understanding complete"] is True
    completion = load_pa_context_grounding_completion(tmp_path)
    assert isinstance(completion, PAContextGroundingCompletionV2)
    assert completion.product_requirement == "assemble Medium Gear"
    assert completion.completion_turn == 2
    assert completion.source_refs[0]["ref"] == "NIST_assembly_instructions.pdf"
    assert completion.typed_context_refs == ()
    assert completion.clarification_refs == ()
    record = completion.to_record()
    assert record["status"] == "context understanding complete"
    assert record["schema_version"] == 2
    assert record["grounding_session_ref"].endswith("revision_0002.json")
    assert record["typed_grounding_contract_ref"].endswith(
        "typed_grounding_contract_0001.json"
    )
    serialized = json.dumps(record)
    for forbidden in (
        "TaskTransitionContract",
        "primitive_steps",
        "RobotAgent",
        "robot_frame",
    ):
        assert forbidden not in serialized


def test_completion_loader_rejects_removed_version_1_artifact(
    tmp_path: Path,
) -> None:
    completion_path = tmp_path / "interaction_record/context_completion_0001.json"
    completion_path.parent.mkdir(parents=True)
    _write_json(completion_path, {"schema_version": 1})

    with pytest.raises(
        GroundingContractError,
        match="schema version 2 is supported",
    ):
        load_pa_context_grounding_completion(tmp_path)


def test_completion_preserves_answered_clarification_ref(tmp_path: Path) -> None:
    grounding = ControlledGroundingRuntime(
        assessments=[request_clarification("Which gear size?"), complete_context()]
    )
    product_agent = _initialize(
        tmp_path,
        grounding,
        product_requirement="assemble gear",
    )
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
    completion = load_pa_context_grounding_completion(tmp_path)
    assert completion.clarification_refs[0]["ref"] == (
        "interaction_record/clarification_0002.json"
    )


@pytest.mark.parametrize("tamper_target", ["completion", "session", "clarification"])
def test_completion_loader_rejects_tampered_records(
    tmp_path: Path,
    tamper_target: str,
) -> None:
    grounding = ControlledGroundingRuntime(
        assessments=[request_clarification("Which gear size?"), complete_context()]
    )
    product_agent = _initialize(
        tmp_path,
        grounding,
        product_requirement="assemble gear",
    )
    _continue(product_agent, tmp_path, grounding)
    asyncio.run(
        submit_pa_clarification_reply(
            product_agent,
            tmp_path,
            "Medium Gear",
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )
    completion_path = tmp_path / "interaction_record/context_completion_0001.json"
    completion = _read_json(completion_path)
    if tamper_target == "completion":
        completion["completion_turn"] = 99
        _write_json(completion_path, completion)
    elif tamper_target == "session":
        session_path = tmp_path / str(completion["grounding_session_ref"])
        session = _read_json(session_path)
        session["information_status"] = "not_enough"
        _write_json(session_path, session)
    else:
        clarification_path = tmp_path / str(
            completion["clarification_refs"][0]["ref"]
        )
        clarification = _read_json(clarification_path)
        clarification["reply"] = "Small Gear"
        _write_json(clarification_path, clarification)

    with pytest.raises(GroundingContractError):
        load_pa_context_grounding_completion(tmp_path)


def test_completion_is_rejected_when_latest_session_is_incomplete(
    tmp_path: Path,
) -> None:
    grounding = _UnresolvedCompletionRuntime(assessments=[complete_context()])
    product_agent = _initialize(
        tmp_path,
        grounding,
        product_requirement="assemble gear",
    )

    result = _continue(product_agent, tmp_path, grounding)

    assert result["failure"]["reason"] == "context_completion_failed"
    assert not (tmp_path / "interaction_record/context_completion_0001.json").exists()


class _UnresolvedCompletionRuntime(ControlledGroundingRuntime):
    """Append an incomplete session before the Phase 3.5 validator runs."""

    async def assess_product_context(  # noqa: PLR0913
        self,
        product_agent: object,
        *,
        interaction_root: Path,
        tbox: object,
        abox: object,
        abox_view: Mapping[str, object],
        attempted_evidence: tuple[str, ...],
        clarification_history: tuple[Mapping[str, object], ...] = (),
        turn_number: int,
        max_pa_turns: int,
    ) -> Mapping[str, object]:
        result = await super().assess_product_context(  # type: ignore[arg-type]
            product_agent,
            interaction_root=interaction_root,
            tbox=tbox,
            abox=abox,
            abox_view=abox_view,
            attempted_evidence=attempted_evidence,
            clarification_history=clarification_history,
            turn_number=turn_number,
            max_pa_turns=max_pa_turns,
        )
        previous = load_latest_grounding_session(interaction_root)
        assert previous is not None
        need = InformationNeed.from_mapping(
            {
                "need_id": "controlled_need_0001",
                "question": "Which user intent remains unresolved?",
                "required": True,
                "sources": ["requirement_0001"],
                "accepted_record_types": ["PAClarification"],
                "status": "exhausted",
                "answer_statement_ids": [],
            }
        )
        decision = GroundingDecision.from_mapping(
            {
                "decision_type": "incomplete",
                "need_id": None,
                "provider_id": None,
                "source_ref": None,
                "source_revision": None,
                "query": None,
                "reason": "The controlled interaction remains incomplete.",
            }
        )
        persist_grounding_session(
            interaction_root,
            GroundingSession.create(
                revision=previous.revision + 1,
                requirement_text=previous.requirement_text,
                statements=previous.statements,
                information_needs=[need],
                attempted_actions=previous.attempted_actions,
                evidence_refs=previous.evidence_refs,
                decision=decision,
                status="incomplete",
                information_status="partial",
            ),
        )
        return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
