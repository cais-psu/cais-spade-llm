"""Tests for the formal Phase 3.5 PA grounding-completion boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from cais_spade_llm.spec2primitives.agents.pa import (
    PAContextGroundingCompletion,
    load_pa_context_grounding_completion,
    submit_pa_clarification_reply,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingContractError,
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


def test_completion_record_proves_only_current_draft_is_ready(
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
    assert isinstance(completion, PAContextGroundingCompletion)
    assert completion.product_requirement == "assemble Medium Gear"
    assert completion.completion_turn == 2
    assert completion.attempted_evidence == ("NIST_assembly_instructions.pdf",)
    assert completion.typed_context_refs == ()
    assert completion.clarification_refs == ()
    record = completion.to_record()
    assert record["status"] == "context understanding complete"
    assert record["unresolved_context_needs"] == []
    serialized = json.dumps(record)
    for forbidden in (
        "TaskTransitionContract",
        "primitive_steps",
        "RobotAgent",
        "robot_frame",
    ):
        assert forbidden not in serialized


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
    assert completion.clarification_refs == (
        "interaction_record/clarification_0002.json",
    )


@pytest.mark.parametrize("tamper_target", ["completion", "draft", "clarification"])
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
    elif tamper_target == "draft":
        draft_path = tmp_path / str(completion["task_transition_draft_ref"])
        draft = _read_json(draft_path)
        draft["required_outcome"] = "tampered"
        _write_json(draft_path, draft)
    else:
        clarification_path = tmp_path / str(completion["clarification_refs"][0])
        clarification = _read_json(clarification_path)
        clarification["reply"] = "Small Gear"
        _write_json(clarification_path, clarification)

    with pytest.raises(GroundingContractError):
        load_pa_context_grounding_completion(tmp_path)


def test_completion_is_rejected_when_latest_draft_still_has_a_need(
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
    """Tamper its controlled draft before the Phase 3.5 validator runs."""

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
        draft_path = sorted(
            (interaction_root / "products/grounding/task_transition").glob(
                "draft_*.json"
            )
        )[-1]
        draft = _read_json(draft_path)
        draft["required_inputs"] = [
            {
                "kind": "user_intent",
                "symbol": "product_requirement",
                "subject_role": "product_context",
                "authority": "PA",
                "frame": None,
                "maximum_age_ns": None,
                "reason": "user intent remains unresolved",
            }
        ]
        draft["unresolved_user_intent"] = "user intent remains unresolved"
        _write_json(draft_path, draft)
        return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
