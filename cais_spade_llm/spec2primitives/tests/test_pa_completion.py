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
    GroundingContractError,
    GroundingNextAction,
    GroundingSession,
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
    assert len(completion.typed_context_refs) == 1
    assert completion.typed_context_refs[0]["ref"].endswith("world_pose_0001.json")
    assert completion.clarification_refs == ()
    record = completion.to_record()
    assert record["status"] == "context understanding complete"
    assert record["schema_version"] == 2
    assert record["grounding_session_ref"].endswith("revision_0002.json")
    assert record["typed_grounding_contract_ref"].endswith(
        "typed_grounding_contract_0001.json"
    )
    assert record["resource_selection_ref"].endswith(
        "selection_0001/resource_selection_record.json"
    )
    assert record["resource_assignment_delta_ref"].endswith("delta_0004.json")
    contract = _read_json(tmp_path / str(record["typed_grounding_contract_ref"]))
    assert contract["schema_version"] == 2
    assert contract["context_summary"] == (
        "The controlled interaction has enough cited product context."
    )
    assert contract["context_evidence_refs"] == [
        "NIST_assembly_instructions.pdf"
    ]
    assert contract["missing_information"] == []
    serialized = json.dumps(record)
    for forbidden in (
        "TaskTransitionContract",
        "primitive_steps",
        "RobotAgent",
        "robot_frame",
    ):
        assert forbidden not in serialized


def test_completion_loader_rejects_changed_resource_selection(tmp_path: Path) -> None:
    grounding = ControlledGroundingRuntime(assessments=[complete_context()])
    product_agent = _initialize(
        tmp_path,
        grounding,
        product_requirement="assemble Medium Gear",
    )
    _continue(product_agent, tmp_path, grounding)
    completion = load_pa_context_grounding_completion(tmp_path)
    selection_path = tmp_path / completion.resource_selection_ref
    selection = _read_json(selection_path)
    selection["selected_resource_jid"] = "changed@localhost"
    _write_json(selection_path, selection)

    with pytest.raises(GroundingContractError, match="ResourceSelectionRecord hash"):
        load_pa_context_grounding_completion(tmp_path)


def test_completion_loader_rejects_changed_assignment_delta_authority(
    tmp_path: Path,
) -> None:
    grounding = ControlledGroundingRuntime(assessments=[complete_context()])
    product_agent = _initialize(
        tmp_path,
        grounding,
        product_requirement="assemble Medium Gear",
    )
    _continue(product_agent, tmp_path, grounding)
    completion = load_pa_context_grounding_completion(tmp_path)
    delta_path = tmp_path / completion.resource_assignment_delta_ref
    delta = _read_json(delta_path)
    delta["producer"] = "ontology_grounding"
    _write_json(delta_path, delta)

    with pytest.raises(GroundingContractError, match="assignment delta hash"):
        load_pa_context_grounding_completion(tmp_path)


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
        session["status"] = "incomplete"
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
        persist_grounding_session(
            interaction_root,
            GroundingSession.create(
                revision=previous.revision + 1,
                requirement_text=previous.requirement_text,
                attempted_actions=previous.attempted_actions,
                next_action=GroundingNextAction.from_mapping(
                    {
                        "action": "incomplete",
                        "reason": "The controlled interaction remains incomplete.",
                    }
                ),
                status="incomplete",
            ),
        )
        return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
