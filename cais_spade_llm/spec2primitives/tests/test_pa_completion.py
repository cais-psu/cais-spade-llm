"""Tests for the native PA grounding-completion contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingContractError,
    PAContextGroundingCompletionV3,
    build_product_context_view,
    load_pa_context_grounding_completion,
    persist_pa_context_grounding_completion_v3,
    persist_product_context_view,
)
from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import (
    OntologyGroundingCandidate,
    OntologyGroundingResult,
    commit_ontology_grounding_candidate,
    propose_and_validate_ontology_grounding,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    commit_resource_assignment,
    derive_resource_assignment_need,
    select_predefined_resource,
)
from cais_spade_llm.spec2primitives.ontology import (
    load_predefined_resource_registry,
    load_predefined_workcell,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    PPR_NAMESPACE,
    ontology_config,
)

_PROPOSAL_CONTEXT_SUMMARY = (
    "The RobotFrameLocationRecord is unavailable in the proposal-time context."
)
_SCOPED_CONTEXT_SUMMARY = (
    "ProductAgent proposal summary (captured before deterministic typed grounding "
    "and resource assignment):\n"
    f"{_PROPOSAL_CONTEXT_SUMMARY}\n\n"
    "Final typed context records and ResourceSelectionRecord are authoritative for "
    "completion state."
)


class _ProposalAgent:
    def __init__(
        self,
        evidence_refs: tuple[str, ...] = ("requirement_0001",),
    ) -> None:
        self.evidence_refs = evidence_refs

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object]]
        ]
        | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, tools, tool_executor, max_tool_rounds
        assert response_format["name"] == "spec2primitives_grounding_result"
        proposal = {
            "individuals": [
                {
                    "individual_index": 1,
                    "class_iri": f"{PPR_NAMESPACE}feature",
                    "grounded_meaning": "The requested assembly feature.",
                    "evidence_refs": list(self.evidence_refs),
                }
            ],
            "relations": [
                {
                    "subject_kind": "specification",
                    "subject_individual_index": None,
                    "subject_iri": None,
                    "predicate_iri": f"{PPR_NAMESPACE}defines",
                    "object_kind": "new_individual",
                    "object_individual_index": 1,
                    "object_iri": None,
                    "evidence_refs": list(self.evidence_refs),
                },
                {
                    "subject_kind": "existing_individual",
                    "subject_individual_index": None,
                    "subject_iri": "https://cais-spade-llm.local/process/assembly",
                    "predicate_iri": f"{PPR_NAMESPACE}realizes",
                    "object_kind": "new_individual",
                    "object_individual_index": 1,
                    "object_iri": None,
                    "evidence_refs": list(self.evidence_refs),
                },
            ],
            "literal_facts": [],
            "context_summary": _PROPOSAL_CONTEXT_SUMMARY,
            "evidence_refs": list(self.evidence_refs),
            "missing_information": [
                "The available evidence does not identify the destination shaft."
            ],
        }
        return {"result": proposal}


def test_completion_v3_pins_location_grounding_without_session(tmp_path: Path) -> None:
    completion = persist_native_completion_fixture(tmp_path)

    assert isinstance(completion, PAContextGroundingCompletionV3)
    record = completion.to_record()
    assert record["schema_version"] == 3
    assert record["status"] == "grounding complete"
    assert "grounding_session_ref" not in record
    assert len(record["typed_context_refs"]) == 1
    contract = _read_json(tmp_path / str(record["typed_grounding_contract_ref"]))
    assert contract["schema_version"] == 3
    assert contract["context_summary"] == _SCOPED_CONTEXT_SUMMARY
    assert contract["context_evidence_refs"] == ["requirement_0001"]
    assert contract["missing_information"] == [
        "The available evidence does not identify the destination shaft."
    ]
    assert contract["tool_call_refs"] == []
    location_ref = record["typed_context_refs"][0]["ref"]
    assert _read_json(tmp_path / location_ref)["record_type"] == (
        "RobotFrameLocationRecord"
    )
    proposal = _read_json(tmp_path / str(record["ontology_projection_ref"]))
    assert proposal["output"]["context_summary"] == _PROPOSAL_CONTEXT_SUMMARY
    serialized = json.dumps(record)
    for forbidden in (
        "GroundingSession",
        "TaskTransitionContract",
        "primitive_steps",
        "RobotFramePoseRecord",
    ):
        assert forbidden not in serialized


def test_completion_v3_loader_accepts_unmarked_existing_summary(
    tmp_path: Path,
) -> None:
    completion = persist_native_completion_fixture(tmp_path).to_record()
    contract_path = tmp_path / str(completion["typed_grounding_contract_ref"])
    contract = _read_json(contract_path)
    contract["context_summary"] = _PROPOSAL_CONTEXT_SUMMARY
    contract["fingerprint"] = _fingerprint_without_fingerprint(contract)
    _write_json(contract_path, contract)

    completion_path = tmp_path / "interaction_record/context_completion_0001.json"
    existing_completion = _read_json(completion_path)
    existing_completion["typed_grounding_contract_sha256"] = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    existing_completion["typed_grounding_contract_fingerprint"] = contract[
        "fingerprint"
    ]
    existing_completion["fingerprint"] = _fingerprint_without_fingerprint(
        existing_completion
    )
    _write_json(completion_path, existing_completion)

    loaded = load_pa_context_grounding_completion(tmp_path)

    assert isinstance(loaded, PAContextGroundingCompletionV3)
    assert _read_json(contract_path)["context_summary"] == _PROPOSAL_CONTEXT_SUMMARY


def test_completion_v3_pins_answered_clarification_source(tmp_path: Path) -> None:
    clarification_ref = "interaction_record/clarification_0001.json"
    clarification_path = tmp_path / clarification_ref
    _write_json(
        clarification_path,
        {
            "schema_version": 1,
            "record_type": "PAClarification",
            "product_requirement": "assemble medium gear",
            "question_turn": 1,
            "question": "Which product variant is intended?",
            "action": "answered",
            "reply": "Medium Gear",
            "recorded_at_ns": 1,
            "fingerprint": "0" * 64,
        },
    )
    completion = persist_native_completion_fixture(
        tmp_path,
        evidence_refs=("requirement_0001", clarification_ref),
    ).to_record()

    pinned = next(
        item for item in completion["source_refs"] if item["ref"] == clarification_ref
    )
    assert pinned["sha256"] == hashlib.sha256(clarification_path.read_bytes()).hexdigest()

    clarification = _read_json(clarification_path)
    clarification["reply"] = "Small Gear"
    _write_json(clarification_path, clarification)
    with pytest.raises(GroundingContractError, match="source hash is invalid"):
        load_pa_context_grounding_completion(tmp_path)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("selection", "resource_selection_ref hash"),
        ("assignment", "resource_assignment_delta_ref hash"),
        ("location", "pinned record changed"),
        ("proposal", "ontology_projection_ref hash"),
    ],
)
def test_completion_loader_rejects_changed_native_inputs(
    tmp_path: Path,
    target: str,
    message: str,
) -> None:
    completion = persist_native_completion_fixture(tmp_path).to_record()
    ref_field = {
        "selection": "resource_selection_ref",
        "assignment": "resource_assignment_delta_ref",
        "proposal": "ontology_projection_ref",
    }.get(target)
    if ref_field is not None:
        path = tmp_path / str(completion[ref_field])
    else:
        path = tmp_path / str(completion["typed_context_refs"][0]["ref"])
    value = _read_json(path)
    value["tampered"] = True
    _write_json(path, value)

    with pytest.raises(GroundingContractError, match=message):
        load_pa_context_grounding_completion(tmp_path)


def test_completion_loader_rejects_changed_decision_and_contract(
    tmp_path: Path,
) -> None:
    decision_root = tmp_path / "decision"
    decision_completion = persist_native_completion_fixture(decision_root).to_record()
    decision_path = decision_root / str(decision_completion["decision_ref"])
    decision = _read_json(decision_path)
    decision["PA_output"]["grounding_status"] = "incomplete"
    _write_json(decision_path, decision)
    with pytest.raises(GroundingContractError, match="decision reference"):
        load_pa_context_grounding_completion(decision_root)

    contract_root = tmp_path / "contract"
    contract_completion = persist_native_completion_fixture(contract_root).to_record()
    contract_path = contract_root / str(
        contract_completion["typed_grounding_contract_ref"]
    )
    contract = _read_json(contract_path)
    contract["missing_information"] = []
    _write_json(contract_path, contract)
    with pytest.raises(GroundingContractError, match="typed_grounding_contract_ref hash"):
        load_pa_context_grounding_completion(contract_root)


def test_unsupported_completion_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "interaction_record/context_completion_0001.json"
    _write_json(path, {"schema_version": 1})

    with pytest.raises(GroundingContractError, match="version is unsupported"):
        load_pa_context_grounding_completion(tmp_path)


def persist_native_completion_fixture(
    root: Path,
    *,
    evidence_refs: tuple[str, ...] = ("requirement_0001",),
) -> PAContextGroundingCompletionV3:
    """Create one complete native record chain for completion and UI tests."""
    requirement = "assemble medium gear"
    _write_json(
        root / "products/user_requirement/product_requirement.json",
        {"product_requirement": requirement},
    )
    tbox = ontology_config().load_tbox()
    initial_abox = initialize_interaction_abox(root, requirement, tbox)
    registry = load_predefined_resource_registry(tbox)
    workcell = load_predefined_workcell(tbox, registry)
    candidate = asyncio.run(
        propose_and_validate_ontology_grounding(
            _ProposalAgent(evidence_refs),
            interaction_root=root,
            tbox=tbox,
            abox=initial_abox,
            workcell=workcell,
            evidence_catalog=[],
            authorized_evidence_refs=set(evidence_refs),
            tools=[],
            tool_executor=_unused_tool,
            max_tool_rounds=1,
            required_output_projection={
                "record_type": "RobotFrameLocationRecord",
                "target_frame": "world",
            },
        )
    )
    assert isinstance(candidate, OntologyGroundingCandidate)
    proposal = commit_ontology_grounding_candidate(
        candidate,
        interaction_root=root,
        tbox=tbox,
        abox=initial_abox,
        workcell=workcell,
        authorized_evidence_refs=set(evidence_refs),
    )
    assert isinstance(proposal, OntologyGroundingResult)

    location_path, source_ref = _write_location(root)
    location_ref = location_path.relative_to(root).as_posix()
    location_merge = validate_and_merge_triple_delta(
        root,
        tbox,
        "synthetic_world_location_provider",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [location_ref],
        },
        authorized_evidence_refs={source_ref},
    )
    need = derive_resource_assignment_need(location_merge.abox, workcell)
    assert need is not None
    selection = select_predefined_resource(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        grounding_record_path=location_path,
    )
    assignment = commit_resource_assignment(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        selection=selection,
    )
    final_view = build_product_context_view(
        root,
        assignment.abox,
        attempted_evidence=(),
        assessed_at_ns=time.time_ns(),
    )
    persist_product_context_view(root, final_view)
    proposal_ref = proposal.proposal_path.relative_to(root).as_posix()
    turn_path = root / "interaction_record/turn_0001.json"
    _write_json(
        turn_path,
        {
            "turn": 1,
            "product_requirement": requirement,
            "PA_input": {"mode": "native_tool_grounding"},
            "PA_output": {
                "grounding_status": "complete",
                "ontology_projection_ref": proposal_ref,
                "resource_selection_ref": selection.record_ref,
                "tool_call_refs": [],
            },
            "failure": None,
        },
    )
    persist_pa_context_grounding_completion_v3(
        root,
        product_requirement=requirement,
        completion_turn=1,
        decision_ref=turn_path.relative_to(root).as_posix(),
        product_context=final_view,
        ontology_projection_ref=proposal_ref,
        resource_selection_ref=selection.record_ref,
        tool_call_refs=(),
    )
    loaded = load_pa_context_grounding_completion(root)
    assert isinstance(loaded, PAContextGroundingCompletionV3)
    return loaded


async def _unused_tool(
    tool_name: str,
    arguments: Mapping[str, object],
) -> Mapping[str, object]:
    del tool_name, arguments
    raise AssertionError("No tool call was expected.")


def _write_location(root: Path) -> tuple[Path, str]:
    destination = root / "products/grounding/synthetic_world_location"
    destination.mkdir(parents=True, exist_ok=True)
    source_path = destination / "source_evidence.json"
    _write_json(source_path, {"source": "synthetic observed center"})
    source_ref = source_path.relative_to(root).as_posix()
    path = destination / "robot_frame_location_record.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "record_type": "RobotFrameLocationRecord",
            "producer": "synthetic_world_location_provider",
            "robot_frame_conversion": "accepted",
            "CAD_correspondence": "accepted",
            "location": "available",
            "source_frame": "camera_optical_frame",
            "target_frame": "world",
            "observation_timestamp_ns": 11,
            "translated_location_m": [0.0, 0.0, 1.1],
            "source_hashes": [
                {
                    "ref": source_ref,
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    return path, source_ref


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint_without_fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
