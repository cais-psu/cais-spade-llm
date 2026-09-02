"""Tests for the native PA grounding-completion contract."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingContractError,
    PAContextGroundingCompletionV6,
    build_product_context_view,
    load_pa_context_grounding_completion,
    persist_pa_context_grounding_completion_v6,
    persist_product_context_view,
)
from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import (
    OntologyGroundingCandidate,
    OntologyGroundingResult,
    commit_ontology_grounding_candidate,
    propose_and_validate_ontology_grounding,
    review_target_feature_semantics,
)
from cais_spade_llm.spec2primitives.agents.pa.presentation_records import (
    AllocationEvidenceSource,
    load_or_create_allocation_presentation,
    load_or_create_evidence_presentation,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    candidate_resource_catalog,
    check_resource_reachability,
    commit_resource_assignment,
    persist_pa_resource_selection,
)
from cais_spade_llm.spec2primitives.agents.ra.feasibility_validation import (
    validate_provisional_allocation,
)
from cais_spade_llm.spec2primitives.ontology import (
    load_predefined_resource_registry,
    load_predefined_workcell,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import ontology_config


class _ProposalAgent:
    def __init__(
        self,
        evidence_refs: tuple[str, ...] = ("requirement_0001",),
        current_location_ref: str | None = None,
        desired_location_ref: str | None = None,
        target_state_record_ref: str | None = None,
    ) -> None:
        self.evidence_refs = evidence_refs
        self.current_location_ref = current_location_ref
        self.desired_location_ref = desired_location_ref
        self.target_state_record_ref = target_state_record_ref

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]
        | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, tools, tool_executor, max_tool_rounds
        if response_format["name"] == "spec2primitives_target_feature_review":
            return {"verdict": "complete", "gap": None}
        assert response_format["name"] == "spec2primitives_grounding_result"
        proposal = {
            "target_feature": {
                "required_process": {
                    "process_iri": "https://cais-spade-llm.local/process/assembly",
                    "evidence_refs": list(self.evidence_refs),
                },
                "current_state": {
                    "statement": {
                        "text": "The medium gear is currently separate from the assembly.",
                        "evidence_refs": list(self.evidence_refs),
                    },
                    "state_values": [],
                },
                "desired_state": {
                    "statement": {
                        "text": "The medium gear is assembled as requested.",
                        "evidence_refs": list(self.evidence_refs),
                    },
                    "state_values": (
                        []
                        if self.target_state_record_ref is None
                        else [
                            {
                                "name": "specified_finish",
                                "value_ref": {
                                    "record_ref": self.target_state_record_ref,
                                    "field_path": "/overview/summary",
                                },
                                "evidence_refs": [self.target_state_record_ref],
                            },
                            {
                                "name": "specified_coating",
                                "value_ref": {
                                    "record_ref": self.target_state_record_ref,
                                    "field_path": "/overview/observations/0",
                                },
                                "evidence_refs": [self.target_state_record_ref],
                            },
                        ]
                    ),
                },
            }
        }
        return {"result": proposal}


class _AcceptedFeasibilityRuntime:
    async def validate_plan_only_allocation(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        assert request["resource_symbol"] == "xarm6"
        assert request["mode"] == "plan_only"
        assert request["motion_executed"] is False
        assert request["moveit_group"] == "xarm6_xarm6"
        assert request["end_effector_link"] == "xarm6_link_tcp"
        assert request["current_state"]["state_iri"].endswith("currentstate_0001")
        assert request["desired_state"]["state_iri"].endswith("desiredstate_0001")
        assert (
            request["current_state"]["location_record_ref"]
            != (request["desired_state"]["location_record_ref"])
        )
        endpoint = {
            "status": "accepted",
            "message": "fixture plan-only path accepted",
            "error_code": 1,
        }
        return {
            "status": "accepted",
            "current_state": dict(endpoint),
            "desired_state": dict(endpoint),
            "feedback": None,
        }


def test_completion_v6_pins_two_states_and_endpoint_motion_allocation(
    tmp_path: Path,
) -> None:
    completion = persist_native_completion_fixture(tmp_path)

    assert isinstance(completion, PAContextGroundingCompletionV6)
    record = completion.to_record()
    assert record["schema_version"] == 6
    assert record["status"] == "grounding complete"
    assert record["allocation_label"] == "validated endpoint-motion allocation"
    assert record["validation_scope"] == "endpoint_motion"
    assert record["motion_executed"] is False
    assert "grounding_session_ref" not in record
    assert len(record["typed_context_refs"]) == 2
    contract = _read_json(tmp_path / str(record["typed_grounding_contract_ref"]))
    assert contract["schema_version"] == 6
    assert contract["semantic_review_ref"] == record["semantic_review_ref"]
    assert contract["source_refs"][0]["ref"] == "requirement_0001"
    assert contract["tool_call_refs"] == []
    assert contract["current_state_iri"].endswith("currentstate_0001")
    assert contract["desired_state_iri"].endswith("desiredstate_0001")
    assert contract["reachability_check_ref"] == record["reachability_check_ref"]
    assert contract["robot_agent_validation_ref"] == record["robot_agent_validation_ref"]
    assert "target_feature" not in contract
    assert "target_feature" not in record
    assert all(
        _read_json(tmp_path / item["ref"])["record_type"] == "RobotFrameLocationRecord"
        for item in record["typed_context_refs"]
    )
    proposal = _read_json(tmp_path / str(record["ontology_projection_ref"]))
    assert proposal["schema_version"] == 8
    assert proposal["feature_iri"].endswith("feature_0001")
    target_feature = proposal["output"]["target_feature"]
    assert target_feature["required_process"]["process_iri"] == (
        "https://cais-spade-llm.local/process/assembly"
    )
    assert target_feature["current_state"]["state_values"] == []
    assert target_feature["desired_state"]["state_values"] == []
    review = _read_json(tmp_path / str(record["semantic_review_ref"]))
    assert review["schema_version"] == 2
    assert review["record_type"] == "TargetFeatureSemanticReview"
    assert review["verdict"] == "complete"
    assert review["gap"] is None
    serialized = json.dumps(record)
    for forbidden in (
        "GroundingSession",
        "TaskTransitionContract",
        "primitive_steps",
        "RobotFramePoseRecord",
        "TargetFeatureGeometryRecord",
        "context_summary",
        "missing_information",
    ):
        assert forbidden not in serialized


def test_completion_v6_loader_rejects_changed_semantic_review(
    tmp_path: Path,
) -> None:
    completion = persist_native_completion_fixture(tmp_path).to_record()
    review_path = tmp_path / str(completion["semantic_review_ref"])
    review = _read_json(review_path)
    review["verdict"] = "incomplete"
    review["gap"] = "The medium gear is omitted."
    _write_json(review_path, review)

    with pytest.raises(GroundingContractError, match="semantic_review_ref hash"):
        load_pa_context_grounding_completion(tmp_path)


def test_completion_v6_pins_multiple_values_from_one_typed_record(
    tmp_path: Path,
) -> None:
    completion = persist_native_completion_fixture(
        tmp_path,
        include_target_state_values=True,
    ).to_record()
    proposal = _read_json(tmp_path / str(completion["ontology_projection_ref"]))
    state_values = proposal["output"]["target_feature"]["desired_state"]["state_values"]
    assert [item["name"] for item in state_values] == [
        "specified_finish",
        "specified_coating",
    ]
    assert len({item["value_ref"]["record_ref"] for item in state_values}) == 1
    assert [item["value_ref"]["field_path"] for item in state_values] == [
        "/overview/summary",
        "/overview/observations/0",
    ]
    state_ref = state_values[0]["value_ref"]["record_ref"]
    contract = _read_json(tmp_path / str(completion["typed_grounding_contract_ref"]))
    assert any(item["ref"] == state_ref for item in contract["typed_record_refs"])

    state_path = tmp_path / state_ref
    changed = _read_json(state_path)
    changed["overview"]["summary"] = "gloss"
    _write_json(state_path, changed)
    with pytest.raises(GroundingContractError, match="pinned record changed"):
        load_pa_context_grounding_completion(tmp_path)


def test_completion_v6_pins_answered_clarification_source(tmp_path: Path) -> None:
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

    pinned = next(item for item in completion["source_refs"] if item["ref"] == clarification_ref)
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
    contract_path = contract_root / str(contract_completion["typed_grounding_contract_ref"])
    contract = _read_json(contract_path)
    contract["source_refs"] = []
    _write_json(contract_path, contract)
    with pytest.raises(GroundingContractError, match="typed_grounding_contract_ref hash"):
        load_pa_context_grounding_completion(contract_root)


@pytest.mark.parametrize(
    "ref_field",
    [
        "registry_snapshot_ref",
        "workcell_snapshot_ref",
        "evidence_presentation_ref",
        "allocation_presentation_ref",
        "reachability_check_ref",
        "robot_agent_validation_ref",
    ],
)
def test_completion_v6_rejects_changed_allocation_lineage(
    tmp_path: Path,
    ref_field: str,
) -> None:
    completion = persist_native_completion_fixture(tmp_path).to_record()
    record_path = tmp_path / str(completion[ref_field])
    record = _read_json(record_path)
    record["tampered"] = True
    _write_json(record_path, record)

    with pytest.raises(GroundingContractError, match=f"{ref_field} hash"):
        load_pa_context_grounding_completion(tmp_path)


def test_unsupported_completion_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "interaction_record/context_completion_0001.json"
    _write_json(path, {"schema_version": 1})

    with pytest.raises(GroundingContractError, match="version is unsupported"):
        load_pa_context_grounding_completion(tmp_path)


def persist_native_completion_fixture(
    root: Path,
    *,
    evidence_refs: tuple[str, ...] = ("requirement_0001",),
    include_target_state_values: bool = False,
) -> PAContextGroundingCompletionV6:
    """Create one complete v6 native record chain for completion and UI tests."""
    requirement = "assemble medium gear"
    _write_json(
        root / "products/user_requirement/product_requirement.json",
        {"product_requirement": requirement},
    )
    tbox = ontology_config().load_tbox()
    initialize_interaction_abox(root, requirement, tbox)
    registry = load_predefined_resource_registry(tbox)
    workcell = load_predefined_workcell(tbox, registry)
    current_location_path, current_source_ref = _write_location(
        root,
        "current",
        (0.0, -0.7, 1.1),
    )
    desired_location_path, desired_source_ref = _write_location(
        root,
        "desired",
        (0.0, -0.2, 1.1),
    )
    current_location_ref = current_location_path.relative_to(root).as_posix()
    desired_location_ref = desired_location_path.relative_to(root).as_posix()
    validate_and_merge_triple_delta(
        root,
        tbox,
        "synthetic_world_location_provider",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [current_location_ref],
        },
        authorized_evidence_refs={current_source_ref},
    )
    desired_merge = validate_and_merge_triple_delta(
        root,
        tbox,
        "synthetic_world_location_provider",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [desired_location_ref],
        },
        authorized_evidence_refs={desired_source_ref},
    )
    target_state_record_ref: str | None = None
    proposal_abox = desired_merge.abox
    authorized_evidence_refs = {
        *evidence_refs,
        current_location_ref,
        desired_location_ref,
    }
    if include_target_state_values:
        target_state_path = _write_target_state_record(root)
        target_state_record_ref = target_state_path.relative_to(root).as_posix()
        authorized_evidence_refs.add(target_state_record_ref)
        target_state_merge = validate_and_merge_triple_delta(
            root,
            tbox,
            "document_evidence",
            {
                "assertions": [],
                "uncertainty": [],
                "unresolved_evidence_needs": [],
                "typed_context_refs": [target_state_record_ref],
            },
            authorized_evidence_refs={"requirement_0001"},
        )
        proposal_abox = target_state_merge.abox

    def resolve_typed_record(record_ref: str) -> Mapping[str, object]:
        record_types = {
            current_location_ref: "RobotFrameLocationRecord",
            desired_location_ref: "RobotFrameLocationRecord",
        }
        if target_state_record_ref is not None:
            record_types[target_state_record_ref] = "DocumentOverviewRecord"
        if record_ref not in record_types:
            raise ValueError("record is not accepted")
        record_path = root / record_ref
        return {
            "record_type": record_types[record_ref],
            "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
            "record": _read_json(record_path),
        }

    proposal_agent = _ProposalAgent(
        evidence_refs,
        current_location_ref,
        desired_location_ref,
        target_state_record_ref,
    )
    candidate = asyncio.run(
        propose_and_validate_ontology_grounding(
            proposal_agent,
            interaction_root=root,
            tbox=tbox,
            abox=proposal_abox,
            workcell=workcell,
            evidence_catalog=[],
            authorized_evidence_refs=authorized_evidence_refs,
            tools=[],
            tool_executor=_unused_tool,
            max_tool_rounds=1,
            required_output_projection={
                "record_type": "RobotFrameLocationRecord",
                "target_frame": "world",
            },
            typed_record_resolver=resolve_typed_record,
        )
    )
    assert isinstance(candidate, OntologyGroundingCandidate)
    semantic_review = asyncio.run(
        review_target_feature_semantics(
            proposal_agent,
            interaction_root=root,
            candidate=candidate,
            product_requirement=requirement,
            evidence_catalog=[],
        )
    )
    proposal = commit_ontology_grounding_candidate(
        candidate,
        interaction_root=root,
        tbox=tbox,
        abox=proposal_abox,
        workcell=workcell,
        authorized_evidence_refs=authorized_evidence_refs,
        semantic_review=semantic_review,
    )
    assert isinstance(proposal, OntologyGroundingResult)

    evidence_presentation = load_or_create_evidence_presentation(
        root,
        sources=((None, "observation", "fresh_on_call"),),
        explicit_order=("__live_observation__",),
    )
    evidence_sources = tuple(
        _allocation_evidence_source(root, path)
        for path in (current_location_path, desired_location_path)
    )
    resource_catalog = candidate_resource_catalog(proposal.merge.abox, registry, workcell)
    process_iri = str(
        proposal.proposal.target_feature["required_process"]["process_iri"]
    )
    allocation_presentation = load_or_create_allocation_presentation(
        root,
        evidence_presentation=evidence_presentation,
        process_symbol=workcell.process_symbol_for_iri(process_iri),
        process_iri=process_iri,
        feature_iri=proposal.proposal.feature_iri,
        current_state_iri=f"{proposal.merge.abox.namespace}currentstate_0001",
        desired_state_iri=f"{proposal.merge.abox.namespace}desiredstate_0001",
        resources=tuple(
            (symbol, str(entry["resource_iri"]), str(entry["resource_jid"]))
            for symbol, entry in resource_catalog.items()
        ),
        evidence_sources=evidence_sources,
        explicit_resource_order=tuple(resource_catalog),
        explicit_candidate_order=tuple(
            source.canonical_key for source in evidence_sources
        ),
    )
    evidence_handles = {
        entry.canonical_key: entry.pa_handle
        for entry in allocation_presentation.evidence_entries
    }
    reachability = check_resource_reachability(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol="xarm6",
        allocation_presentation=allocation_presentation,
        current_state_evidence_handle=evidence_handles[
            evidence_sources[0].canonical_key
        ],
        desired_state_evidence_handle=evidence_handles[
            evidence_sources[1].canonical_key
        ],
        current_location_record_path=current_location_path,
        desired_location_record_path=desired_location_path,
    )
    validation = asyncio.run(
        validate_provisional_allocation(
            _AcceptedFeasibilityRuntime(),
            interaction_root=root,
            workcell=workcell,
            reachability=reachability,
        )
    )
    selection = persist_pa_resource_selection(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        reachability=reachability,
        allocation_presentation=allocation_presentation,
        robot_agent_validation_path=validation.record_path,
    )
    assignment = commit_resource_assignment(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
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
                "allocation_label": "validated endpoint-motion allocation",
            },
            "failure": None,
        },
    )
    persist_pa_context_grounding_completion_v6(
        root,
        tbox=tbox,
        product_requirement=requirement,
        completion_turn=1,
        decision_ref=turn_path.relative_to(root).as_posix(),
        product_context=final_view,
        ontology_projection_ref=proposal_ref,
        resource_selection_ref=selection.record_ref,
        tool_call_refs=(),
    )
    loaded = load_pa_context_grounding_completion(root)
    assert isinstance(loaded, PAContextGroundingCompletionV6)
    return loaded


async def _unused_tool(
    tool_name: str,
    arguments: Mapping[str, object],
) -> Mapping[str, object]:
    del tool_name, arguments
    raise AssertionError("No tool call was expected.")


def _write_target_state_record(root: Path) -> Path:
    path = root / "products/grounding/test_target_state/document_overview.json"
    _write_json(
        path,
        {
            "schema_version": 2,
            "record_type": "DocumentOverviewRecord",
            "producer": "document_evidence",
            "overview": {
                "summary": "matte",
                "observations": ["primer"],
            },
        },
    )
    return path


def _write_location(
    root: Path,
    name: str,
    translation: tuple[float, float, float],
) -> tuple[Path, str]:
    destination = root / "products/grounding/synthetic_world_location" / name
    destination.mkdir(parents=True, exist_ok=True)
    source_path = destination / "source_evidence.json"
    _write_json(source_path, {"source": "synthetic observed center"})
    source_ref = source_path.relative_to(root).as_posix()
    rgb_path = destination / "selected_rgb.png"
    rgb_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
            "AAAAC0lEQVR42mP8/x8AAusB9Y9Zl4sAAAAASUVORK5CYII="
        )
    )
    rgb_ref = rgb_path.relative_to(root).as_posix()
    segmentation_path = destination / "segmentation_record.json"
    _write_json(
        segmentation_path,
        {
            "schema_version": 2,
            "record_type": "RGBDSegmentationRecord",
            "producer": "synthetic_neutral_segmentation_provider",
            "cameras": [
                {
                    "observation_handle": f"observation_{name}",
                    "source_artifacts": {
                        "rgb": {
                            "ref": rgb_ref,
                            "sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                        }
                    },
                    "candidates": [
                        {
                            "candidate_handle": f"candidate_{name}",
                            "pixel_bounds_uv": {
                                "minimum": [10, 10],
                                "maximum": [45, 45],
                            },
                        }
                    ],
                    "label_mask_artifact": {"shape": [64, 64]},
                }
            ],
        },
    )
    segmentation_ref = segmentation_path.relative_to(root).as_posix()
    path = destination / "robot_frame_location_record.json"
    _write_json(
        path,
        {
            "schema_version": 2,
            "record_type": "RobotFrameLocationRecord",
            "producer": "synthetic_world_location_provider",
            "method": "neutral_fixture",
            "source_segmentation": {
                "ref": segmentation_ref,
                "sha256": hashlib.sha256(segmentation_path.read_bytes()).hexdigest(),
            },
            "robot_frame_conversion": "accepted",
            "location": "available",
            "source_frame": "camera_optical_frame",
            "target_frame": "world",
            "observation_timestamp_ns": 11,
            "translated_location_m": list(translation),
            "candidate_reference": {
                "observation_handle": f"observation_{name}",
                "candidate_handle": f"candidate_{name}",
            },
            "source_evidence": {
                "ref": source_ref,
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            },
        },
    )
    return path, source_ref


def _allocation_evidence_source(
    root: Path,
    location_path: Path,
) -> AllocationEvidenceSource:
    record = _read_json(location_path)
    record_ref = location_path.relative_to(root).as_posix()
    candidate_reference = record["candidate_reference"]
    return AllocationEvidenceSource(
        canonical_key=f"{record_ref}#/translated_location_m",
        record_type="RobotFrameLocationRecord",
        record_ref=record_ref,
        record_sha256=hashlib.sha256(location_path.read_bytes()).hexdigest(),
        field_path="/translated_location_m",
        observation_handle=str(candidate_reference["observation_handle"]),
        candidate_handle=str(candidate_reference["candidate_handle"]),
        source_frame=str(record["source_frame"]),
        neutral_projection={"location_record_available": True},
    )


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
