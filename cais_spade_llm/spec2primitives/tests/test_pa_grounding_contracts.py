"""Tests for the PA-owned Phase 4.3 grounding contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import RDF

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingActionAttempt,
    GroundingContractError,
    GroundingDecision,
    GroundingProducerDescriptor,
    GroundingSession,
    GroundingStatement,
    InformationNeed,
    build_product_context_view,
    load_latest_grounding_session,
    persist_grounding_session,
    persist_product_context_view,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    PPR_NAMESPACE,
    ontology_config,
)


def _descriptor(
    provider_id: str,
    record_type: str,
) -> GroundingProducerDescriptor:
    return GroundingProducerDescriptor.from_mapping(
        {
            "provider_id": provider_id,
            "description": "Controlled provider capability.",
            "accepted_evidence_types": ["document"],
            "produced_record_types": [record_type],
            "prerequisites": {record_type: []},
            "availability": True,
            "estimated_cost": 1,
        }
    )

def test_grounding_session_round_trip_is_source_cited_and_replay_safe(
    tmp_path: Path,
) -> None:
    statement = GroundingStatement.from_mapping(
        {
            "statement_id": "statement_0001",
            "text": "The named item is Medium Gear.",
            "status": "directly_stated",
            "sources": ["requirement_0001"],
            "reason": "The exact requirement names the item.",
        }
    )
    need = InformationNeed.from_mapping(
        {
            "need_id": "need_0001",
            "question": "What approved evidence describes the named item?",
            "required": True,
            "sources": ["requirement_0001"],
            "accepted_record_types": ["DocumentOverviewRecord"],
            "status": "open",
            "answer_statement_ids": [],
        }
    )
    attempt = GroundingActionAttempt.from_mapping(
        {
            "attempt_id": "attempt_0001",
            "need_id": "need_0001",
            "provider_id": "document_evidence",
            "source_ref": "manual.pdf",
            "source_revision": "a" * 64,
            "status": "no_change",
            "record_refs": [],
        }
    )
    decision = GroundingDecision.from_mapping(
        {
            "decision_type": "request_evidence",
            "need_id": "need_0001",
            "provider_id": "document_evidence",
            "source_ref": "manual.pdf",
            "source_revision": "b" * 64,
            "query": "Find information about the named item.",
            "reason": "A changed approved source revision remains available.",
        }
    )
    session = GroundingSession.create(
        revision=2,
        requirement_text="assemble medium gear",
        statements=[statement],
        information_needs=[need],
        attempted_actions=[attempt],
        evidence_refs=["requirement_0001"],
        decision=decision,
        status="waiting_for_evidence",
        information_status="partial",
    )

    path = persist_grounding_session(tmp_path, session)

    assert path.name == "revision_0002.json"
    assert load_latest_grounding_session(tmp_path) == session
    assert session.missing_information == (need.question,)
    assert len(session.fingerprint) == 64


def test_grounding_session_rejects_answer_shaped_fields_and_repeated_actions() -> None:
    statement = GroundingStatement.from_mapping(
        {
            "statement_id": "statement_0001",
            "text": "The named item is Medium Gear.",
            "status": "directly_stated",
            "sources": ["requirement_0001"],
            "reason": "The exact requirement names the item.",
        }
    )
    value = statement.to_record()
    value["assembly_context"] = {"destination": "expected answer"}
    with pytest.raises(GroundingContractError, match="fields are invalid"):
        GroundingStatement.from_mapping(value)

    need = InformationNeed.from_mapping(
        {
            "need_id": "need_0001",
            "question": "What evidence describes the named item?",
            "required": True,
            "sources": ["requirement_0001"],
            "accepted_record_types": ["DocumentOverviewRecord"],
            "status": "open",
            "answer_statement_ids": [],
        }
    )
    attempt = GroundingActionAttempt.from_mapping(
        {
            "attempt_id": "attempt_0001",
            "need_id": need.need_id,
            "provider_id": "document_evidence",
            "source_ref": "manual.pdf",
            "source_revision": "a" * 64,
            "status": "no_change",
            "record_refs": [],
        }
    )
    repeated = GroundingDecision.from_mapping(
        {
            "decision_type": "request_evidence",
            "need_id": need.need_id,
            "provider_id": attempt.provider_id,
            "source_ref": attempt.source_ref,
            "source_revision": attempt.source_revision,
            "query": "Try the same source again.",
            "reason": "This action must be rejected.",
        }
    )
    with pytest.raises(GroundingContractError, match="repeats an attempted"):
        GroundingSession.create(
            revision=2,
            requirement_text="assemble medium gear",
            statements=[statement],
            information_needs=[need],
            attempted_actions=[attempt],
            evidence_refs=["requirement_0001"],
            decision=repeated,
            status="waiting_for_evidence",
            information_status="partial",
        )


def test_view_joins_semantic_assertions_and_validated_typed_records(
    tmp_path: Path,
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    record_path = (
        tmp_path
        / "products/grounding/rgb_d_cad_grounding/pose_0001/pose_record.json"
    )
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "CADPoseEstimationRecord",
                "producer": "rgb_d_cad_grounding",
                "evidence_refs": ["Gear_Medium.STL", "observation_0001"],
                "pose": "accepted",
                "coordinate_frame": "cam_mk4_2_optical_frame",
                "observation_timestamp_ns": 950,
            }
        ),
        encoding="utf-8",
    )
    product_iri = f"{abox.namespace}product_1"
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "rgb_d_cad_grounding",
        {
            "assertions": [
                {
                    "subject": product_iri,
                    "predicate": str(RDF.type),
                    "object": {
                        "kind": "iri",
                        "value": f"{PPR_NAMESPACE}product",
                    },
                    "evidence_refs": ["Gear_Medium.STL"],
                }
            ],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [
                "products/grounding/rgb_d_cad_grounding/pose_0001/pose_record.json"
            ],
        },
        authorized_evidence_refs=["Gear_Medium.STL", "observation_0001"],
    )

    view = build_product_context_view(
        tmp_path,
        merge.abox,
        attempted_evidence=("Gear_Medium.STL", "observation_0001"),
        assessed_at_ns=1_000,
    )

    assert view.delta_count == 1
    assert view.typed_bindings[0].record_type == "CADPoseEstimationRecord"
    assert view.typed_bindings[0].status == "accepted"
    assert view.typed_bindings[0].frame == "cam_mk4_2_optical_frame"
    assert len(view.fingerprint) == 64
    persisted = persist_product_context_view(tmp_path, view)
    assert json.loads(persisted.read_text(encoding="utf-8"))["fingerprint"] == (
        view.fingerprint
    )


def test_tampered_or_out_of_root_typed_context_is_rejected(tmp_path: Path) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    delta_path = abox.ontology_root / "delta_0001.json"
    delta_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "delta_number": 1,
                "producer": "rgb_d_cad_grounding",
                "assertions": [],
                "uncertainty": [],
                "unresolved_evidence_needs": [],
                "typed_context_refs": ["../../outside.json"],
            }
        ),
        encoding="utf-8",
    )
    object.__setattr__(abox, "delta_count", 1)

    with pytest.raises(GroundingContractError, match="leaves products/grounding"):
        build_product_context_view(
            tmp_path,
            abox,
            attempted_evidence=(),
            assessed_at_ns=1,
        )


def test_document_typed_record_requires_explicit_record_type(tmp_path: Path) -> None:
    tbox = ontology_config().load_tbox()
    initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    record_path = (
        tmp_path / "products/grounding/document_evidence/overview_0001.json"
    )
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer": "document_evidence",
                "evidence_refs": [],
                "overview": {},
            }
        ),
        encoding="utf-8",
    )
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "document_evidence",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [str(record_path.relative_to(tmp_path))],
        },
        authorized_evidence_refs=[],
    )

    with pytest.raises(
        GroundingContractError,
        match="typed record_type must be a non-empty string",
    ):
        build_product_context_view(
            tmp_path,
            merge.abox,
            attempted_evidence=(),
            assessed_at_ns=1,
        )


def test_descriptor_advertises_capability_without_controller_priority() -> None:
    descriptor = _descriptor("document_evidence", "DocumentOverviewRecord")

    assert descriptor.supports_record_type("DocumentOverviewRecord")
    assert descriptor.prerequisites_for("DocumentOverviewRecord") == ()
    assert "priority" not in descriptor.to_record()


def test_provider_descriptor_preserves_exact_fixed_output_symbol() -> None:
    descriptor_value = _descriptor(
        "document_evidence",
        "DocumentOverviewRecord",
    ).to_record()
    descriptor_value["produced_record_types"] = ["documentoverviewrecord"]
    descriptor_value["prerequisites"] = {"documentoverviewrecord": []}
    changed = GroundingProducerDescriptor.from_mapping(descriptor_value)
    assert not changed.supports_record_type("DocumentOverviewRecord")
