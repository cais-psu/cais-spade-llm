"""Tests for the PA-owned Phase 4.3 grounding contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from rdflib import RDF

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingActionAttempt,
    GroundingContractError,
    GroundingNextAction,
    GroundingProducerDescriptor,
    GroundingSession,
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

def test_legacy_grounding_session_round_trip_remains_readable(
    tmp_path: Path,
) -> None:
    attempt = GroundingActionAttempt.from_mapping(
        {
            "attempt_id": "attempt_0001",
            "action": "retrieve",
            "provider_id": "document_evidence",
            "source_ref": "manual.pdf",
            "source_revision": "a" * 64,
            "question": None,
            "status": "no_change",
            "record_refs": [],
        }
    )
    next_action = GroundingNextAction.from_mapping(
        {"action": "retrieve", "source_ref": "manual.pdf"}
    )
    session = GroundingSession.create(
        revision=2,
        requirement_text="assemble medium gear",
        attempted_actions=[attempt],
        next_action=next_action,
        selected_provider_id="document_evidence",
        selected_source_revision="b" * 64,
        status="waiting_for_evidence",
    )

    path = persist_grounding_session(tmp_path, session)

    assert path.name == "revision_0002.json"
    assert load_latest_grounding_session(tmp_path) == session
    assert attempt.action_key == (
        "retrieve",
        "document_evidence",
        "manual.pdf",
        "a" * 64,
        None,
    )
    assert session.next_action.source_ref == "manual.pdf"
    assert not {
        "statements",
        "information_needs",
        "information_need_transitions",
        "answer_statement_ids",
        "understanding",
    }.intersection(session.to_record())
    assert len(session.fingerprint) == 64

def test_view_joins_semantic_assertions_and_validated_typed_records(
    tmp_path: Path,
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    record_path = (
        tmp_path
        / "products/grounding/robot_frame_location/location_0001.json"
    )
    record_path.parent.mkdir(parents=True)
    source_path = record_path.parent / "source_0001.json"
    source_path.write_text('{"observation":"accepted"}\n', encoding="utf-8")
    record_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "RobotFrameLocationRecord",
                "producer": "robot_frame_location_provider",
                "robot_frame_conversion": "accepted",
                "CAD_correspondence": "accepted",
                "location": "available",
                "source_frame": "cam_mk4_2_optical_frame",
                "target_frame": "world",
                "observation_timestamp_ns": 950,
                "translated_location_m": [0.0, -0.5, 1.1],
                "source_hashes": [
                    {
                        "ref": source_path.relative_to(tmp_path).as_posix(),
                        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    product_iri = f"{abox.namespace}product_1"
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "robot_frame_location_provider",
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
                "products/grounding/robot_frame_location/location_0001.json"
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
    assert view.typed_bindings[0].record_type == "RobotFrameLocationRecord"
    assert view.typed_bindings[0].status == "accepted"
    assert view.typed_bindings[0].frame == "world"
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


def test_changed_embedded_source_reopens_world_location_as_stale(
    tmp_path: Path,
) -> None:
    tbox = ontology_config().load_tbox()
    initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    record_root = tmp_path / "products/grounding/world_location_provider"
    record_root.mkdir(parents=True)
    source_path = record_root / "source_0001.json"
    source_path.write_text('{"capture":"accepted"}\n', encoding="utf-8")
    source_ref = source_path.relative_to(tmp_path).as_posix()
    location_path = record_root / "world_location_0001.json"
    location_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "RobotFrameLocationRecord",
                "producer": "world_location_provider",
                "source_frame": "camera_optical_frame",
                "target_frame": "world",
                "observation_timestamp_ns": 10,
                "robot_frame_conversion": "accepted",
                "CAD_correspondence": "accepted",
                "location": "available",
                "translated_location_m": [0.0, -0.5, 1.1],
                "source_hashes": [
                    {
                        "ref": source_ref,
                        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "world_location_provider",
        {
            "assertions": [],
            "typed_context_refs": [location_path.relative_to(tmp_path).as_posix()],
        },
        authorized_evidence_refs=[],
    )
    accepted = build_product_context_view(
        tmp_path,
        merge.abox,
        attempted_evidence=(),
        assessed_at_ns=10,
    )
    assert accepted.typed_bindings[0].status == "accepted"

    source_path.write_text('{"capture":"changed"}\n', encoding="utf-8")
    reopened = build_product_context_view(
        tmp_path,
        merge.abox,
        attempted_evidence=(),
        assessed_at_ns=11,
    )

    assert reopened.typed_bindings[0].status == "stale"
    assert reopened.typed_bindings[0].record_ref == accepted.typed_bindings[0].record_ref


def test_dynamic_document_and_layout_records_enter_hash_validated_typed_context(
    tmp_path: Path,
) -> None:
    def fingerprinted(value: dict[str, object]) -> dict[str, object]:
        value["fingerprint"] = hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return value

    tbox = ontology_config().load_tbox()
    initialize_interaction_abox(tmp_path, "an unforeseen requirement", tbox)
    evidence_root = tmp_path / "products/grounding"
    document_root = evidence_root / "document_evidence"
    geometry_root = evidence_root / "rgb_d_cad_grounding"
    document_root.mkdir(parents=True)
    geometry_root.mkdir(parents=True)

    source_index = fingerprinted(
        {"pages": [{"page": 1, "extracted_text": "text"}]}
    )
    source_index_path = document_root / "source_index_0001.json"
    source_index_path.write_text(
        json.dumps(
            fingerprinted(
            {
                "schema_version": 1,
                "record_type": "DocumentSourceIndexRecord",
                "producer": "document_evidence",
                "status": "accepted",
                "evidence_refs": ["manual.pdf#page=1"],
                "source_index": source_index,
            }
            )
        ),
        encoding="utf-8",
    )
    source_index_ref = source_index_path.relative_to(tmp_path).as_posix()
    query_path = document_root / "query_0001.json"
    query_path.write_text(
        json.dumps(
            fingerprinted(
            {
                "schema_version": 1,
                "record_type": "DocumentQueryRecord",
                "producer": "document_evidence",
                "status": "supported",
                "question": "What is stated?",
                "question_sha256": hashlib.sha256(b"What is stated?").hexdigest(),
                "source_index": {
                    "ref": source_index_ref,
                    "sha256": hashlib.sha256(source_index_path.read_bytes()).hexdigest(),
                },
                "claims": [{"predicate_text": "open predicate", "arguments": []}],
                "uncertainty": [],
                "evidence_refs": [source_index_ref, "manual.pdf#page=1"],
            }
            )
        ),
        encoding="utf-8",
    )
    query_ref = query_path.relative_to(tmp_path).as_posix()
    historical_path = document_root / "overview_0001.json"
    historical_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "record_type": "DocumentOverviewRecord",
                "producer": "document_evidence",
                "overview": {},
                "evidence_refs": [],
            }
        ),
        encoding="utf-8",
    )
    historical_ref = historical_path.relative_to(tmp_path).as_posix()
    document_merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "document_evidence",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [source_index_ref, query_ref, historical_ref],
        },
        authorized_evidence_refs=["manual.pdf#page=1", source_index_ref],
    )

    comparison_path = geometry_root / "comparison.json"
    segmentation_path = geometry_root / "segmentation.json"
    comparison_path.write_text('{"comparison":"ambiguous"}\n', encoding="utf-8")
    segmentation_path.write_text('{"candidates":3}\n', encoding="utf-8")
    comparison_ref = comparison_path.relative_to(tmp_path).as_posix()
    segmentation_ref = segmentation_path.relative_to(tmp_path).as_posix()
    relation_path = geometry_root / "relation_0001.json"
    relation_path.write_text(
        json.dumps(
            fingerprinted(
            {
                "schema_version": 1,
                "record_type": "CandidateSpatialRelationRecord",
                "producer": "rgb_d_cad_grounding",
                "status": "accepted",
                "comparison": {
                    "ref": comparison_ref,
                    "sha256": hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
                },
                "segmentation": {
                    "ref": segmentation_ref,
                    "sha256": hashlib.sha256(segmentation_path.read_bytes()).hexdigest(),
                },
                "candidate_count": 3,
                "candidates": ["candidate_a", "candidate_b", "candidate_c"],
                "relations": [{"predicate_text": "between"}],
                "evidence_refs": [comparison_ref, segmentation_ref],
            }
            )
        ),
        encoding="utf-8",
    )
    relation_ref = relation_path.relative_to(tmp_path).as_posix()
    geometry_merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "rgb_d_cad_grounding",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [relation_ref],
        },
        authorized_evidence_refs=[comparison_ref, segmentation_ref],
    )
    view = build_product_context_view(
        tmp_path,
        geometry_merge.abox,
        attempted_evidence=(),
        assessed_at_ns=1,
    )
    statuses = {
        binding.record_type: binding.status for binding in view.typed_bindings
    }
    assert statuses == {
        "DocumentSourceIndexRecord": "accepted",
        "DocumentQueryRecord": "accepted",
        "DocumentOverviewRecord": "accepted",
        "CandidateSpatialRelationRecord": "accepted",
    }
    assert document_merge.abox.delta_count == 1

    changed_source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    changed_source_index["source_index"]["pages"][0]["extracted_text"] = "changed"
    source_index_path.write_text(json.dumps(changed_source_index), encoding="utf-8")
    reopened = build_product_context_view(
        tmp_path,
        geometry_merge.abox,
        attempted_evidence=(),
        assessed_at_ns=2,
    )
    query_binding = next(
        binding
        for binding in reopened.typed_bindings
        if binding.record_type == "DocumentQueryRecord"
    )
    assert query_binding.status == "stale"


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
