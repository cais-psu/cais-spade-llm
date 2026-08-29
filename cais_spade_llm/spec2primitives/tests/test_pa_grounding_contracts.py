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

def test_grounding_session_round_trip_is_source_cited_and_replay_safe(
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
        {"action": "inspect", "source_ref": "manual.pdf", "question": "How is it assembled?"}
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
    assert session.next_action.question == "How is it assembled?"
    assert not {
        "statements",
        "information_needs",
        "information_need_transitions",
        "answer_statement_ids",
        "understanding",
    }.intersection(session.to_record())
    assert len(session.fingerprint) == 64


def test_grounding_session_rejects_answer_shaped_fields_and_repeated_actions() -> None:
    with pytest.raises(GroundingContractError, match="action is invalid"):
        GroundingNextAction.from_mapping({"action": "fill_ontology"})

    value = {"action": "propose_grounding", "understanding": "hidden state"}
    with pytest.raises(GroundingContractError, match="fields are invalid"):
        GroundingNextAction.from_mapping(value)

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
    duplicate = GroundingActionAttempt.from_mapping(
        {**attempt.to_record(), "attempt_id": "attempt_0002"}
    )
    with pytest.raises(GroundingContractError, match="must not repeat an action"):
        GroundingSession.create(
            revision=2,
            requirement_text="assemble medium gear",
            attempted_actions=[attempt, duplicate],
            next_action=GroundingNextAction.from_mapping(
                {"action": "propose_grounding"}
            ),
            status="waiting_for_evidence",
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


def test_changed_embedded_source_reopens_world_pose_as_stale(
    tmp_path: Path,
) -> None:
    tbox = ontology_config().load_tbox()
    initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    record_root = tmp_path / "products/grounding/world_pose_provider"
    record_root.mkdir(parents=True)
    source_path = record_root / "source_0001.json"
    source_path.write_text('{"capture":"accepted"}\n', encoding="utf-8")
    source_ref = source_path.relative_to(tmp_path).as_posix()
    pose_path = record_root / "world_pose_0001.json"
    pose_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "RobotFramePoseRecord",
                "producer": "world_pose_provider",
                "target_frame": "world",
                "observation_timestamp_ns": 10,
                "robot_frame_conversion": "accepted",
                "source_evidence": {
                    "ref": source_ref,
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                },
                "robot_frame_pose": {
                    "CAD_origin_translation_m": [0.0, -0.5, 1.1]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "world_pose_provider",
        {
            "assertions": [],
            "typed_context_refs": [pose_path.relative_to(tmp_path).as_posix()],
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
