"""Tests for the PA-owned Phase 4.3 grounding contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import RDF

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    ContextNeed,
    GroundingContractError,
    GroundingProducerDescriptor,
    TaskTransitionDraft,
    build_product_context_view,
    persist_product_context_view,
    select_grounding_producer,
    unresolved_context_needs,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    PPR_NAMESPACE,
    ontology_config,
)


def _need(
    kind: str,
    symbol: str,
    *,
    frame: str | None = None,
    maximum_age_ns: int | None = None,
) -> ContextNeed:
    return ContextNeed.from_mapping(
        {
            "kind": kind,
            "symbol": symbol,
            "subject_role": "requested_product",
            "authority": "PA",
            "frame": frame,
            "maximum_age_ns": maximum_age_ns,
            "reason": "required by the robot-independent task draft",
        }
    )


def _descriptor(
    producer: str,
    kind: str,
    symbol: str,
    *,
    priority: int,
) -> GroundingProducerDescriptor:
    return GroundingProducerDescriptor.from_mapping(
        {
            "producer": producer,
            "supported_outputs": [{"kind": kind, "symbol": symbol}],
            "evidence_types": ["document"],
            "required_record_types": [],
            "priority": priority,
        }
    )


def _draft(view_fingerprint: str, needs: list[ContextNeed]) -> TaskTransitionDraft:
    return TaskTransitionDraft.from_mapping(
        {
            "version": 1,
            "product_requirement": "assemble Medium Gear",
            "requested_process": f"{PPR_NAMESPACE}assembly",
            "required_outcome": "Medium Gear assembled",
            "required_inputs": [need.to_record() for need in needs],
            "unresolved_user_intent": None,
            "source_view_fingerprint": view_fingerprint,
        }
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


def test_required_inputs_minus_valid_context_is_deterministic(tmp_path: Path) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    product_need = _need("class", f"{PPR_NAMESPACE}product")
    pose_need = _need(
        "typed_context_record",
        "CADPoseEstimationRecord",
        frame="cam_mk4_2_optical_frame",
        maximum_age_ns=100,
    )
    view = build_product_context_view(
        tmp_path,
        abox,
        attempted_evidence=(),
        assessed_at_ns=1_000,
    )

    assert unresolved_context_needs(_draft(view.fingerprint, [product_need, pose_need]), view) == (
        product_need,
        pose_need,
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


def test_descriptor_selection_uses_exact_output_priority_and_attempt_history() -> None:
    need = _need("class", f"{PPR_NAMESPACE}product")
    primary = _descriptor(
        "document_evidence",
        "class",
        f"{PPR_NAMESPACE}product",
        priority=0,
    )
    alternative = _descriptor(
        "approved_product_record",
        "class",
        f"{PPR_NAMESPACE}product",
        priority=1,
    )

    assert select_grounding_producer(need, [alternative, primary]) == primary
    assert (
        select_grounding_producer(
            need,
            [primary, alternative],
            attempted_producers=["document_evidence"],
        )
        == alternative
    )
    with pytest.raises(GroundingContractError, match="No untried"):
        select_grounding_producer(
            need,
            [primary, alternative],
            attempted_producers=["document_evidence", "approved_product_record"],
        )


def test_pre_ra_contract_rejects_ra_authority_and_fixed_symbol_changes() -> None:
    value = _need("class", f"{PPR_NAMESPACE}product").to_record()
    value["authority"] = "RA"
    with pytest.raises(GroundingContractError, match="must be PA"):
        ContextNeed.from_mapping(value)

    descriptor_value = _descriptor(
        "document_evidence",
        "class",
        f"{PPR_NAMESPACE}product",
        priority=0,
    ).to_record()
    descriptor_value["supported_outputs"] = [
        {"kind": "class", "symbol": f"{PPR_NAMESPACE}Product"}
    ]
    changed = GroundingProducerDescriptor.from_mapping(descriptor_value)
    assert not changed.supports(_need("class", f"{PPR_NAMESPACE}product"))
