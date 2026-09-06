from __future__ import annotations

"""Tests for the PA-owned Phase 4.3 grounding contracts."""


import copy
import hashlib
import json
from pathlib import Path

import pytest
from rdflib import RDF, Namespace, URIRef

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingContractError,
    GroundingProducerDescriptor,
    build_product_context_view,
    persist_product_context_view,
    source_uncertainty,
)
from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import (
    OntologyGroundingAttempt,
    OntologyGroundingError,
    eligible_allocation_pairs,
    validate_ontology_grounding_attempt,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.ontology import (
    load_predefined_resource_registry,
    load_predefined_workcell,
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


def test_view_joins_semantic_assertions_and_validated_typed_records(
    tmp_path: Path,
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    record_path = tmp_path / "products/grounding/robot_frame_location/location_0001.json"
    record_path.parent.mkdir(parents=True)
    source_path = record_path.parent / "source_0001.json"
    source_path.write_text('{"observation":"accepted"}\n', encoding="utf-8")
    record_path.write_text(
        json.dumps(
            {
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
            "typed_context_refs": ["products/grounding/robot_frame_location/location_0001.json"],
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
    assert json.loads(persisted.read_text(encoding="utf-8"))["fingerprint"] == (view.fingerprint)


def test_tampered_or_out_of_root_typed_context_is_rejected(tmp_path: Path) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    delta_path = abox.ontology_root / "delta_0001.json"
    delta_path.write_text(
        json.dumps(
            {
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

    source_index = fingerprinted({"pages": [{"page": 1, "extracted_text": "text"}]})
    source_index_path = document_root / "source_index_0001.json"
    source_index_path.write_text(
        json.dumps(
            fingerprinted(
                {
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

    from cais_spade_llm.spec2primitives.tests.test_cad_size_correspondence import (
        _write_layout_inputs,
    )
    from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import analyze_candidate_layout

    segmentation_path, candidate_paths = _write_layout_inputs(
        tmp_path, ((0.0, 0.0, 0.5), (0.1, 0.0, 0.5), (0.2, 0.0, 0.5))
    )
    relation = analyze_candidate_layout(
        interaction_root=tmp_path,
        segmentation_record_path=segmentation_path,
        candidate_field_paths=candidate_paths,
    )
    relation_ref = relation.record_path.relative_to(tmp_path).as_posix()
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
        authorized_evidence_refs=[segmentation_path.relative_to(tmp_path).as_posix()],
    )
    view = build_product_context_view(
        tmp_path,
        geometry_merge.abox,
        attempted_evidence=(),
        assessed_at_ns=1,
    )
    statuses = {binding.record_type: binding.status for binding in view.typed_bindings}
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
    record_path = tmp_path / "products/grounding/document_evidence/overview_0001.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
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


def _target():
    return {
        "required_process": {
            "process_iri": "https://cais-spade-llm.local/process/assembly",
            "evidence_refs": ["requirement_0001"],
        },
        "current_state": {
            "statement": {
                "text": "Current attachment is unresolved.",
                "evidence_refs": ["requirement_0001"],
            },
            "state_values": [],
        },
        "desired_state": {
            "statement": {
                "text": "The requested product is assembled.",
                "evidence_refs": ["requirement_0001"],
            },
            "state_values": [],
        },
        "assembly_feature_association": [],
    }


def _association(index=0):
    return {
        "state_names": ["desired_state"],
        "assembly": {"name": "Assembly", "evidence_refs": ["requirement_0001"]},
        "assembly_features": [
            {
                "name": f"feature_{index}_{number}",
                "owner": {
                    "name": f"Part {number}",
                    "type": "Part",
                    "evidence_refs": ["requirement_0001"],
                },
                "state_name": None,
                "state_value_name": None,
                "evidence_refs": ["requirement_0001"],
            }
            for number in range(2)
        ],
        "evidence_refs": ["requirement_0001"],
    }


def _candidate(root, target):
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble product", tbox)
    workcell = load_predefined_workcell(tbox, load_predefined_resource_registry(tbox))
    attempt = OntologyGroundingAttempt(
        root / "products/grounding/ontology_grounding/proposal_0001.json",
        1,
        {"target_feature": target},
    )
    return validate_ontology_grounding_attempt(
        attempt,
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        workcell=workcell,
        authorized_evidence_refs={"requirement_0001"},
    ), abox


@pytest.mark.parametrize("count", [0, 1, 3])
def test_pairwise_associations_do_not_force_coordinate_roles(tmp_path, count):
    target = _target()
    target["assembly_feature_association"] = [_association(index) for index in range(count)]
    candidate, abox = _candidate(tmp_path, target)
    graph = candidate.provisional_abox.graph
    ppr = Namespace("http://PAonto.com#")
    assert (URIRef(candidate.proposal.feature_iri), RDF.type, ppr.feature) in graph
    associations = set(graph.subjects(RDF.type, ppr.AssemblyFeatureAssociation))
    assert len(associations) == count
    assert all(
        len(set(graph.objects(iri, ppr.relatesAssemblyFeature))) == 2 for iri in associations
    )
    assert len(set(graph.subjects(RDF.type, ppr.Part))) == (2 if count else 0)
    assert candidate.proposal.assembly_state_value_refs == {}
    assert abox.delta_count == 0


@pytest.mark.parametrize("count", [0, 1, 3])
def test_current_proposals_attach_pairwise_relationships_to_their_stated_state(tmp_path, count):
    target = _target()
    target["assembly_feature_association"] = [
        {**_association(index), "state_names": ["desired_state"]} for index in range(count)
    ]
    candidate, _ = _candidate(tmp_path, target)
    graph = candidate.provisional_abox.graph
    ppr = Namespace("http://PAonto.com#")
    associations = set(graph.subjects(RDF.type, ppr.AssemblyFeatureAssociation))
    assert len(associations) == count
    for association in associations:
        assert len(set(graph.objects(association, ppr.relatesAssemblyFeature))) == 2
        assert len(set(graph.objects(association, ppr.hasdesiredstate))) == 1
        assert not list(graph.objects(association, ppr.hascurrentstate))


def test_observation_presentation_hides_rank_and_resolves_exact_sources(tmp_path):
    from cais_spade_llm.spec2primitives.tools.observation_presentation import (
        ObservationPresentation,
    )

    root = tmp_path / "interaction"
    path = (
        root / "products/grounding/rgb_d_cad_grounding/segmentation_0001/segmentation_record.json"
    )
    path.parent.mkdir(parents=True)
    cameras = [
        {
            "observation_handle": "view_0001",
            "candidates": [
                {"candidate_handle": f"candidate_0001_{index:04d}", "centroid_m": [index, 0, 0]}
                for index in range(1, 5)
            ],
        }
    ]
    path.write_text(json.dumps({"cameras": cameras}))
    presentation = ObservationPresentation(root, create=True)
    model = presentation.project({"views": cameras})
    assert "view_0001" not in json.dumps(model)
    assert "candidate_0001_" not in json.dumps(model)
    for candidate in model["views"][0]["candidates"]:
        resolved = presentation.resolve(candidate)
        assert resolved in cameras[0]["candidates"]
    pointer = {"field_path": "/cameras/0/candidates/2/centroid_m"}
    assert presentation.resolve(presentation.project(pointer)) == pointer
    layout = {"candidate_field_paths": ["/cameras/0/candidates/1", "/cameras/0/candidates/2"]}
    assert presentation.resolve(presentation.project(layout)) == layout
    with pytest.raises(ValueError, match="not presented"):
        presentation.resolve(layout)
    permuted = copy.deepcopy(cameras)
    permuted[0]["candidates"].reverse()
    assert presentation.project({"views": permuted}) == model
    assert ObservationPresentation(root).project({"views": cameras}) == model
    other = ObservationPresentation(tmp_path / "other", create=True)
    assert other.project({"views": cameras}) != model
    with pytest.raises(ValueError, match="not presented"):
        presentation.resolve(pointer)
    with pytest.raises(ValueError, match="not presented"):
        presentation.resolve({"candidate_handle": "candidate_" + "f" * 24})
    assert json.loads(path.read_text())["cameras"] == cameras


def test_partial_endpoint_binding_is_invalid(tmp_path):
    target = _target()
    target["assembly_feature_association"] = [_association()]
    target["assembly_feature_association"][0]["assembly_features"][0]["state_name"] = (
        "current_state"
    )
    with pytest.raises(OntologyGroundingError):
        _candidate(tmp_path, target)


def test_allocation_pairs_are_exact_deduplicated_and_order_independent():
    associations = [_association(0), _association(1)]
    values = []
    for index, association in enumerate(associations):
        for role, endpoint in zip(
            ("current_state", "desired_state"), association["assembly_features"], strict=True
        ):
            name = f"{role}_{index}"
            endpoint.update(state_name=role, state_value_name=name)
            values.append(
                {
                    "state": role,
                    "name": name,
                    "value_ref": {"record_ref": "record", "field_path": f"/{role}/{index}"},
                    "record_sha256": "a" * 64,
                }
            )
    pairs = eligible_allocation_pairs(associations, values)
    assert len(pairs) == 2
    assert eligible_allocation_pairs(associations * 2, values) == pairs
    assert eligible_allocation_pairs(list(reversed(associations)), values) == tuple(reversed(pairs))


def test_source_uncertainty_preserves_selected_candidate_and_document_caveats():
    target = _target()
    target["current_state"]["state_values"] = [
        {
            "name": "candidate",
            "value_ref": {"record_ref": "seg", "field_path": "/cameras/0/candidates/1"},
            "evidence_refs": ["seg"],
        }
    ]
    target["desired_state"]["statement"]["evidence_refs"] = ["query"]
    records = {
        "query": {
            "record_type": "DocumentQueryRecord",
            "uncertainty": [{"description": "Figure inference.", "evidence_refs": ["page"]}],
        },
        "review": {
            "record_type": "ObservationCandidateReview",
            "source_segmentation": {"ref": "seg"},
            "candidates": [
                {"source_field_path": "/cameras/0/candidates/0", "uncertainty": "Unselected."},
                {
                    "source_field_path": "/cameras/0/candidates/1",
                    "uncertainty": "Appears surface-mounted; contact unclear.",
                },
            ],
        },
    }
    caveats = source_uncertainty(target, records)
    assert [item["description"] for item in caveats] == [
        "Figure inference.",
        "Appears surface-mounted; contact unclear.",
    ]


def test_endpoints_can_share_a_state_without_creating_an_allocation_pair():
    from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import (
        _validated_assembly_feature_association,
    )

    association = _association()
    resolved = []
    for index, endpoint in enumerate(association["assembly_features"]):
        name = f"observed_{index}"
        endpoint.update(state_name="current_state", state_value_name=name)
        resolved.append(
            {
                "state": "current_state",
                "name": name,
                "record_type": "RobotFrameLocationRecord",
                "record_sha256": "a" * 64,
                "value_ref": {"record_ref": str(index), "field_path": "/translated_location_m"},
            }
        )
    _validated_assembly_feature_association(
        association,
        resolved_state_values=resolved,
        authorized_evidence_refs={"requirement_0001"},
    )
    assert eligible_allocation_pairs([association], resolved) == ()


def test_owners_are_reused_only_by_exact_identity(tmp_path):
    target = _target()
    first, second = _association(0), _association(1)
    second["assembly_features"][0]["owner"]["name"] = "part 0"
    target["assembly_feature_association"] = [first, second]
    candidate, _ = _candidate(tmp_path, target)
    assert (
        len(
            set(
                candidate.provisional_abox.graph.subjects(
                    RDF.type, URIRef("http://PAonto.com#Part")
                )
            )
        )
        == 3
    )


def test_several_bound_associations_leave_allocation_unselected(tmp_path):
    from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import _validated_proposal

    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble product", tbox)
    workcell = load_predefined_workcell(tbox, load_predefined_resource_registry(tbox))
    target = _target()
    for index in range(2):
        association = _association(index)
        for role, endpoint in zip(
            ("current_state", "desired_state"), association["assembly_features"], strict=True
        ):
            name = f"location_{index}"
            endpoint.update(state_name=role, state_value_name=name)
            target[role]["state_values"].append(
                {
                    "name": name,
                    "value_ref": {
                        "record_ref": f"{role}_{index}",
                        "field_path": "/translated_location_m",
                    },
                    "evidence_refs": ["requirement_0001"],
                }
            )
        target["assembly_feature_association"].append(association)
    proposal = _validated_proposal(
        {"target_feature": target},
        abox=abox,
        workcell=workcell,
        authorized_evidence_refs={"requirement_0001"},
        typed_record_resolver=lambda ref: {
            "record_type": "RobotFrameLocationRecord",
            "record_sha256": "a" * 64,
            "record": {"translated_location_m": [0.0, 0.0, 0.0]},
        },
    )
    assert len(proposal.assembly_feature_association) == 2
    assert proposal.assembly_state_value_refs == {}


def test_converted_coordinate_binding_keeps_observation_attachment_caveat():
    target = _target()
    target["current_state"]["state_values"] = [
        {
            "name": "location",
            "value_ref": {"record_ref": "location", "field_path": "/translated_location_m"},
            "evidence_refs": ["location"],
        }
    ]
    candidate = {"observation_handle": "view_0001", "candidate_handle": "candidate_0001_0002"}
    records = {
        "location": {
            "record_type": "RobotFrameLocationRecord",
            "source_segmentation": {"ref": "seg"},
            "candidate_reference": candidate,
        },
        "review": {
            "record_type": "ObservationCandidateReview",
            "source_segmentation": {"ref": "seg"},
            "candidates": [
                {
                    **candidate,
                    "source_field_path": "/cameras/0/candidates/1",
                    "uncertainty": "Appears surface-mounted; contact unclear.",
                }
            ],
        },
    }
    assert source_uncertainty(target, records) == [
        {
            "description": "Appears surface-mounted; contact unclear.",
            "evidence_refs": ["review", "seg", "location"],
        }
    ]
