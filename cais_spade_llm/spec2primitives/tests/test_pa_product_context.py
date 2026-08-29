"""Tests for the Phase 4.0 PA-owned product context."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from rdflib import RDF, Graph, Literal, Namespace, URIRef

from cais_spade_llm.spec2primitives.agents.pa import product_context
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    TripleAssertion,
    TripleDelta,
    TripleObject,
    initialize_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.ontology import (
    OntologyContextError,
    load_ppr_tbox,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures/ontology"
MINIMAL_TBOX_PATH = FIXTURE_ROOT / "minimal_ppr_tbox.owl"
MIXED_PAONTO_PATH = FIXTURE_ROOT / "mixed_paonto_sample.owl"
PPR_NAMESPACE = "http://PAonto.com#"
PPR = Namespace(PPR_NAMESPACE)


def test_interaction_abox_preserves_only_the_exact_unresolved_requirement(
    tmp_path: Path,
) -> None:
    product_requirement = "  assemble Medium Gear exactly  \n"
    abox = _initialize(tmp_path, product_requirement=product_requirement)
    ontology_root = tmp_path / "products/grounding/ontology"

    assert abox.ontology_root == ontology_root
    assert abox.abox_path == ontology_root / "interaction_abox.ttl"
    assert abox.manifest_path == ontology_root / "abox_manifest.json"
    assert abox.provenance_path == ontology_root / "assertion_provenance.json"
    assert abox.abox_path.is_file()
    assert abox.manifest_path.is_file()
    assert abox.provenance_path.is_file()
    assert not list(ontology_root.glob("delta_*.json"))

    persisted_graph = Graph().parse(abox.abox_path, format="turtle")
    specification_iri = URIRef(str(abox.specification_iri))
    assert set(persisted_graph) == {
        (specification_iri, RDF.type, PPR.specification),
        (specification_iri, RDF.value, Literal(product_requirement)),
    }
    for ungrounded_type in (
        PPR.product,
        PPR.feature,
        PPR.process,
        PPR.resource,
        PPR.capability,
    ):
        assert not set(persisted_graph.subjects(RDF.type, ungrounded_type))

    manifest = _read_json(abox.manifest_path)
    assert manifest["status"] == "unresolved"
    assert manifest["product_requirement"] == product_requirement
    assert manifest["interaction_namespace"] == str(abox.namespace)
    assert manifest["specification_iri"] == str(specification_iri)
    assert manifest["tbox_fingerprint"] == abox.tbox_fingerprint
    assert manifest["delta_count"] == 0
    assert manifest["accepted_assertion_count"] == 0

    provenance = _read_json(abox.provenance_path)
    assert len(provenance["assertions"]) == 2
    assert all(
        record["producer"] == "interaction_initializer"
        and record["evidence_refs"] == ["products/user_requirement/product_requirement.json"]
        and record["delta_ref"] is None
        for record in provenance["assertions"]
    )


def test_rejected_mixed_paonto_facts_do_not_seed_product_context(
    tmp_path: Path,
) -> None:
    with pytest.raises(OntologyContextError):
        load_ppr_tbox(MIXED_PAONTO_PATH, ppr_namespace=PPR_NAMESPACE)

    abox = _initialize(tmp_path, product_requirement="assemble Medium Gear")
    for forbidden_node in (PPR.PD1, PPR.Fd1, PPR.D1, PPR.specA1):
        assert not any(forbidden_node in triple for triple in abox.graph)


@pytest.mark.parametrize("product_requirement", ["", " \t\n"])
def test_empty_requirement_does_not_initialize_an_abox(
    tmp_path: Path,
    product_requirement: str,
) -> None:
    with pytest.raises(OntologyContextError):
        initialize_interaction_abox(tmp_path, product_requirement, _load_tbox())

    assert not tmp_path.exists() or not any(tmp_path.rglob("*"))


def test_interactions_receive_independent_namespaces_and_aboxes(tmp_path: Path) -> None:
    first = _initialize(tmp_path / "first", product_requirement="requirement one")
    second = _initialize(tmp_path / "second", product_requirement="requirement two")

    assert first.namespace != second.namespace
    assert first.specification_iri != second.specification_iri
    assert first.abox_path != second.abox_path
    assert (
        URIRef(first.specification_iri),
        RDF.value,
        Literal("requirement one"),
    ) in first.graph
    assert (
        URIRef(second.specification_iri),
        RDF.value,
        Literal("requirement two"),
    ) in second.graph
    assert not any(URIRef(second.specification_iri) in triple for triple in first.graph)
    assert not any(URIRef(first.specification_iri) in triple for triple in second.graph)


def test_existing_interaction_abox_is_not_overwritten(tmp_path: Path) -> None:
    abox = _initialize(tmp_path, product_requirement="original requirement")
    before = _ontology_file_bytes(abox)

    with pytest.raises(OntologyContextError):
        initialize_interaction_abox(
            tmp_path,
            "replacement requirement",
            _load_tbox(),
        )

    assert _ontology_file_bytes(abox) == before


def test_malformed_delta_dataclass_is_rejected_atomically(tmp_path: Path) -> None:
    tbox = _load_tbox()
    abox = initialize_interaction_abox(tmp_path, "dataclass requirement", tbox)
    evidence_ref = "approved_evidence"
    delta = TripleDelta(
        assertions=(
            TripleAssertion(
                subject=f"{abox.namespace}candidate_feature",
                predicate=str(RDF.type),
                object=TripleObject(kind="unsupported", value=str(PPR.feature)),
                evidence_refs=(evidence_ref,),
            ),
        ),
        uncertainty=(),
        unresolved_evidence_needs=(),
        typed_context_refs=(),
    )
    before = _ontology_file_bytes(abox)

    with pytest.raises(OntologyContextError):
        validate_and_merge_triple_delta(
            tmp_path,
            tbox,
            "controlled_evidence",
            delta,
            authorized_evidence_refs=[evidence_ref],
        )

    assert _ontology_file_bytes(abox) == before


def test_document_and_scene_deltas_use_one_validator_and_retain_provenance(
    tmp_path: Path,
) -> None:
    tbox = _load_tbox()
    abox = initialize_interaction_abox(
        tmp_path,
        "assemble Medium Gear",
        tbox,
    )
    namespace = str(abox.namespace)
    required_feature = URIRef(f"{namespace}required_feature_7")
    requested_process = URIRef(f"{namespace}requested_process_3")
    observed_product = URIRef(f"{namespace}observed_product_9")
    document_ref = "NIST_assembly_instructions.pdf#page=4"
    observation_ref = "observation_0001"

    document_result = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "document_evidence",
        {
            "assertions": [
                _iri_assertion(required_feature, RDF.type, PPR.feature, document_ref),
                _iri_assertion(requested_process, RDF.type, PPR.process, document_ref),
                _iri_assertion(abox.specification_iri, PPR.defines, required_feature, document_ref),
                _iri_assertion(requested_process, PPR.realizes, required_feature, document_ref),
            ],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": ["products/grounding/document_page_4.json"],
        },
        authorized_evidence_refs=[
            document_ref,
        ],
    )
    scene_result = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "rgb_d_cad_grounding",
        {
            "assertions": [
                _iri_assertion(observed_product, RDF.type, PPR.product, observation_ref)
            ],
            "uncertainty": [{"candidate": "unresolved"}],
            "unresolved_evidence_needs": ["receiving geometry"],
            "typed_context_refs": ["products/grounding/pose_record_1.json"],
        },
        authorized_evidence_refs=[
            observation_ref,
        ],
    )

    assert document_result.accepted is True
    assert document_result.assertion_count == 4
    assert document_result.delta_path.name == "delta_0001.json"
    assert scene_result.accepted is True
    assert scene_result.assertion_count == 1
    assert scene_result.delta_path.name == "delta_0002.json"
    graph = scene_result.abox.graph
    assert (required_feature, RDF.type, PPR.feature) in graph
    assert (requested_process, RDF.type, PPR.process) in graph
    assert (URIRef(abox.specification_iri), PPR.defines, required_feature) in graph
    assert (requested_process, PPR.realizes, required_feature) in graph
    assert (observed_product, RDF.type, PPR.product) in graph

    provenance = _read_json(abox.provenance_path)["assertions"]
    document_records = [
        record for record in provenance if record["producer"] == "document_evidence"
    ]
    scene_records = [record for record in provenance if record["producer"] == "rgb_d_cad_grounding"]
    assert len(document_records) == 4
    assert all(record["evidence_refs"] == [document_ref] for record in document_records)
    assert all(record["delta_ref"] == "delta_0001.json" for record in document_records)
    assert len(scene_records) == 1
    assert scene_records[0]["evidence_refs"] == [observation_ref]
    assert scene_records[0]["delta_ref"] == "delta_0002.json"

    scene_delta = _read_json(scene_result.delta_path)
    assert scene_delta["uncertainty"] == [{"candidate": "unresolved"}]
    assert scene_delta["unresolved_evidence_needs"] == ["receiving geometry"]
    assert scene_delta["typed_context_refs"] == ["products/grounding/pose_record_1.json"]
    serialized_graph = abox.abox_path.read_text(encoding="utf-8")
    assert "pose_record_1.json" not in serialized_graph
    assert "observation_0001" not in serialized_graph

    manifest = _read_json(abox.manifest_path)
    assert manifest["status"] == "unresolved"
    assert manifest["delta_count"] == 2
    assert manifest["accepted_assertion_count"] == 5


@pytest.mark.parametrize(
    "source_order",
    [
        ("document",),
        ("scene", "document"),
        ("cad", "diagram", "scene", "document", "observation"),
    ],
)
def test_one_two_and_many_dynamic_deltas_merge_in_source_selected_order(
    tmp_path: Path,
    source_order: tuple[str, ...],
) -> None:
    tbox = _load_tbox()
    abox = initialize_interaction_abox(tmp_path, "dynamic requirement", tbox)
    producer = "controlled_evidence"
    subjects: list[URIRef] = []

    assert not any("dynamic_feature" in str(node) for triple in abox.graph for node in triple)
    for delta_number, source in enumerate(source_order, start=1):
        subject = URIRef(f"{abox.namespace}dynamic_feature_{source}_{delta_number}")
        evidence_ref = f"{source}_evidence_{delta_number}"
        subjects.append(subject)
        result = validate_and_merge_triple_delta(
            tmp_path,
            tbox,
            producer,
            {"assertions": [_iri_assertion(subject, RDF.type, PPR.feature, evidence_ref)]},
            authorized_evidence_refs=[evidence_ref],
        )
        assert result.delta_path.name == f"delta_{delta_number:04d}.json"

    persisted_graph = Graph().parse(abox.abox_path, format="turtle")
    assert all((subject, RDF.type, PPR.feature) in persisted_graph for subject in subjects)
    assert _read_json(abox.manifest_path)["delta_count"] == len(source_order)


def test_uncertain_delta_with_no_assertions_stays_outside_the_abox(
    tmp_path: Path,
) -> None:
    tbox = _load_tbox()
    abox = initialize_interaction_abox(tmp_path, "uncertain requirement", tbox)
    before_graph = set(abox.graph)
    context_ref = "products/grounding/uncertain_match.json"

    result = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "rgb_d_cad_grounding",
        {
            "assertions": [],
            "uncertainty": ["candidate match below threshold"],
            "unresolved_evidence_needs": ["another observation"],
            "typed_context_refs": [context_ref],
        },
        authorized_evidence_refs=[],
    )

    assert result.accepted is True
    assert result.assertion_count == 0
    assert set(result.abox.graph) == before_graph
    assert _read_json(result.delta_path)["typed_context_refs"] == [context_ref]
    assert context_ref not in abox.abox_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "invalid_case",
    [
        "unknown_predicate",
        "missing_evidence",
        "unauthorized_evidence",
        "malformed_iri_object",
        "unknown_class",
        "external_subject",
    ],
)
def test_invalid_delta_is_rejected_atomically(
    tmp_path: Path,
    invalid_case: str,
) -> None:
    tbox = _load_tbox()
    abox = initialize_interaction_abox(tmp_path, "atomic requirement", tbox)
    evidence_ref = "approved_evidence"
    subject = URIRef(f"{abox.namespace}candidate_feature")
    producer = "controlled_evidence"
    assertion = _iri_assertion(subject, RDF.type, PPR.feature, evidence_ref)
    authorized_refs = [evidence_ref]

    if invalid_case == "unknown_predicate":
        assertion["predicate"] = "http://unknown.example/predicate"
    elif invalid_case == "missing_evidence":
        assertion["evidence_refs"] = []
    elif invalid_case == "unauthorized_evidence":
        assertion["evidence_refs"] = ["unapproved_evidence"]
    elif invalid_case == "malformed_iri_object":
        assertion["object"] = {"kind": "iri", "value": "not-an-absolute-iri"}
    elif invalid_case == "unknown_class":
        assertion["object"] = {
            "kind": "iri",
            "value": "http://unknown.example/Class",
        }
    elif invalid_case == "external_subject":
        assertion["subject"] = "http://external.example/subject"

    before = _ontology_file_bytes(abox)
    with pytest.raises(OntologyContextError):
        validate_and_merge_triple_delta(
            tmp_path,
            tbox,
            producer,
            {"assertions": [assertion]},
            authorized_evidence_refs=authorized_refs,
        )

    assert _ontology_file_bytes(abox) == before
    assert not list(abox.ontology_root.glob("delta_*.json"))


@pytest.mark.parametrize(
    ("predicate", "object_iri"),
    [
        (RDF.type, PPR.resource),
        (RDF.type, PPR.capability),
        (PPR.capableOf, PPR.process),
        (PPR.provides, PPR.capability),
    ],
)
def test_resource_catalog_assertion_is_rejected_from_the_pa_abox(
    tmp_path: Path,
    predicate: URIRef,
    object_iri: URIRef,
) -> None:
    tbox = _load_tbox()
    abox = initialize_interaction_abox(tmp_path, "resource-free requirement", tbox)
    evidence_ref = "resource_card"
    assertion = _iri_assertion(
        URIRef(f"{abox.namespace}forbidden_resource_fact"),
        predicate,
        object_iri,
        evidence_ref,
    )

    before = _ontology_file_bytes(abox)
    with pytest.raises(OntologyContextError):
        validate_and_merge_triple_delta(
            tmp_path,
            tbox,
            "controlled_evidence",
            {"assertions": [assertion]},
            authorized_evidence_refs=[evidence_ref],
        )

    assert _ontology_file_bytes(abox) == before
    assert not set(
        Graph().parse(abox.abox_path, format="turtle").triples((None, predicate, object_iri))
    )


def test_pa_authored_execution_assignment_is_rejected_atomically(
    tmp_path: Path,
) -> None:
    tbox = _load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    evidence_ref = "pa_proposal"
    execution = URIRef(f"{abox.namespace}process_execution_0001")
    before = _ontology_file_bytes(abox)

    with pytest.raises(OntologyContextError):
        validate_and_merge_triple_delta(
            tmp_path,
            tbox,
            "ontology_grounding",
            {
                "assertions": [
                    _iri_assertion(
                        URIRef(abox.specification_iri),
                        PPR.hasProcessExecution,
                        execution,
                        evidence_ref,
                    ),
                    _iri_assertion(
                        execution,
                        RDF.type,
                        PPR.processExecution,
                        evidence_ref,
                    ),
                    _iri_assertion(
                        execution,
                        PPR.runsProcess,
                        URIRef("https://cais-spade-llm.local/process/assembly"),
                        evidence_ref,
                    ),
                    _iri_assertion(
                        execution,
                        PPR.runsOnResource,
                        URIRef("https://cais-spade-llm.local/resource/xarm6"),
                        evidence_ref,
                    ),
                ]
            },
            authorized_evidence_refs=[evidence_ref],
        )

    assert _ontology_file_bytes(abox) == before
    assert not list(abox.ontology_root.glob("delta_*.json"))


@pytest.mark.parametrize("recipe_predicate", [PPR.requires, PPR.precedes])
def test_task_to_primitive_recipe_assertion_is_rejected_from_the_pa_abox(
    tmp_path: Path,
    recipe_predicate: URIRef,
) -> None:
    tbox = _load_tbox()
    abox = initialize_interaction_abox(tmp_path, "recipe-free requirement", tbox)
    evidence_ref = "resource_catalog"
    primitive = URIRef(f"{abox.namespace}primitive_move_arm_xyz")
    requested_process = URIRef(f"{abox.namespace}requested_process")
    before = _ontology_file_bytes(abox)
    with pytest.raises(OntologyContextError):
        validate_and_merge_triple_delta(
            tmp_path,
            tbox,
            "controlled_evidence",
            {
                "assertions": [
                    _iri_assertion(
                        requested_process,
                        recipe_predicate,
                        primitive,
                        evidence_ref,
                    )
                ]
            },
            authorized_evidence_refs=[evidence_ref],
        )

    assert _ontology_file_bytes(abox) == before
    assert not set(
        Graph().parse(abox.abox_path, format="turtle").triples((None, recipe_predicate, None))
    )


def test_declared_consists_of_cannot_decompose_a_process(tmp_path: Path) -> None:
    source = MINIMAL_TBOX_PATH.read_text(encoding="utf-8")
    tbox_path = tmp_path / "process_structure_tbox.owl"
    tbox_path.write_text(
        source.replace(
            "</rdf:RDF>",
            '  <owl:ObjectProperty rdf:about="#consistsOf"/>\n\n</rdf:RDF>',
        ),
        encoding="utf-8",
    )
    tbox = load_ppr_tbox(tbox_path, ppr_namespace=PPR_NAMESPACE)
    interaction_root = tmp_path / "interaction"
    abox = initialize_interaction_abox(interaction_root, "compose process", tbox)
    requested_process = URIRef(f"{abox.namespace}requested_process")
    component_process = URIRef(f"{abox.namespace}component_process")
    evidence_ref = "approved_document"
    before = _ontology_file_bytes(abox)

    with pytest.raises(OntologyContextError):
        validate_and_merge_triple_delta(
            interaction_root,
            tbox,
            "controlled_evidence",
            {
                "assertions": [
                    _iri_assertion(
                        requested_process,
                        RDF.type,
                        PPR.process,
                        evidence_ref,
                    ),
                    _iri_assertion(
                        component_process,
                        RDF.type,
                        PPR.process,
                        evidence_ref,
                    ),
                    _iri_assertion(
                        requested_process,
                        PPR.consistsOf,
                        component_process,
                        evidence_ref,
                    ),
                ]
            },
            authorized_evidence_refs=[evidence_ref],
        )

    assert _ontology_file_bytes(abox) == before


def test_realizes_must_target_the_feature_defined_by_the_specification(
    tmp_path: Path,
) -> None:
    tbox = _load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble product", tbox)
    namespace = str(abox.namespace)
    defined_feature = URIRef(f"{namespace}defined_feature")
    other_feature = URIRef(f"{namespace}other_feature")
    process = URIRef(f"{namespace}requested_process")
    evidence_ref = "approved_document"
    producer = "controlled_evidence"
    validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        producer,
        {
            "assertions": [
                _iri_assertion(defined_feature, RDF.type, PPR.feature, evidence_ref),
                _iri_assertion(
                    abox.specification_iri,
                    PPR.defines,
                    defined_feature,
                    evidence_ref,
                ),
            ]
        },
        authorized_evidence_refs=[evidence_ref],
    )
    before = _ontology_file_bytes(abox)

    with pytest.raises(OntologyContextError):
        validate_and_merge_triple_delta(
            tmp_path,
            tbox,
            producer,
            {
                "assertions": [
                    _iri_assertion(other_feature, RDF.type, PPR.feature, evidence_ref),
                    _iri_assertion(process, RDF.type, PPR.process, evidence_ref),
                    _iri_assertion(
                        process,
                        PPR.realizes,
                        other_feature,
                        evidence_ref,
                    ),
                ]
            },
            authorized_evidence_refs=[evidence_ref],
        )

    assert _ontology_file_bytes(abox) == before
    persisted_graph = Graph().parse(abox.abox_path, format="turtle")
    assert (process, PPR.realizes, other_feature) not in persisted_graph


def test_defines_must_start_at_the_initialized_specification(tmp_path: Path) -> None:
    tbox = _load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble product", tbox)
    namespace = str(abox.namespace)
    feature = URIRef(f"{namespace}required_feature")
    wrong_subject = URIRef(f"{namespace}not_the_specification")
    evidence_ref = "approved_document"
    before = _ontology_file_bytes(abox)

    with pytest.raises(OntologyContextError):
        validate_and_merge_triple_delta(
            tmp_path,
            tbox,
            "controlled_evidence",
            {
                "assertions": [
                    _iri_assertion(feature, RDF.type, PPR.feature, evidence_ref),
                    _iri_assertion(
                        wrong_subject,
                        RDF.type,
                        PPR.product,
                        evidence_ref,
                    ),
                    _iri_assertion(
                        wrong_subject,
                        PPR.defines,
                        feature,
                        evidence_ref,
                    ),
                ]
            },
            authorized_evidence_refs=[evidence_ref],
        )

    assert _ontology_file_bytes(abox) == before


def test_hidden_expected_answer_ref_is_never_authorized_by_the_delta(
    tmp_path: Path,
) -> None:
    tbox = _load_tbox()
    abox = initialize_interaction_abox(tmp_path, "ground this requirement", tbox)
    feature = URIRef(f"{abox.namespace}candidate_feature")
    hidden_ref = "evaluations/ground_truth/hidden_expected_answer.json"
    assertion = _iri_assertion(feature, RDF.type, PPR.feature, hidden_ref)
    before = _ontology_file_bytes(abox)

    with pytest.raises(OntologyContextError):
        validate_and_merge_triple_delta(
            tmp_path,
            tbox,
            "controlled_evidence",
            {"assertions": [assertion]},
            authorized_evidence_refs=["approved_document"],
        )

    assert _ontology_file_bytes(abox) == before
    assert hidden_ref not in abox.abox_path.read_text(encoding="utf-8")
    assert hidden_ref not in abox.provenance_path.read_text(encoding="utf-8")
    assert not list(abox.ontology_root.glob("delta_*.json"))


def test_merges_do_not_modify_the_immutable_tbox_fixture(tmp_path: Path) -> None:
    before = MINIMAL_TBOX_PATH.read_bytes()
    tbox = _load_tbox()
    abox = initialize_interaction_abox(tmp_path, "immutable schema", tbox)
    evidence_ref = "approved_document"
    feature = URIRef(f"{abox.namespace}runtime_feature")

    validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "controlled_evidence",
        {"assertions": [_iri_assertion(feature, RDF.type, PPR.feature, evidence_ref)]},
        authorized_evidence_refs=[evidence_ref],
    )

    assert MINIMAL_TBOX_PATH.read_bytes() == before


def test_product_context_has_no_ra_dependency() -> None:
    imported_modules = _imported_modules(product_context)

    assert not any("agents.ra" in name for name in imported_modules)


def _load_tbox() -> Any:
    return load_ppr_tbox(MINIMAL_TBOX_PATH, ppr_namespace=PPR_NAMESPACE)


def _initialize(
    interaction_root: Path,
    *,
    product_requirement: str,
) -> Any:
    return initialize_interaction_abox(
        interaction_root,
        product_requirement,
        _load_tbox(),
    )


def _iri_assertion(
    subject: URIRef | str,
    predicate: URIRef,
    object_iri: URIRef,
    evidence_ref: str,
) -> dict[str, object]:
    return {
        "subject": str(subject),
        "predicate": str(predicate),
        "object": {"kind": "iri", "value": str(object_iri)},
        "evidence_refs": [evidence_ref],
    }


def _ontology_file_bytes(abox: Any) -> dict[str, bytes]:
    return {
        "abox": abox.abox_path.read_bytes(),
        "manifest": abox.manifest_path.read_bytes(),
        "provenance": abox.provenance_path.read_bytes(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _imported_modules(module: ModuleType) -> set[str]:
    source_path = Path(module.__file__ or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    return imported_modules
