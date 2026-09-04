"""Tests for shared schema-only PPR TBox handling."""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

import pytest
from rdflib import OWL, RDF, RDFS, Graph, Literal, Namespace, URIRef

from cais_spade_llm.spec2primitives.ontology import (
    OntologyContextError,
    TBoxSnapshot,
    load_ppr_tbox,
    ppr_tbox,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures/ontology"
MINIMAL_TBOX_PATH = FIXTURE_ROOT / "minimal_ppr_tbox.owl"
MIXED_PAONTO_PATH = FIXTURE_ROOT / "mixed_paonto_sample.owl"
PROJECT_TBOX_PATH = Path(__file__).parents[1] / "ontology/spec2primitives_ppr_tbox.owl"
PPR_NAMESPACE = "http://PAonto.com#"
PPR = Namespace(PPR_NAMESPACE)


def test_project_tbox_is_the_exact_schema_only_spec2primitives_vocabulary() -> None:
    tbox = load_ppr_tbox(PROJECT_TBOX_PATH, ppr_namespace=PPR_NAMESPACE)
    expected_classes = {
        PPR.specification,
        PPR.product,
        PPR.feature,
        PPR.state,
        PPR.process,
        PPR.processExecution,
        PPR.resource,
        PPR.capability,
        PPR.Assembly,
        PPR.Part,
        PPR.AssemblyFeature,
        PPR.AssemblyFeatureAssociation,
    }
    expected_properties = {
        PPR.defines,
        PPR.realizes,
        PPR.hascurrentstate,
        PPR.hasdesiredstate,
        PPR.capableOf,
        PPR.hasProcessExecution,
        PPR.runsProcess,
        PPR.runsOnResource,
        PPR.hasPart,
        PPR.hasAssemblyFeature,
        PPR.hasAssemblyFeatureAssociation,
        PPR.relatesAssemblyFeature,
    }

    assert set(tbox.classes) == {str(value) for value in expected_classes}
    assert set(tbox.object_properties) == {str(value) for value in expected_properties}
    assert tbox.datatype_properties == frozenset()
    assert set(tbox.graph.objects(PPR.defines, RDFS.domain)) == {PPR.specification}
    assert set(tbox.graph.objects(PPR.defines, RDFS.range)) == {PPR.feature}
    assert set(tbox.graph.objects(PPR.realizes, RDFS.domain)) == {PPR.process}
    assert set(tbox.graph.objects(PPR.realizes, RDFS.range)) == {PPR.feature}
    for property_iri in (PPR.hascurrentstate, PPR.hasdesiredstate):
        assert set(tbox.graph.objects(property_iri, RDFS.domain)) == {PPR.feature}
        assert set(tbox.graph.objects(property_iri, RDFS.range)) == {PPR.state}
        assert (property_iri, RDF.type, OWL.FunctionalProperty) in tbox.graph
    assert set(tbox.graph.objects(PPR.capableOf, RDFS.domain)) == {PPR.resource}
    assert set(tbox.graph.objects(PPR.capableOf, RDFS.range)) == {PPR.process}
    assert set(tbox.graph.objects(PPR.hasProcessExecution, RDFS.domain)) == {PPR.specification}
    assert set(tbox.graph.objects(PPR.hasProcessExecution, RDFS.range)) == {PPR.processExecution}
    assert set(tbox.graph.objects(PPR.runsProcess, RDFS.domain)) == {PPR.processExecution}
    assert set(tbox.graph.objects(PPR.runsProcess, RDFS.range)) == {PPR.process}
    assert set(tbox.graph.objects(PPR.runsOnResource, RDFS.domain)) == {PPR.processExecution}
    assert set(tbox.graph.objects(PPR.runsOnResource, RDFS.range)) == {PPR.resource}
    assert (PPR.runsProcess, RDF.type, OWL.FunctionalProperty) in tbox.graph
    assert (PPR.runsOnResource, RDF.type, OWL.FunctionalProperty) in tbox.graph
    assert (PPR.hasProcessExecution, RDF.type, OWL.FunctionalProperty) not in tbox.graph
    assert set(tbox.graph.objects(PPR.Assembly, RDFS.subClassOf)) >= {PPR.product}
    assert set(tbox.graph.objects(PPR.Part, RDFS.subClassOf)) == {PPR.product}
    assert set(tbox.graph.objects(PPR.AssemblyFeature, RDFS.subClassOf)) == {PPR.feature}
    assert PPR.feature in set(
        tbox.graph.objects(PPR.AssemblyFeatureAssociation, RDFS.subClassOf)
    )
    assembly_properties = {
        PPR.hasPart: (PPR.Assembly, PPR.product),
        PPR.hasAssemblyFeature: (PPR.product, PPR.AssemblyFeature),
        PPR.hasAssemblyFeatureAssociation: (
            PPR.Assembly,
            PPR.AssemblyFeatureAssociation,
        ),
        PPR.relatesAssemblyFeature: (
            PPR.AssemblyFeatureAssociation,
            PPR.AssemblyFeature,
        ),
    }
    for property_iri, (domain, range_) in assembly_properties.items():
        assert set(tbox.graph.objects(property_iri, RDFS.domain)) == {domain}
        assert set(tbox.graph.objects(property_iri, RDFS.range)) == {range_}
        assert (property_iri, RDF.type, OWL.FunctionalProperty) not in tbox.graph
    assert not any(tbox.graph.subjects(RDF.type, OWL.NamedIndividual))
    restrictions = list(tbox.graph.subjects(RDF.type, OWL.Restriction))
    assert len(restrictions) == 1
    restriction = restrictions[0]
    assert (restriction, OWL.onProperty, PPR.relatesAssemblyFeature) in tbox.graph
    assert (restriction, OWL.onClass, PPR.AssemblyFeature) in tbox.graph
    cardinalities = list(tbox.graph.objects(restriction, OWL.qualifiedCardinality))
    assert len(cardinalities) == 1
    assert cardinalities[0].toPython() == 2
    for recipe_property in (PPR.requires, PPR.precedes, PPR.consistsOf):
        assert not any(tbox.graph.triples((None, recipe_property, None)))


def test_schema_only_tbox_loads_with_exact_required_vocabulary() -> None:
    tbox = _load_tbox()
    required_classes = {
        PPR.specification,
        PPR.product,
        PPR.feature,
        PPR.state,
        PPR.process,
        PPR.processExecution,
        PPR.resource,
        PPR.capability,
    }
    required_properties = {
        PPR.defines,
        PPR.realizes,
        PPR.hascurrentstate,
        PPR.hasdesiredstate,
        PPR.capableOf,
        PPR.hasProcessExecution,
        PPR.runsProcess,
        PPR.runsOnResource,
    }

    assert isinstance(tbox, TBoxSnapshot)
    assert {str(value) for value in required_classes} <= set(tbox.classes)
    assert {str(value) for value in required_properties} <= set(tbox.object_properties)
    assert tbox.ppr_namespace == PPR_NAMESPACE
    assert tbox.source_path == MINIMAL_TBOX_PATH.resolve()
    assert len(tbox.fingerprint) == 64
    assert tbox.fingerprint == _load_tbox().fingerprint


def test_tbox_snapshot_supports_shared_immutability_and_class_queries() -> None:
    tbox = _load_tbox()

    tbox.assert_unchanged()
    assert tbox.is_class_or_subclass(PPR.feature, PPR.feature)
    assert not tbox.is_class_or_subclass(PPR.product, PPR.feature)

    tbox.graph.add((PPR.feature, RDFS.label, Literal("changed after validation")))
    with pytest.raises(OntologyContextError):
        tbox.assert_unchanged()
    with pytest.raises(OntologyContextError):
        tbox.is_class_or_subclass(PPR.feature, PPR.feature)


def test_tbox_fingerprint_changes_when_the_schema_changes(tmp_path: Path) -> None:
    source = MINIMAL_TBOX_PATH.read_text(encoding="utf-8")
    changed_path = tmp_path / "changed.owl"
    changed_path.write_text(
        source.replace(
            "</rdf:RDF>",
            '  <owl:Class rdf:about="#additionalSchemaClass"/>\n\n</rdf:RDF>',
        ),
        encoding="utf-8",
    )

    assert (
        load_ppr_tbox(
            changed_path,
            ppr_namespace=PPR_NAMESPACE,
        ).fingerprint
        != _load_tbox().fingerprint
    )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            '<rdfs:domain rdf:resource="#specification"/>',
            '<rdfs:domain rdf:resource="#feature"/>',
            "invalid domain",
        ),
        (
            '<rdfs:range rdf:resource="#process"/>',
            '<rdfs:range rdf:resource="#feature"/>',
            "invalid range",
        ),
        (
            '    <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#FunctionalProperty"/>\n',
            "",
            "invalid functional profile",
        ),
    ],
)
def test_process_execution_slice_fails_closed_on_profile_changes(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    source = MINIMAL_TBOX_PATH.read_text(encoding="utf-8")
    assert old in source
    changed_path = tmp_path / "changed_execution_slice.owl"
    changed_path.write_text(source.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(OntologyContextError, match=message):
        load_ppr_tbox(changed_path, ppr_namespace=PPR_NAMESPACE)


def test_malformed_tbox_is_rejected(tmp_path: Path) -> None:
    malformed_path = tmp_path / "malformed.owl"
    malformed_path.write_text("<rdf:RDF>", encoding="utf-8")

    with pytest.raises(OntologyContextError):
        load_ppr_tbox(malformed_path, ppr_namespace=PPR_NAMESPACE)


@pytest.mark.parametrize(
    ("kind", "symbol"),
    [
        ("Class", "specification"),
        ("Class", "product"),
        ("Class", "feature"),
        ("Class", "state"),
        ("Class", "process"),
        ("Class", "resource"),
        ("Class", "capability"),
        ("ObjectProperty", "defines"),
        ("ObjectProperty", "realizes"),
        ("ObjectProperty", "capableOf"),
    ],
)
def test_tbox_missing_a_required_symbol_is_rejected(
    tmp_path: Path,
    kind: str,
    symbol: str,
) -> None:
    marker = f'  <owl:{kind} rdf:about="#{symbol}"/>\n'
    source = MINIMAL_TBOX_PATH.read_text(encoding="utf-8")
    assert marker in source
    incomplete_path = tmp_path / f"missing_{symbol}.owl"
    incomplete_path.write_text(source.replace(marker, ""), encoding="utf-8")

    with pytest.raises(OntologyContextError):
        load_ppr_tbox(incomplete_path, ppr_namespace=PPR_NAMESPACE)


def test_mixed_paonto_fixture_parses_but_fails_the_production_profile() -> None:
    mixed_graph = Graph().parse(MIXED_PAONTO_PATH)
    assert (PPR.PD1, RDF.type, OWL.NamedIndividual) in mixed_graph
    assert (PPR.PD1, PPR.realizes, PPR.Fd1) in mixed_graph
    assert (PPR.D1, PPR.capableOf, PPR.PD1) in mixed_graph
    assert (PPR.specA1, PPR.defines, PPR.Fd1) in mixed_graph
    assert set(mixed_graph.objects(None, OWL.onProperty)) == {
        PPR.requires,
        PPR.fprecedes,
        PPR.fconflicts,
        PPR.frequires,
    }

    with pytest.raises(OntologyContextError):
        load_ppr_tbox(MIXED_PAONTO_PATH, ppr_namespace=PPR_NAMESPACE)


@pytest.mark.parametrize("recipe_property", ["requires", "precedes"])
def test_schema_only_task_recipe_restriction_is_rejected(
    tmp_path: Path,
    recipe_property: str,
) -> None:
    source = MINIMAL_TBOX_PATH.read_text(encoding="utf-8")
    restriction = f"""
  <owl:ObjectProperty rdf:about="#{recipe_property}"/>
  <owl:Class rdf:about="#recipeProcess">
    <rdfs:subClassOf>
      <owl:Restriction>
        <owl:onProperty rdf:resource="#{recipe_property}"/>
        <owl:someValuesFrom rdf:resource="#process"/>
      </owl:Restriction>
    </rdfs:subClassOf>
    <rdfs:subClassOf rdf:resource="#process"/>
  </owl:Class>
"""
    recipe_path = tmp_path / f"{recipe_property}_recipe.owl"
    recipe_path.write_text(
        source.replace("</rdf:RDF>", f"{restriction}\n</rdf:RDF>"),
        encoding="utf-8",
    )

    with pytest.raises(OntologyContextError):
        load_ppr_tbox(recipe_path, ppr_namespace=PPR_NAMESPACE)


def test_implicit_instance_assertion_is_rejected(tmp_path: Path) -> None:
    source = MINIMAL_TBOX_PATH.read_text(encoding="utf-8")
    instance_path = tmp_path / "implicit_instance.owl"
    instance_path.write_text(
        source.replace(
            "</rdf:RDF>",
            '  <feature rdf:about="#runtimeFeature"/>\n\n</rdf:RDF>',
        ),
        encoding="utf-8",
    )

    with pytest.raises(OntologyContextError):
        load_ppr_tbox(instance_path, ppr_namespace=PPR_NAMESPACE)


def test_blank_node_rdf_value_assertion_is_rejected(tmp_path: Path) -> None:
    source = MINIMAL_TBOX_PATH.read_text(encoding="utf-8")
    instance_path = tmp_path / "blank_node_instance.owl"
    instance_path.write_text(
        source.replace(
            "</rdf:RDF>",
            "  <rdf:Description><rdf:value>runtime fact</rdf:value>"
            "</rdf:Description>\n\n</rdf:RDF>",
        ),
        encoding="utf-8",
    )

    with pytest.raises(OntologyContextError):
        load_ppr_tbox(instance_path, ppr_namespace=PPR_NAMESPACE)


def test_literal_has_value_restriction_remains_valid_schema(tmp_path: Path) -> None:
    source = MINIMAL_TBOX_PATH.read_text(encoding="utf-8")
    schema_path = tmp_path / "literal_has_value.owl"
    restriction = """
  <owl:DatatypeProperty rdf:about="#mode"/>
  <owl:Class rdf:about="#modeFeature">
    <rdfs:subClassOf rdf:resource="#feature"/>
    <rdfs:subClassOf>
      <owl:Restriction>
        <owl:onProperty rdf:resource="#mode"/>
        <owl:hasValue>manual</owl:hasValue>
      </owl:Restriction>
    </rdfs:subClassOf>
  </owl:Class>
"""
    schema_path.write_text(
        source.replace("</rdf:RDF>", f"{restriction}\n</rdf:RDF>"),
        encoding="utf-8",
    )

    tbox = load_ppr_tbox(schema_path, ppr_namespace=PPR_NAMESPACE)

    assert str(PPR.mode) in tbox.datatype_properties
    assert str(PPR.modeFeature) in tbox.classes


def test_version_iri_remains_valid_schema_metadata(tmp_path: Path) -> None:
    source = MINIMAL_TBOX_PATH.read_text(encoding="utf-8")
    schema_path = tmp_path / "versioned.owl"
    schema_path.write_text(
        source.replace(
            '<owl:Ontology rdf:about="http://PAonto.com"/>',
            '<owl:Ontology rdf:about="http://PAonto.com">\n'
            '  <owl:versionIRI rdf:resource="http://PAonto.com#version1"/>\n'
            "</owl:Ontology>",
        ),
        encoding="utf-8",
    )

    tbox = load_ppr_tbox(schema_path, ppr_namespace=PPR_NAMESPACE)

    assert (
        URIRef("http://PAonto.com"),
        OWL.versionIRI,
        URIRef("http://PAonto.com#version1"),
    ) in tbox.graph


def test_shared_ontology_has_no_pa_or_ra_dependency() -> None:
    imported_modules = _imported_modules(ppr_tbox)

    assert not any("agents.pa" in name or "agents.ra" in name for name in imported_modules)


def _load_tbox() -> TBoxSnapshot:
    return load_ppr_tbox(MINIMAL_TBOX_PATH, ppr_namespace=PPR_NAMESPACE)


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
