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
PPR_NAMESPACE = "http://PAonto.com#"
PPR = Namespace(PPR_NAMESPACE)


def test_schema_only_tbox_loads_with_exact_required_vocabulary() -> None:
    tbox = _load_tbox()
    required_classes = {
        PPR.specification,
        PPR.product,
        PPR.feature,
        PPR.process,
        PPR.resource,
        PPR.capability,
    }
    required_properties = {PPR.defines, PPR.realizes, PPR.capableOf}

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
