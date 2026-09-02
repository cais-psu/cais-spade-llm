"""Tests for the immutable predefined assembly Workcell projection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import Namespace, URIRef
from rdflib.namespace import RDF

from cais_spade_llm.spec2primitives.config import load_workcell_profile
from cais_spade_llm.spec2primitives.ontology import (
    PredefinedWorkcellError,
    ResourceRegistrySnapshot,
    TBoxSnapshot,
    load_ppr_tbox,
    load_predefined_resource_registry,
    load_predefined_workcell,
)

SPEC2PRIMITIVES_ROOT = Path(__file__).parents[1]
TBOX_PATH = SPEC2PRIMITIVES_ROOT / "ontology/spec2primitives_ppr_tbox.owl"
PPR_NAMESPACE = "http://PAonto.com#"
PROCESS_NAMESPACE = "https://cais-spade-llm.local/process/"
RESOURCE_NAMESPACE = "https://cais-spade-llm.local/resource/"


def test_predefined_workcell_contains_only_broad_assembly_capabilities() -> None:
    tbox, registry = _load_authorities()

    workcell = load_predefined_workcell(tbox, registry)

    ppr = Namespace(PPR_NAMESPACE)
    assembly = URIRef(f"{PROCESS_NAMESPACE}assembly")
    xarm6 = URIRef(f"{RESOURCE_NAMESPACE}xarm6")
    ur5e = URIRef(f"{RESOURCE_NAMESPACE}ur5e")
    assert workcell.processes == (("assembly", str(assembly)),)
    assert workcell.resource_iris == (str(xarm6), str(ur5e))
    assert workcell.resource_capabilities == (
        (str(xarm6), (str(assembly),)),
        (str(ur5e), (str(assembly),)),
    )
    assert workcell.process_symbol_for_iri(str(assembly)) == "assembly"
    assert workcell.capable_resource_iris(str(assembly)) == (str(xarm6), str(ur5e))
    assert set(workcell.graph) == {
        (assembly, RDF.type, ppr.process),
        (xarm6, RDF.type, ppr.resource),
        (xarm6, ppr.capableOf, assembly),
        (ur5e, RDF.type, ppr.resource),
        (ur5e, ppr.capableOf, assembly),
    }
    assert set(registry.graph) == {
        (xarm6, RDF.type, ppr.resource),
        (ur5e, RDF.type, ppr.resource),
    }
    assert not any(registry.graph.triples((None, ppr.capableOf, None)))
    workcell.assert_unchanged()


def test_predefined_workcell_record_is_deterministic_and_configuration_free() -> None:
    tbox, registry = _load_authorities()

    first = load_predefined_workcell(tbox, registry)
    second = load_predefined_workcell(tbox, registry)

    assert first.fingerprint == second.fingerprint
    assert first.to_record() == second.to_record()
    record = first.to_record()
    assert record["schema_version"] == 2
    assert record["record_type"] == "PredefinedWorkcellSnapshot"
    assert record["tbox_fingerprint"] == tbox.fingerprint
    assert record["registry_fingerprint"] == registry.fingerprint
    serialized = json.dumps(record, sort_keys=True)
    for forbidden_detail in (
        "jid",
        "password",
        "workspace_bounds",
        "gripper_reach",
        "controller",
        "named_positions",
        "services",
    ):
        assert forbidden_detail not in serialized


def test_workcell_projects_multiple_configured_process_capabilities(tmp_path: Path) -> None:
    tbox = load_ppr_tbox(TBOX_PATH, ppr_namespace=PPR_NAMESPACE)
    process_iris = {
        "process_alpha": "https://example.local/process/process_alpha",
        "process_beta": "https://example.local/process/process_beta",
    }
    resources = {
        "resource_alpha": (process_iris["process_alpha"],),
        "resource_beta": (process_iris["process_beta"],),
    }
    for symbol in resources:
        (tmp_path / f"{symbol}.json").write_text(
            json.dumps({symbol: {"type": "robot", "jid": f"{symbol}@localhost"}}),
            encoding="utf-8",
        )
    profile_path = tmp_path / "workcell_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "processes": [
                    {"symbol": symbol, "iri": iri} for symbol, iri in process_iris.items()
                ],
                "resources": [
                    {
                        "symbol": symbol,
                        "iri": f"https://example.local/resource/{symbol}",
                        "manifest_ref": f"{symbol}.json",
                        "capable_process_iris": list(capabilities),
                    }
                    for symbol, capabilities in resources.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    profile = load_workcell_profile(profile_path, repository_root=tmp_path)
    registry = load_predefined_resource_registry(tbox, profile=profile)

    workcell = load_predefined_workcell(tbox, registry, profile=profile)

    ppr = Namespace(PPR_NAMESPACE)
    assert workcell.processes == tuple(process_iris.items())
    assert workcell.capable_resource_iris(process_iris["process_alpha"]) == (
        "https://example.local/resource/resource_alpha",
    )
    assert workcell.capable_resource_iris(process_iris["process_beta"]) == (
        "https://example.local/resource/resource_beta",
    )
    assert (
        URIRef("https://example.local/resource/resource_alpha"),
        ppr.capableOf,
        URIRef(process_iris["process_alpha"]),
    ) in workcell.graph
    assert (
        URIRef("https://example.local/resource/resource_alpha"),
        ppr.capableOf,
        URIRef(process_iris["process_beta"]),
    ) not in workcell.graph


def test_predefined_workcell_detects_graph_and_pinned_authority_changes() -> None:
    tbox, registry = _load_authorities()
    workcell = load_predefined_workcell(tbox, registry)
    ppr = Namespace(PPR_NAMESPACE)
    unexpected = URIRef(f"{RESOURCE_NAMESPACE}unexpected")

    workcell.graph.add((unexpected, RDF.type, ppr.resource))
    with pytest.raises(PredefinedWorkcellError, match="graph changed"):
        workcell.assert_unchanged()

    workcell.graph.remove((unexpected, RDF.type, ppr.resource))
    registry.graph.add((unexpected, RDF.type, ppr.resource))
    with pytest.raises(PredefinedWorkcellError, match="registry changed"):
        workcell.assert_unchanged()

    registry.graph.remove((unexpected, RDF.type, ppr.resource))
    tbox.graph.add((ppr.processExecution, ppr.label, URIRef(f"{PPR_NAMESPACE}changed")))
    with pytest.raises(PredefinedWorkcellError, match="TBox changed"):
        workcell.assert_unchanged()


def test_predefined_workcell_rejects_a_registry_from_another_tbox(
    tmp_path: Path,
) -> None:
    original_tbox, registry = _load_authorities()
    changed_path = tmp_path / "changed_tbox.owl"
    changed_path.write_text(
        TBOX_PATH.read_text(encoding="utf-8").replace(
            "</rdf:RDF>",
            '  <owl:Class rdf:about="#additionalSchemaClass"/>\n\n</rdf:RDF>',
        ),
        encoding="utf-8",
    )
    changed_tbox = load_ppr_tbox(changed_path, ppr_namespace=PPR_NAMESPACE)
    assert changed_tbox.fingerprint != original_tbox.fingerprint

    with pytest.raises(PredefinedWorkcellError, match="does not match"):
        load_predefined_workcell(changed_tbox, registry)


def _load_authorities() -> tuple[TBoxSnapshot, ResourceRegistrySnapshot]:
    tbox = load_ppr_tbox(TBOX_PATH, ppr_namespace=PPR_NAMESPACE)
    return tbox, load_predefined_resource_registry(tbox)
