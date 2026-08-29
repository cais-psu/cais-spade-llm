"""Tests for the predefined flat resource registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from rdflib import Namespace, URIRef
from rdflib.namespace import RDF

from cais_spade_llm.spec2primitives.ontology import (
    ResourceRegistryError,
    load_ppr_tbox,
    load_predefined_resource_registry,
)

SPEC2PRIMITIVES_ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = SPEC2PRIMITIVES_ROOT.parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
TBOX_PATH = SPEC2PRIMITIVES_ROOT / "ontology/spec2primitives_ppr_tbox.owl"
PPR_NAMESPACE = "http://PAonto.com#"
RESOURCE_NAMESPACE = "https://cais-spade-llm.local/resource/"
DEFAULT_MANIFEST_PATHS = {
    "xarm6": PACKAGE_ROOT / "initialization/resources/robot_xarm6.json",
    "ur5e": PACKAGE_ROOT / "initialization/resources/robot_ur5e.json",
}


def test_predefined_registry_projects_only_exact_resource_identities() -> None:
    tbox = load_ppr_tbox(TBOX_PATH, ppr_namespace=PPR_NAMESPACE)

    registry = load_predefined_resource_registry(tbox)

    assert registry.resource_namespace == RESOURCE_NAMESPACE
    assert [entry.resource_symbol for entry in registry.resources] == ["xarm6", "ur5e"]
    assert [entry.resource_jid for entry in registry.resources] == [
        "xarm6@localhost",
        "ur5e@localhost",
    ]
    assert [entry.resource_iri for entry in registry.resources] == [
        f"{RESOURCE_NAMESPACE}xarm6",
        f"{RESOURCE_NAMESPACE}ur5e",
    ]
    ppr = Namespace(PPR_NAMESPACE)
    assert set(registry.graph) == {
        (URIRef(f"{RESOURCE_NAMESPACE}xarm6"), RDF.type, ppr.resource),
        (URIRef(f"{RESOURCE_NAMESPACE}ur5e"), RDF.type, ppr.resource),
    }
    assert not any(registry.graph.triples((None, ppr.capableOf, None)))
    registry.assert_unchanged()


def test_registry_record_pins_sources_without_copying_configuration() -> None:
    tbox = load_ppr_tbox(TBOX_PATH, ppr_namespace=PPR_NAMESPACE)

    registry = load_predefined_resource_registry(tbox)
    record = registry.to_record()

    assert record["record_type"] == "ResourceRegistrySnapshot"
    assert len(str(record["fingerprint"])) == 64
    for entry in registry.resources:
        source_path = DEFAULT_MANIFEST_PATHS[entry.resource_symbol]
        assert entry.source_ref == source_path.relative_to(REPOSITORY_ROOT).as_posix()
        assert entry.source_sha256 == hashlib.sha256(source_path.read_bytes()).hexdigest()
    serialized = json.dumps(record, sort_keys=True)
    for forbidden_detail in (
        "password",
        "workspace_bounds",
        "gripper_reach",
        "controller",
        "named_positions",
        "services",
    ):
        assert forbidden_detail not in serialized


def test_registry_fingerprint_is_deterministic() -> None:
    tbox = load_ppr_tbox(TBOX_PATH, ppr_namespace=PPR_NAMESPACE)

    first = load_predefined_resource_registry(tbox)
    second = load_predefined_resource_registry(tbox)

    assert first.fingerprint == second.fingerprint
    assert first.to_record() == second.to_record()


def test_registry_detects_graph_and_manifest_changes(tmp_path: Path) -> None:
    manifest_paths = _write_valid_manifests(tmp_path)
    tbox = load_ppr_tbox(TBOX_PATH, ppr_namespace=PPR_NAMESPACE)
    registry = load_predefined_resource_registry(
        tbox,
        manifest_paths=manifest_paths,
        source_root=tmp_path,
    )

    registry.graph.add((URIRef(f"{RESOURCE_NAMESPACE}unexpected"), RDF.type, Namespace(PPR_NAMESPACE).resource))
    with pytest.raises(ResourceRegistryError, match="graph changed"):
        registry.assert_unchanged()

    registry.graph.remove(
        (URIRef(f"{RESOURCE_NAMESPACE}unexpected"), RDF.type, Namespace(PPR_NAMESPACE).resource)
    )
    manifest_paths["xarm6"].write_text(
        json.dumps({"xarm6": {"type": "robot", "jid": "changed@localhost"}}),
        encoding="utf-8",
    )
    with pytest.raises(ResourceRegistryError, match="manifest changed"):
        registry.assert_unchanged()


@pytest.mark.parametrize(
    ("resource_symbol", "payload", "message"),
    [
        ("xarm6", {"wrong": {"type": "robot", "jid": "xarm6@localhost"}}, "exact symbol"),
        ("xarm6", {"xarm6": {"type": "machine", "jid": "xarm6@localhost"}}, "exactly 'robot'"),
        ("xarm6", {"xarm6": {"type": "robot", "jid": ""}}, "exact non-empty"),
        (
            "xarm6",
            {"xarm6": {"type": "robot", "jid": " xarm6@localhost"}},
            "without whitespace",
        ),
    ],
)
def test_registry_rejects_invalid_resource_identity(
    tmp_path: Path,
    resource_symbol: str,
    payload: object,
    message: str,
) -> None:
    manifest_paths = _write_valid_manifests(tmp_path)
    manifest_paths[resource_symbol].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResourceRegistryError, match=message):
        load_predefined_resource_registry(
            load_ppr_tbox(TBOX_PATH, ppr_namespace=PPR_NAMESPACE),
            manifest_paths=manifest_paths,
            source_root=tmp_path,
        )


def test_registry_rejects_missing_malformed_duplicate_and_unexpected_resources(
    tmp_path: Path,
) -> None:
    tbox = load_ppr_tbox(TBOX_PATH, ppr_namespace=PPR_NAMESPACE)
    manifest_paths = _write_valid_manifests(tmp_path)
    manifest_paths["xarm6"].unlink()
    with pytest.raises(ResourceRegistryError, match="cannot be read"):
        load_predefined_resource_registry(
            tbox,
            manifest_paths=manifest_paths,
            source_root=tmp_path,
        )

    manifest_paths = _write_valid_manifests(tmp_path)
    manifest_paths["xarm6"].write_text("{", encoding="utf-8")
    with pytest.raises(ResourceRegistryError, match="not valid JSON"):
        load_predefined_resource_registry(
            tbox,
            manifest_paths=manifest_paths,
            source_root=tmp_path,
        )

    manifest_paths = _write_valid_manifests(tmp_path)
    manifest_paths["ur5e"].write_text(
        json.dumps({"ur5e": {"type": "robot", "jid": "xarm6@localhost"}}),
        encoding="utf-8",
    )
    with pytest.raises(ResourceRegistryError, match="not unique"):
        load_predefined_resource_registry(
            tbox,
            manifest_paths=manifest_paths,
            source_root=tmp_path,
        )

    unexpected_paths = {**_write_valid_manifests(tmp_path), "third": tmp_path / "third.json"}
    with pytest.raises(ResourceRegistryError, match="exactly the predefined symbols"):
        load_predefined_resource_registry(
            tbox,
            manifest_paths=unexpected_paths,
            source_root=tmp_path,
        )


def _write_valid_manifests(root: Path) -> dict[str, Path]:
    manifest_paths = {
        "xarm6": root / "robot_xarm6.json",
        "ur5e": root / "robot_ur5e.json",
    }
    for resource_symbol, manifest_path in manifest_paths.items():
        manifest_path.write_text(
            json.dumps(
                {
                    resource_symbol: {
                        "type": "robot",
                        "jid": f"{resource_symbol}@localhost",
                        "password": "not-projected",
                        "controller": {"detail": "not-projected"},
                    }
                }
            ),
            encoding="utf-8",
        )
    return manifest_paths
