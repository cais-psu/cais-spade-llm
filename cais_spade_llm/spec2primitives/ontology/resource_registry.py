"""Project predefined robot resources into a minimal immutable registry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph, Namespace, URIRef
from rdflib.compare import to_isomorphic
from rdflib.namespace import RDF

from .ppr_tbox import OntologyContextError, TBoxSnapshot

_RESOURCE_NAMESPACE = "https://cais-spade-llm.local/resource/"
_EXPECTED_RESOURCE_SYMBOLS = ("xarm6", "ur5e")
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT = _PACKAGE_ROOT.parent
_DEFAULT_MANIFEST_PATHS = {
    "xarm6": _PACKAGE_ROOT / "initialization/resources/robot_xarm6.json",
    "ur5e": _PACKAGE_ROOT / "initialization/resources/robot_ur5e.json",
}


class ResourceRegistryError(OntologyContextError):
    """Raised when a predefined resource registry cannot be trusted."""


@dataclass(frozen=True)
class ResourceRegistryEntry:
    """Identify one predefined resource and its authoritative manifest."""

    resource_symbol: str
    resource_iri: str
    resource_jid: str
    resource_type: str
    source_ref: str
    source_sha256: str

    def to_record(self) -> dict[str, str]:
        """Return the configuration-free public registry entry."""
        return {
            "resource_symbol": self.resource_symbol,
            "resource_iri": self.resource_iri,
            "resource_jid": self.resource_jid,
            "resource_type": self.resource_type,
            "source_ref": self.source_ref,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class ResourceRegistrySnapshot:
    """Hold the validated flat resource graph and its source fingerprints."""

    graph: Graph = field(repr=False, compare=False)
    resources: tuple[ResourceRegistryEntry, ...]
    ppr_namespace: str
    resource_namespace: str
    tbox_fingerprint: str
    fingerprint: str
    _source_paths: tuple[Path, ...] = field(repr=False, compare=False)
    _tbox: TBoxSnapshot = field(repr=False, compare=False)

    def assert_unchanged(self) -> None:
        """Raise if the TBox, graph, or an authoritative manifest changed."""
        try:
            self._tbox.assert_unchanged()
        except OntologyContextError as exc:
            raise ResourceRegistryError(
                "Resource registry TBox changed after validation."
            ) from exc
        if self._tbox.fingerprint != self.tbox_fingerprint:
            raise ResourceRegistryError(
                "Resource registry TBox fingerprint changed after validation."
            )

        ppr = Namespace(self.ppr_namespace)
        expected_graph = {
            (URIRef(entry.resource_iri), RDF.type, ppr.resource)
            for entry in self.resources
        }
        if set(self.graph) != expected_graph:
            raise ResourceRegistryError(
                "Resource registry graph changed after validation."
            )

        for entry, source_path in zip(self.resources, self._source_paths, strict=True):
            if _source_sha256(source_path) != entry.source_sha256:
                raise ResourceRegistryError(
                    f"Resource manifest changed after validation: {entry.source_ref}"
                )

        payload = self.to_record()
        payload.pop("fingerprint")
        if _record_fingerprint(payload) != self.fingerprint:
            raise ResourceRegistryError(
                "Resource registry fingerprint changed after validation."
            )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe registry record without manifest configuration."""
        return {
            "schema_version": 1,
            "record_type": "ResourceRegistrySnapshot",
            "ppr_namespace": self.ppr_namespace,
            "resource_namespace": self.resource_namespace,
            "tbox_fingerprint": self.tbox_fingerprint,
            "graph_fingerprint": _graph_fingerprint(self.graph),
            "resources": [entry.to_record() for entry in self.resources],
            "fingerprint": self.fingerprint,
        }


def load_predefined_resource_registry(
    tbox: TBoxSnapshot,
    *,
    manifest_paths: Mapping[str, Path] | None = None,
    source_root: Path | None = None,
) -> ResourceRegistrySnapshot:
    """Load the exact predefined robot identities from authoritative manifests.

    Args:
        tbox: Validated immutable PPR TBox used to type resource individuals.
        manifest_paths: Optional exact symbol-to-manifest mapping for tests.
        source_root: Root used to create portable manifest references.

    Returns:
        A configuration-free registry snapshot for ``xarm6`` and ``ur5e``.

    Raises:
        ResourceRegistryError: If a manifest, resource identity, or source
            reference does not match the predefined flat-resource profile.
    """
    if not isinstance(tbox, TBoxSnapshot):
        raise ResourceRegistryError("Resource registry requires a TBoxSnapshot.")
    try:
        tbox.assert_unchanged()
    except OntologyContextError as exc:
        raise ResourceRegistryError("Resource registry TBox is not immutable.") from exc

    configured_paths = dict(
        _DEFAULT_MANIFEST_PATHS if manifest_paths is None else manifest_paths
    )
    if set(configured_paths) != set(_EXPECTED_RESOURCE_SYMBOLS):
        raise ResourceRegistryError(
            "Resource registry requires exactly the predefined symbols: "
            + ", ".join(_EXPECTED_RESOURCE_SYMBOLS)
        )

    resolved_source_root = Path(
        _REPOSITORY_ROOT if source_root is None else source_root
    ).resolve()
    resources: list[ResourceRegistryEntry] = []
    source_paths: list[Path] = []
    resource_jids: set[str] = set()
    for resource_symbol in _EXPECTED_RESOURCE_SYMBOLS:
        entry, source_path = _load_resource_entry(
            resource_symbol,
            configured_paths[resource_symbol],
            source_root=resolved_source_root,
        )
        if entry.resource_jid in resource_jids:
            raise ResourceRegistryError(
                f"Resource JID is not unique: {entry.resource_jid}"
            )
        resource_jids.add(entry.resource_jid)
        resources.append(entry)
        source_paths.append(source_path)

    ppr = Namespace(tbox.ppr_namespace)
    graph = Graph()
    graph.bind("ppr", ppr)
    graph.bind("resource", Namespace(_RESOURCE_NAMESPACE))
    for entry in resources:
        graph.add((URIRef(entry.resource_iri), RDF.type, ppr.resource))

    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "ResourceRegistrySnapshot",
        "ppr_namespace": tbox.ppr_namespace,
        "resource_namespace": _RESOURCE_NAMESPACE,
        "tbox_fingerprint": tbox.fingerprint,
        "graph_fingerprint": _graph_fingerprint(graph),
        "resources": [entry.to_record() for entry in resources],
    }
    return ResourceRegistrySnapshot(
        graph=graph,
        resources=tuple(resources),
        ppr_namespace=tbox.ppr_namespace,
        resource_namespace=_RESOURCE_NAMESPACE,
        tbox_fingerprint=tbox.fingerprint,
        fingerprint=_record_fingerprint(payload),
        _source_paths=tuple(source_paths),
        _tbox=tbox,
    )


def _load_resource_entry(
    resource_symbol: str,
    manifest_path: Path,
    *,
    source_root: Path,
) -> tuple[ResourceRegistryEntry, Path]:
    source_path = Path(manifest_path).resolve()
    source_bytes = _read_source_bytes(source_path)
    try:
        payload = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResourceRegistryError(
            f"Resource manifest is not valid JSON: {source_path}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {resource_symbol}:
        raise ResourceRegistryError(
            f"Resource manifest must contain only the exact symbol '{resource_symbol}'."
        )
    resource = payload[resource_symbol]
    if not isinstance(resource, dict):
        raise ResourceRegistryError(
            f"Resource manifest entry must be an object: {resource_symbol}"
        )

    resource_type = resource.get("type")
    if resource_type != "robot":
        raise ResourceRegistryError(
            f"Resource type must be exactly 'robot': {resource_symbol}"
        )
    resource_jid = _exact_nonempty_string(
        resource.get("jid"),
        f"Resource JID for {resource_symbol}",
    )
    source_ref = _relative_source_ref(source_path, source_root)
    return (
        ResourceRegistryEntry(
            resource_symbol=resource_symbol,
            resource_iri=f"{_RESOURCE_NAMESPACE}{resource_symbol}",
            resource_jid=resource_jid,
            resource_type=resource_type,
            source_ref=source_ref,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        ),
        source_path,
    )


def _read_source_bytes(source_path: Path) -> bytes:
    try:
        return source_path.read_bytes()
    except OSError as exc:
        raise ResourceRegistryError(
            f"Resource manifest cannot be read: {source_path}"
        ) from exc


def _source_sha256(source_path: Path) -> str:
    return hashlib.sha256(_read_source_bytes(source_path)).hexdigest()


def _relative_source_ref(source_path: Path, source_root: Path) -> str:
    try:
        return source_path.relative_to(source_root).as_posix()
    except ValueError as exc:
        raise ResourceRegistryError(
            f"Resource manifest is outside the configured source root: {source_path}"
        ) from exc


def _exact_nonempty_string(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise ResourceRegistryError(
            f"{label} must be an exact non-empty string without whitespace."
        )
    return value


def _graph_fingerprint(graph: Graph) -> str:
    digest = to_isomorphic(graph).graph_digest()
    return hashlib.sha256(str(digest).encode("ascii")).hexdigest()


def _record_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
