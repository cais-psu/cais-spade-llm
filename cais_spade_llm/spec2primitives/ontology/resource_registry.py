from __future__ import annotations

"""Project predefined robot resources into a minimal immutable registry."""


import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph, Namespace, URIRef
from rdflib.compare import to_isomorphic
from rdflib.namespace import RDF

from cais_spade_llm.spec2primitives.config import WorkcellProfile, load_workcell_profile

from .ppr_tbox import OntologyContextError, TBoxSnapshot


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
    _profile: WorkcellProfile = field(repr=False, compare=False)

    def assert_unchanged(self) -> None:
        """Raise if the TBox, graph, or an authoritative manifest changed."""
        try:
            self._tbox.assert_unchanged()
        except OntologyContextError as exc:
            raise ResourceRegistryError("Resource registry TBox changed after validation.") from exc
        self._profile.assert_unchanged()
        if self._tbox.fingerprint != self.tbox_fingerprint:
            raise ResourceRegistryError(
                "Resource registry TBox fingerprint changed after validation."
            )

        ppr = Namespace(self.ppr_namespace)
        expected_graph = {
            (URIRef(entry.resource_iri), RDF.type, ppr.resource) for entry in self.resources
        }
        if set(self.graph) != expected_graph:
            raise ResourceRegistryError("Resource registry graph changed after validation.")

        for entry, source_path in zip(self.resources, self._source_paths, strict=True):
            if _source_sha256(source_path) != entry.source_sha256:
                raise ResourceRegistryError(
                    f"Resource manifest changed after validation: {entry.source_ref}"
                )

        payload = self.to_record()
        payload.pop("fingerprint")
        if _record_fingerprint(payload) != self.fingerprint:
            raise ResourceRegistryError("Resource registry fingerprint changed after validation.")

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe registry record without manifest configuration."""
        return {
            "record_type": "ResourceRegistrySnapshot",
            "ppr_namespace": self.ppr_namespace,
            "resource_namespace": self.resource_namespace,
            "tbox_fingerprint": self.tbox_fingerprint,
            "workcell_profile_sha256": self._profile.source_sha256,
            "graph_fingerprint": _graph_fingerprint(self.graph),
            "resources": [entry.to_record() for entry in self.resources],
            "fingerprint": self.fingerprint,
        }


def load_predefined_resource_registry(
    tbox: TBoxSnapshot,
    *,
    profile: WorkcellProfile | None = None,
) -> ResourceRegistrySnapshot:
    """Load ordered resource identities from the validated workcell profile.

    Args:
        tbox: Validated immutable PPR TBox used to type resource individuals.
        profile: Validated process, ordered resources, and manifest authorities.

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

    configured_profile = load_workcell_profile() if profile is None else profile
    configured_profile.assert_unchanged()
    resource_profiles = configured_profile.resources
    resources: list[ResourceRegistryEntry] = []
    source_paths: list[Path] = []
    resource_jids: set[str] = set()
    for resource_profile in resource_profiles:
        resource_symbol = resource_profile.symbol
        entry, source_path = _load_resource_entry(
            resource_symbol,
            resource_profile.manifest_path,
            resource_iri=resource_profile.iri,
            source_ref=resource_profile.manifest_ref,
        )
        if entry.resource_jid in resource_jids:
            raise ResourceRegistryError(f"Resource JID is not unique: {entry.resource_jid}")
        resource_jids.add(entry.resource_jid)
        resources.append(entry)
        source_paths.append(source_path)

    ppr = Namespace(tbox.ppr_namespace)
    graph = Graph()
    graph.bind("ppr", ppr)
    resource_namespace = _common_resource_namespace(
        tuple(entry.resource_iri for entry in resources)
    )
    graph.bind("resource", Namespace(resource_namespace))
    for entry in resources:
        graph.add((URIRef(entry.resource_iri), RDF.type, ppr.resource))

    payload: dict[str, object] = {
        "record_type": "ResourceRegistrySnapshot",
        "ppr_namespace": tbox.ppr_namespace,
        "resource_namespace": resource_namespace,
        "tbox_fingerprint": tbox.fingerprint,
        "workcell_profile_sha256": configured_profile.source_sha256,
        "graph_fingerprint": _graph_fingerprint(graph),
        "resources": [entry.to_record() for entry in resources],
    }
    return ResourceRegistrySnapshot(
        graph=graph,
        resources=tuple(resources),
        ppr_namespace=tbox.ppr_namespace,
        resource_namespace=resource_namespace,
        tbox_fingerprint=tbox.fingerprint,
        fingerprint=_record_fingerprint(payload),
        _source_paths=tuple(source_paths),
        _tbox=tbox,
        _profile=configured_profile,
    )


def _load_resource_entry(
    resource_symbol: str,
    manifest_path: Path,
    *,
    resource_iri: str,
    source_ref: str,
) -> tuple[ResourceRegistryEntry, Path]:
    source_path = Path(manifest_path).resolve()
    source_bytes = _read_source_bytes(source_path)
    try:
        payload = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResourceRegistryError(f"Resource manifest is not valid JSON: {source_path}") from exc
    if not isinstance(payload, dict) or set(payload) != {resource_symbol}:
        raise ResourceRegistryError(
            f"Resource manifest must contain only the exact symbol '{resource_symbol}'."
        )
    resource = payload[resource_symbol]
    if not isinstance(resource, dict):
        raise ResourceRegistryError(f"Resource manifest entry must be an object: {resource_symbol}")

    resource_type = resource.get("type")
    if resource_type != "robot":
        raise ResourceRegistryError(f"Resource type must be exactly 'robot': {resource_symbol}")
    resource_jid = _exact_nonempty_string(
        resource.get("jid"),
        f"Resource JID for {resource_symbol}",
    )
    return (
        ResourceRegistryEntry(
            resource_symbol=resource_symbol,
            resource_iri=resource_iri,
            resource_jid=resource_jid,
            resource_type=resource_type,
            source_ref=source_ref,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        ),
        source_path,
    )


def _common_resource_namespace(resource_iris: tuple[str, ...]) -> str:
    """Return the common IRI prefix used only for readable RDF bindings."""
    if not resource_iris:
        raise ResourceRegistryError("Resource registry is empty.")
    prefixes = [iri.rsplit("/", 1)[0] + "/" for iri in resource_iris]
    if len(set(prefixes)) != 1:
        raise ResourceRegistryError("Resource IRIs do not share one namespace.")
    return prefixes[0]


def _read_source_bytes(source_path: Path) -> bytes:
    try:
        return source_path.read_bytes()
    except OSError as exc:
        raise ResourceRegistryError(f"Resource manifest cannot be read: {source_path}") from exc


def _source_sha256(source_path: Path) -> str:
    return hashlib.sha256(_read_source_bytes(source_path)).hexdigest()


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
