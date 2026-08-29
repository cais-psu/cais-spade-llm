"""Project the predefined assembly workcell into an immutable minimal ABox."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from rdflib import Graph, Namespace, URIRef
from rdflib.compare import to_isomorphic
from rdflib.namespace import RDF

from .ppr_tbox import OntologyContextError, TBoxSnapshot
from .resource_registry import ResourceRegistrySnapshot

_PROCESS_NAMESPACE = "https://cais-spade-llm.local/process/"
_ASSEMBLY_SYMBOL = "assembly"
_EXPECTED_RESOURCES = (
    ("xarm6", "https://cais-spade-llm.local/resource/xarm6"),
    ("ur5e", "https://cais-spade-llm.local/resource/ur5e"),
)


class PredefinedWorkcellError(OntologyContextError):
    """Raised when the predefined Workcell ABox cannot be trusted."""


@dataclass(frozen=True)
class PredefinedWorkcellSnapshot:
    """Hold the validated broad assembly capabilities for the fixed workcell."""

    graph: Graph = field(repr=False, compare=False)
    process_symbol: str
    process_iri: str
    resource_iris: tuple[str, ...]
    ppr_namespace: str
    process_namespace: str
    resource_namespace: str
    tbox_fingerprint: str
    registry_fingerprint: str
    fingerprint: str
    _tbox: TBoxSnapshot = field(repr=False, compare=False)
    _registry: ResourceRegistrySnapshot = field(repr=False, compare=False)

    def assert_unchanged(self) -> None:
        """Raise if a pinned ontology authority or this graph has changed."""
        try:
            self._tbox.assert_unchanged()
        except OntologyContextError as exc:
            raise PredefinedWorkcellError(
                "Predefined Workcell TBox changed after validation."
            ) from exc
        try:
            self._registry.assert_unchanged()
        except OntologyContextError as exc:
            raise PredefinedWorkcellError(
                "Predefined Workcell resource registry changed after validation."
            ) from exc

        if self._tbox.fingerprint != self.tbox_fingerprint:
            raise PredefinedWorkcellError(
                "Predefined Workcell TBox fingerprint changed after validation."
            )
        if self._registry.fingerprint != self.registry_fingerprint:
            raise PredefinedWorkcellError(
                "Predefined Workcell registry fingerprint changed after validation."
            )
        _validate_snapshot_symbols(self)
        if set(self.graph) != _expected_graph(
            ppr_namespace=self.ppr_namespace,
            process_iri=self.process_iri,
        ):
            raise PredefinedWorkcellError(
                "Predefined Workcell graph changed after validation."
            )

        payload = self.to_record()
        payload.pop("fingerprint")
        if _record_fingerprint(payload) != self.fingerprint:
            raise PredefinedWorkcellError(
                "Predefined Workcell fingerprint changed after validation."
            )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe identity and provenance projection."""
        return {
            "schema_version": 1,
            "record_type": "PredefinedWorkcellSnapshot",
            "process_symbol": self.process_symbol,
            "process_iri": self.process_iri,
            "resource_iris": list(self.resource_iris),
            "ppr_namespace": self.ppr_namespace,
            "process_namespace": self.process_namespace,
            "resource_namespace": self.resource_namespace,
            "tbox_fingerprint": self.tbox_fingerprint,
            "registry_fingerprint": self.registry_fingerprint,
            "graph_fingerprint": _graph_fingerprint(self.graph),
            "fingerprint": self.fingerprint,
        }


def load_predefined_workcell(
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
) -> PredefinedWorkcellSnapshot:
    """Load the exact broad assembly capabilities for the predefined workcell.

    Args:
        tbox: Validated immutable PPR TBox that defines the Workcell vocabulary.
        registry: Validated identity-only registry for ``xarm6`` and ``ur5e``.

    Returns:
        A five-triple Workcell ABox pinned to both ontology authorities.

    Raises:
        PredefinedWorkcellError: If either input or the fixed Workcell profile
            cannot be validated exactly.
    """
    if not isinstance(tbox, TBoxSnapshot):
        raise PredefinedWorkcellError("Predefined Workcell requires a TBoxSnapshot.")
    if not isinstance(registry, ResourceRegistrySnapshot):
        raise PredefinedWorkcellError(
            "Predefined Workcell requires a ResourceRegistrySnapshot."
        )
    try:
        tbox.assert_unchanged()
    except OntologyContextError as exc:
        raise PredefinedWorkcellError(
            "Predefined Workcell TBox is not immutable."
        ) from exc
    try:
        registry.assert_unchanged()
    except OntologyContextError as exc:
        raise PredefinedWorkcellError(
            "Predefined Workcell resource registry is not immutable."
        ) from exc

    if registry.tbox_fingerprint != tbox.fingerprint:
        raise PredefinedWorkcellError(
            "Predefined Workcell TBox does not match the resource registry."
        )
    if registry.ppr_namespace != tbox.ppr_namespace:
        raise PredefinedWorkcellError(
            "Predefined Workcell PPR namespace does not match the resource registry."
        )
    resource_identities = tuple(
        (entry.resource_symbol, entry.resource_iri) for entry in registry.resources
    )
    if resource_identities != _EXPECTED_RESOURCES:
        # Broad capability assertions are safe only for the reviewed fixed workcell.
        raise PredefinedWorkcellError(
            "Predefined Workcell requires exactly xarm6 and ur5e in registry order."
        )

    process_iri = f"{_PROCESS_NAMESPACE}{_ASSEMBLY_SYMBOL}"
    graph = Graph()
    graph.bind("ppr", Namespace(tbox.ppr_namespace))
    graph.bind("process", Namespace(_PROCESS_NAMESPACE))
    graph.bind("resource", Namespace(registry.resource_namespace))
    for triple in _expected_graph(
        ppr_namespace=tbox.ppr_namespace,
        process_iri=process_iri,
    ):
        graph.add(triple)

    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "PredefinedWorkcellSnapshot",
        "process_symbol": _ASSEMBLY_SYMBOL,
        "process_iri": process_iri,
        "resource_iris": [resource_iri for _, resource_iri in _EXPECTED_RESOURCES],
        "ppr_namespace": tbox.ppr_namespace,
        "process_namespace": _PROCESS_NAMESPACE,
        "resource_namespace": registry.resource_namespace,
        "tbox_fingerprint": tbox.fingerprint,
        "registry_fingerprint": registry.fingerprint,
        "graph_fingerprint": _graph_fingerprint(graph),
    }
    snapshot = PredefinedWorkcellSnapshot(
        graph=graph,
        process_symbol=_ASSEMBLY_SYMBOL,
        process_iri=process_iri,
        resource_iris=tuple(resource_iri for _, resource_iri in _EXPECTED_RESOURCES),
        ppr_namespace=tbox.ppr_namespace,
        process_namespace=_PROCESS_NAMESPACE,
        resource_namespace=registry.resource_namespace,
        tbox_fingerprint=tbox.fingerprint,
        registry_fingerprint=registry.fingerprint,
        fingerprint=_record_fingerprint(payload),
        _tbox=tbox,
        _registry=registry,
    )
    snapshot.assert_unchanged()
    return snapshot


def _validate_snapshot_symbols(snapshot: PredefinedWorkcellSnapshot) -> None:
    expected_process_iri = f"{_PROCESS_NAMESPACE}{_ASSEMBLY_SYMBOL}"
    expected_resource_iris = tuple(
        resource_iri for _, resource_iri in _EXPECTED_RESOURCES
    )
    if (
        snapshot.process_symbol != _ASSEMBLY_SYMBOL
        or snapshot.process_iri != expected_process_iri
        or snapshot.resource_iris != expected_resource_iris
        or snapshot.process_namespace != _PROCESS_NAMESPACE
        or snapshot.ppr_namespace != snapshot._tbox.ppr_namespace
        or snapshot.resource_namespace != snapshot._registry.resource_namespace
    ):
        raise PredefinedWorkcellError(
            "Predefined Workcell fixed symbols changed after validation."
        )


def _expected_graph(
    *,
    ppr_namespace: str,
    process_iri: str,
) -> set[tuple[URIRef, URIRef, URIRef]]:
    ppr = Namespace(ppr_namespace)
    process = URIRef(process_iri)
    graph = {(process, RDF.type, ppr.process)}
    for _, resource_iri in _EXPECTED_RESOURCES:
        resource = URIRef(resource_iri)
        graph.add((resource, RDF.type, ppr.resource))
        graph.add((resource, ppr.capableOf, process))
    return graph


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
