"""Project configured process capabilities into an immutable Workcell ABox."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from rdflib import Graph, Namespace, URIRef
from rdflib.compare import to_isomorphic
from rdflib.namespace import RDF

from cais_spade_llm.spec2primitives.config import WorkcellProfile

from .ppr_tbox import OntologyContextError, TBoxSnapshot
from .resource_registry import ResourceRegistrySnapshot


class PredefinedWorkcellError(OntologyContextError):
    """Raised when the predefined Workcell ABox cannot be trusted."""


@dataclass(frozen=True)
class PredefinedWorkcellSnapshot:
    """Hold the validated broad process capabilities for the workcell."""

    graph: Graph = field(repr=False, compare=False)
    processes: tuple[tuple[str, str], ...]
    resource_iris: tuple[str, ...]
    resource_capabilities: tuple[tuple[str, tuple[str, ...]], ...]
    ppr_namespace: str
    resource_namespace: str
    tbox_fingerprint: str
    registry_fingerprint: str
    fingerprint: str
    _tbox: TBoxSnapshot = field(repr=False, compare=False)
    _registry: ResourceRegistrySnapshot = field(repr=False, compare=False)
    _profile: WorkcellProfile = field(repr=False, compare=False)

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
        self._profile.assert_unchanged()

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
            resource_iris=self.resource_iris,
            processes=self.processes,
            resource_capabilities=self.resource_capabilities,
        ):
            raise PredefinedWorkcellError("Predefined Workcell graph changed after validation.")

        payload = self.to_record()
        payload.pop("fingerprint")
        if _record_fingerprint(payload) != self.fingerprint:
            raise PredefinedWorkcellError(
                "Predefined Workcell fingerprint changed after validation."
            )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe identity and provenance projection."""
        return {
            "schema_version": 2,
            "record_type": "PredefinedWorkcellSnapshot",
            "processes": [
                {"process_symbol": symbol, "process_iri": iri} for symbol, iri in self.processes
            ],
            "resource_iris": list(self.resource_iris),
            "resource_capabilities": [
                {
                    "resource_iri": resource_iri,
                    "capable_process_iris": list(process_iris),
                }
                for resource_iri, process_iris in self.resource_capabilities
            ],
            "ppr_namespace": self.ppr_namespace,
            "resource_namespace": self.resource_namespace,
            "tbox_fingerprint": self.tbox_fingerprint,
            "registry_fingerprint": self.registry_fingerprint,
            "workcell_profile_sha256": self._profile.source_sha256,
            "graph_fingerprint": _graph_fingerprint(self.graph),
            "fingerprint": self.fingerprint,
        }

    def process_symbol_for_iri(self, process_iri: str) -> str:
        """Return the exact configured symbol for one authorized process IRI."""
        matches = [symbol for symbol, iri in self.processes if iri == process_iri]
        if len(matches) != 1:
            raise PredefinedWorkcellError("Process IRI is not uniquely authorized.")
        return matches[0]

    def capable_resource_iris(self, process_iri: str) -> tuple[str, ...]:
        """Return configured resources broadly capable of one exact process."""
        if process_iri not in {iri for _symbol, iri in self.processes}:
            raise PredefinedWorkcellError("Process IRI is not authorized by the workcell.")
        return tuple(
            resource_iri
            for resource_iri, process_iris in self.resource_capabilities
            if process_iri in process_iris
        )


def load_predefined_workcell(
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    *,
    profile: WorkcellProfile | None = None,
) -> PredefinedWorkcellSnapshot:
    """Load exact broad process capabilities for the predefined workcell.

    Args:
        tbox: Validated immutable PPR TBox that defines the Workcell vocabulary.
        registry: Validated identity-only resource registry.

    Returns:
        A Workcell ABox pinned to both ontology authorities.

    Raises:
        PredefinedWorkcellError: If either input or the fixed Workcell profile
            cannot be validated exactly.
    """
    if not isinstance(tbox, TBoxSnapshot):
        raise PredefinedWorkcellError("Predefined Workcell requires a TBoxSnapshot.")
    if not isinstance(registry, ResourceRegistrySnapshot):
        raise PredefinedWorkcellError("Predefined Workcell requires a ResourceRegistrySnapshot.")
    try:
        tbox.assert_unchanged()
    except OntologyContextError as exc:
        raise PredefinedWorkcellError("Predefined Workcell TBox is not immutable.") from exc
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
    configured_profile = registry._profile if profile is None else profile
    configured_profile.assert_unchanged()
    resource_identities = tuple(
        (entry.resource_symbol, entry.resource_iri) for entry in registry.resources
    )
    configured_resources = tuple(
        (entry.symbol, entry.iri) for entry in configured_profile.resources
    )
    if resource_identities != configured_resources:
        raise PredefinedWorkcellError(
            "Predefined Workcell resources do not match the validated profile."
        )

    processes = tuple((entry.symbol, entry.iri) for entry in configured_profile.processes)
    resource_iris = tuple(iri for _, iri in configured_resources)
    resource_capabilities = tuple(
        (entry.iri, entry.capable_process_iris) for entry in configured_profile.resources
    )
    graph = Graph()
    graph.bind("ppr", Namespace(tbox.ppr_namespace))
    graph.bind("resource", Namespace(registry.resource_namespace))
    for triple in _expected_graph(
        ppr_namespace=tbox.ppr_namespace,
        resource_iris=resource_iris,
        processes=processes,
        resource_capabilities=resource_capabilities,
    ):
        graph.add(triple)

    payload: dict[str, object] = {
        "schema_version": 2,
        "record_type": "PredefinedWorkcellSnapshot",
        "processes": [{"process_symbol": symbol, "process_iri": iri} for symbol, iri in processes],
        "resource_iris": list(resource_iris),
        "resource_capabilities": [
            {
                "resource_iri": resource_iri,
                "capable_process_iris": list(process_iris),
            }
            for resource_iri, process_iris in resource_capabilities
        ],
        "ppr_namespace": tbox.ppr_namespace,
        "resource_namespace": registry.resource_namespace,
        "tbox_fingerprint": tbox.fingerprint,
        "registry_fingerprint": registry.fingerprint,
        "workcell_profile_sha256": configured_profile.source_sha256,
        "graph_fingerprint": _graph_fingerprint(graph),
    }
    snapshot = PredefinedWorkcellSnapshot(
        graph=graph,
        processes=processes,
        resource_iris=resource_iris,
        resource_capabilities=resource_capabilities,
        ppr_namespace=tbox.ppr_namespace,
        resource_namespace=registry.resource_namespace,
        tbox_fingerprint=tbox.fingerprint,
        registry_fingerprint=registry.fingerprint,
        fingerprint=_record_fingerprint(payload),
        _tbox=tbox,
        _registry=registry,
        _profile=configured_profile,
    )
    snapshot.assert_unchanged()
    return snapshot


def _validate_snapshot_symbols(snapshot: PredefinedWorkcellSnapshot) -> None:
    expected_resource_iris = tuple(item.iri for item in snapshot._profile.resources)
    expected_processes = tuple((item.symbol, item.iri) for item in snapshot._profile.processes)
    expected_capabilities = tuple(
        (item.iri, item.capable_process_iris) for item in snapshot._profile.resources
    )
    if (
        snapshot.processes != expected_processes
        or snapshot.resource_iris != expected_resource_iris
        or snapshot.resource_capabilities != expected_capabilities
        or snapshot.ppr_namespace != snapshot._tbox.ppr_namespace
        or snapshot.resource_namespace != snapshot._registry.resource_namespace
    ):
        raise PredefinedWorkcellError("Predefined Workcell fixed symbols changed after validation.")


def _expected_graph(
    *,
    ppr_namespace: str,
    resource_iris: tuple[str, ...],
    processes: tuple[tuple[str, str], ...],
    resource_capabilities: tuple[tuple[str, tuple[str, ...]], ...],
) -> set[tuple[URIRef, URIRef, URIRef]]:
    ppr = Namespace(ppr_namespace)
    graph = {
        (URIRef(process_iri), RDF.type, ppr.process) for _process_symbol, process_iri in processes
    }
    capability_map = dict(resource_capabilities)
    for resource_iri in resource_iris:
        resource = URIRef(resource_iri)
        graph.add((resource, RDF.type, ppr.resource))
        for process_iri in capability_map[resource_iri]:
            graph.add((resource, ppr.capableOf, URIRef(process_iri)))
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
