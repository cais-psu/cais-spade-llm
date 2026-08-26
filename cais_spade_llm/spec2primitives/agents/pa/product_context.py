"""Maintain the Phase 4.0 PA-owned product context.

The immutable TBox constrains vocabulary. Each interaction receives an
independent ABox whose only initial individual is the unresolved specification.
Controlled evidence tools propose JSON-shaped deltas; this module validates and
persists them without deciding context completeness or composing primitives.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.exceptions import ParserError
from rdflib.namespace import RDF, RDFS
from rdflib.plugin import PluginException

from cais_spade_llm.spec2primitives.ontology import (
    OntologyContextError,
    TBoxProfileError,
    TBoxSnapshot,
)

_ONTOLOGY_ROOT = Path("products/grounding/ontology")
_ABOX_NAME = "interaction_abox.ttl"
_MANIFEST_NAME = "abox_manifest.json"
_PROVENANCE_NAME = "assertion_provenance.json"
_REQUIREMENT_EVIDENCE_REF = "products/user_requirement/product_requirement.json"
_SCHEMA_VERSION = 1

_PROHIBITED_PA_PROPERTIES = frozenset({"capableOf", "provides", "requires", "precedes"})
_TYPED_CONTEXT_PREFIX = "products/grounding/"
_DELTA_KEYS = frozenset(
    {
        "assertions",
        "uncertainty",
        "unresolved_evidence_needs",
        "typed_context_refs",
    }
)
_ASSERTION_KEYS = frozenset({"subject", "predicate", "object", "evidence_refs"})
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "interaction_namespace",
        "specification_iri",
        "tbox_fingerprint",
        "product_requirement",
        "delta_count",
        "accepted_assertion_count",
    }
)
_PROVENANCE_KEYS = frozenset({"schema_version", "tbox_fingerprint", "assertions"})
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")


class TripleDeltaError(OntologyContextError):
    """Raised when a proposed evidence delta is invalid."""


class OntologyPersistenceError(OntologyContextError):
    """Raised when an ontology context cannot be safely persisted or reloaded."""


@dataclass(frozen=True)
class ABoxSnapshot:
    """Persisted per-interaction PA ABox and its record paths."""

    graph: Graph = field(repr=False, compare=False)
    interaction_root: Path
    ontology_root: Path
    abox_path: Path
    manifest_path: Path
    provenance_path: Path
    namespace: str
    specification_iri: str
    tbox_fingerprint: str
    product_requirement: str
    delta_count: int
    accepted_assertion_count: int


@dataclass(frozen=True)
class TripleObject:
    """Disambiguated IRI or literal object in a proposed assertion."""

    kind: str
    value: str | int | float | bool
    datatype: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class TripleAssertion:
    """One evidence-backed factual assertion proposed by a tool."""

    subject: str
    predicate: str
    object: TripleObject
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class TripleDelta:
    """Side-effect-free evidence-tool result awaiting validation."""

    assertions: tuple[TripleAssertion, ...]
    uncertainty: tuple[object, ...]
    unresolved_evidence_needs: tuple[object, ...]
    typed_context_refs: tuple[str, ...]


@dataclass(frozen=True)
class MergeResult:
    """Result of one successful atomic-validation merge."""

    accepted: bool
    assertion_count: int
    delta_path: Path
    abox: ABoxSnapshot


def initialize_interaction_abox(
    interaction_root: Path,
    product_requirement: str,
    tbox: TBoxSnapshot,
) -> ABoxSnapshot:
    """Create one unresolved, independent interaction ABox.

    Args:
        interaction_root: Caller-owned `contexts/<interaction_identifier>/` root.
        product_requirement: Exact non-empty user requirement to preserve.
        tbox: Validated immutable TBox snapshot.

    Returns:
        The newly persisted ABox snapshot.

    Raises:
        OntologyContextError: If input or persistence validation fails.
    """
    _validate_tbox_snapshot(tbox)
    if not isinstance(product_requirement, str) or not product_requirement.strip():
        raise OntologyContextError("product_requirement must contain non-whitespace text.")

    root = Path(interaction_root).resolve()
    ontology_root = root / _ONTOLOGY_ROOT
    if ontology_root.exists():
        raise OntologyPersistenceError(f"Ontology context already exists: {ontology_root}")

    interaction_identifier = hashlib.sha256(os.urandom(32)).hexdigest()
    interaction_namespace = f"https://cais-spade-llm.local/context/{interaction_identifier}/"
    specification_iri = URIRef(f"{interaction_namespace}specification_1")
    ppr = Namespace(tbox.ppr_namespace)
    graph = Graph()
    graph.bind("ctx", Namespace(interaction_namespace))
    graph.bind("ppr", ppr)
    graph.bind("rdf", RDF)
    graph.add((specification_iri, RDF.type, ppr.specification))
    graph.add((specification_iri, RDF.value, Literal(product_requirement)))

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "status": "unresolved",
        "interaction_namespace": interaction_namespace,
        "specification_iri": str(specification_iri),
        "tbox_fingerprint": tbox.fingerprint,
        "product_requirement": product_requirement,
        "delta_count": 0,
        "accepted_assertion_count": 0,
    }
    provenance = {
        "schema_version": _SCHEMA_VERSION,
        "tbox_fingerprint": tbox.fingerprint,
        "assertions": _initializer_provenance(
            str(specification_iri),
            product_requirement,
            tbox.ppr_namespace,
        ),
    }
    _persist_new_abox(ontology_root, graph, manifest, provenance)
    return _snapshot_from_graph(root, graph, manifest)


def load_interaction_abox(
    interaction_root: Path,
    tbox: TBoxSnapshot,
) -> ABoxSnapshot:
    """Reload and validate one persisted PA interaction ABox.

    Args:
        interaction_root: Root containing the initialized ontology context.
        tbox: The exact immutable TBox snapshot used at initialization.

    Returns:
        The validated current ABox snapshot.

    Raises:
        OntologyContextError: If the graph, manifest, provenance, or TBox does
            not match the initialized interaction.
    """
    _validate_tbox_snapshot(tbox)
    abox, _manifest, _provenance = _load_persisted_abox(
        Path(interaction_root).resolve(),
        tbox,
    )
    return abox


def validate_and_merge_triple_delta(
    interaction_root: Path,
    tbox: TBoxSnapshot,
    producer: str,
    delta: TripleDelta | Mapping[str, object],
    *,
    authorized_evidence_refs: Iterable[str],
) -> MergeResult:
    """Validate and persist one evidence-backed delta without partial rejection.

    Args:
        interaction_root: Root containing a previously initialized ABox.
        tbox: The exact TBox snapshot used to initialize that ABox.
        producer: Fixed application-owned producer symbol used for audit records.
        delta: Proposed assertions plus non-RDF uncertainty and context refs.
        authorized_evidence_refs: Exact evidence refs permitted this turn. The
            caller constructs this trusted allowlist; it must never come from
            the proposed delta or its producing tool.

    Returns:
        A successful merge result. Invalid deltas raise instead of returning a
        partially accepted result.

    Raises:
        OntologyContextError: If the producer, delta, ABox, or write is invalid.
    """
    _validate_tbox_snapshot(tbox)
    validated_producer = _require_nonempty_string(
        producer,
        "producer",
        TripleDeltaError,
    )
    if not _SAFE_IDENTIFIER.fullmatch(validated_producer):
        raise TripleDeltaError("producer must be a fixed non-path identifier.")
    authorized_refs = _validated_authorized_refs(authorized_evidence_refs)
    normalized_delta = _coerce_delta(delta)
    root = Path(interaction_root).resolve()
    abox, manifest, provenance = _load_persisted_abox(root, tbox)

    _validate_delta_metadata(normalized_delta)
    candidate = Graph()
    for prefix, namespace in abox.graph.namespaces():
        candidate.bind(prefix, namespace)
    for triple in abox.graph:
        candidate.add(triple)

    normalized_assertions: list[dict[str, object]] = []
    for assertion in normalized_delta.assertions:
        triple, normalized_record = _validated_assertion(
            assertion,
            abox,
            tbox,
            authorized_refs,
        )
        candidate.add(triple)
        normalized_assertions.append(normalized_record)

    _validate_pa_abox_graph(candidate, manifest, tbox)
    next_delta_number = int(manifest["delta_count"]) + 1
    delta_path = abox.ontology_root / f"delta_{next_delta_number:04d}.json"
    if delta_path.exists():
        raise OntologyPersistenceError(f"Delta record already exists: {delta_path.name}")

    delta_record = {
        "schema_version": _SCHEMA_VERSION,
        "delta_number": next_delta_number,
        "producer": validated_producer,
        "assertions": normalized_assertions,
        "uncertainty": list(normalized_delta.uncertainty),
        "unresolved_evidence_needs": list(normalized_delta.unresolved_evidence_needs),
        "typed_context_refs": list(normalized_delta.typed_context_refs),
    }
    updated_provenance = dict(provenance)
    provenance_assertions = list(provenance["assertions"])
    for assertion in normalized_assertions:
        provenance_assertions.append(
            {
                **assertion,
                "producer": validated_producer,
                "delta_ref": delta_path.name,
            }
        )
    updated_provenance["assertions"] = provenance_assertions
    updated_manifest = dict(manifest)
    updated_manifest["delta_count"] = next_delta_number
    updated_manifest["accepted_assertion_count"] = int(manifest["accepted_assertion_count"]) + len(
        normalized_assertions
    )

    _persist_merge(
        abox,
        candidate,
        updated_manifest,
        updated_provenance,
        delta_path,
        delta_record,
    )
    updated_abox = _snapshot_from_graph(root, candidate, updated_manifest)
    return MergeResult(
        accepted=True,
        assertion_count=len(normalized_assertions),
        delta_path=delta_path,
        abox=updated_abox,
    )


def _validate_tbox_snapshot(tbox: object) -> None:
    if not isinstance(tbox, TBoxSnapshot):
        raise TBoxProfileError("Expected a validated TBoxSnapshot.")
    tbox.assert_unchanged()


def _initializer_provenance(
    specification_iri: str,
    product_requirement: str,
    ppr_namespace: str,
) -> list[dict[str, object]]:
    common = {
        "producer": "interaction_initializer",
        "evidence_refs": [_REQUIREMENT_EVIDENCE_REF],
        "delta_ref": None,
    }
    return [
        {
            "subject": specification_iri,
            "predicate": str(RDF.type),
            "object": {
                "kind": "iri",
                "value": f"{ppr_namespace}specification",
            },
            **common,
        },
        {
            "subject": specification_iri,
            "predicate": str(RDF.value),
            "object": {"kind": "literal", "value": product_requirement},
            **common,
        },
    ]


def _persist_new_abox(
    ontology_root: Path,
    graph: Graph,
    manifest: Mapping[str, object],
    provenance: Mapping[str, object],
) -> None:
    parent = ontology_root.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.mkdtemp(prefix=".ontology-", dir=parent))
    except OSError as exc:
        raise OntologyPersistenceError("Ontology context directory could not be created.") from exc
    try:
        graph.serialize(destination=temporary_root / _ABOX_NAME, format="turtle")
        _write_json(temporary_root / _MANIFEST_NAME, manifest)
        _write_json(temporary_root / _PROVENANCE_NAME, provenance)
        if ontology_root.exists():
            raise OntologyPersistenceError(f"Ontology context already exists: {ontology_root}")
        temporary_root.rename(ontology_root)
    except OntologyContextError:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    except (OSError, TypeError, ValueError) as exc:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise OntologyPersistenceError("Ontology context initialization failed.") from exc


def _snapshot_from_graph(
    interaction_root: Path,
    graph: Graph,
    manifest: Mapping[str, object],
) -> ABoxSnapshot:
    ontology_root = interaction_root / _ONTOLOGY_ROOT
    return ABoxSnapshot(
        graph=graph,
        interaction_root=interaction_root,
        ontology_root=ontology_root,
        abox_path=ontology_root / _ABOX_NAME,
        manifest_path=ontology_root / _MANIFEST_NAME,
        provenance_path=ontology_root / _PROVENANCE_NAME,
        namespace=str(manifest["interaction_namespace"]),
        specification_iri=str(manifest["specification_iri"]),
        tbox_fingerprint=str(manifest["tbox_fingerprint"]),
        product_requirement=str(manifest["product_requirement"]),
        delta_count=int(manifest["delta_count"]),
        accepted_assertion_count=int(manifest["accepted_assertion_count"]),
    )


def _coerce_delta(delta: TripleDelta | Mapping[str, object]) -> TripleDelta:
    if isinstance(delta, TripleDelta):
        delta = {
            "assertions": delta.assertions,
            "uncertainty": delta.uncertainty,
            "unresolved_evidence_needs": delta.unresolved_evidence_needs,
            "typed_context_refs": delta.typed_context_refs,
        }
    if not isinstance(delta, Mapping) or not set(delta).issubset(_DELTA_KEYS):
        raise TripleDeltaError("Delta contains fields outside the generic triple-delta contract.")
    if "assertions" not in delta:
        raise TripleDeltaError("Delta must contain assertions.")
    assertions_value = _object_sequence(delta["assertions"], "assertions")
    assertions = tuple(
        _coerce_assertion(value, index) for index, value in enumerate(assertions_value)
    )
    uncertainty = tuple(_object_sequence(delta.get("uncertainty", []), "uncertainty"))
    unresolved = tuple(
        _object_sequence(
            delta.get("unresolved_evidence_needs", []),
            "unresolved_evidence_needs",
        )
    )
    for label, values in (
        ("uncertainty", uncertainty),
        ("unresolved_evidence_needs", unresolved),
    ):
        for value in values:
            _require_json_value(value, label)
    return TripleDelta(
        assertions=assertions,
        uncertainty=uncertainty,
        unresolved_evidence_needs=unresolved,
        typed_context_refs=_string_tuple(
            delta.get("typed_context_refs", []),
            "typed_context_refs",
            TripleDeltaError,
        ),
    )


def _coerce_assertion(value: object, index: int) -> TripleAssertion:
    if isinstance(value, TripleAssertion):
        triple_object: object = value.object
        if isinstance(triple_object, TripleObject):
            object_record: dict[str, object] = {
                "kind": triple_object.kind,
                "value": triple_object.value,
            }
            if triple_object.datatype is not None:
                object_record["datatype"] = triple_object.datatype
            if triple_object.language is not None:
                object_record["language"] = triple_object.language
            triple_object = object_record
        value = {
            "subject": value.subject,
            "predicate": value.predicate,
            "object": triple_object,
            "evidence_refs": value.evidence_refs,
        }
    if not isinstance(value, Mapping) or set(value) != _ASSERTION_KEYS:
        raise TripleDeltaError(f"assertions[{index}] fields do not match the assertion contract.")
    object_value = value["object"]
    if not isinstance(object_value, Mapping):
        raise TripleDeltaError(f"assertions[{index}].object must be an object.")
    kind = object_value.get("kind")
    allowed_keys = {"kind", "value"} if kind == "iri" else {"kind", "value", "datatype", "language"}
    if "value" not in object_value or not set(object_value).issubset(allowed_keys):
        raise TripleDeltaError(f"assertions[{index}].object fields are invalid.")
    if kind not in {"iri", "literal"}:
        raise TripleDeltaError(f"assertions[{index}].object.kind must be iri or literal.")
    datatype = object_value.get("datatype")
    language = object_value.get("language")
    if datatype is not None and not isinstance(datatype, str):
        raise TripleDeltaError(f"assertions[{index}].object.datatype is invalid.")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise TripleDeltaError(f"assertions[{index}].object.language is invalid.")
    if datatype is not None and language is not None:
        raise TripleDeltaError(
            f"assertions[{index}].object cannot use datatype and language together."
        )
    return TripleAssertion(
        subject=_require_nonempty_string(
            value["subject"],
            f"assertions[{index}].subject",
            TripleDeltaError,
        ),
        predicate=_require_nonempty_string(
            value["predicate"],
            f"assertions[{index}].predicate",
            TripleDeltaError,
        ),
        object=TripleObject(
            kind=kind,
            value=_validated_object_scalar(object_value["value"], index),
            datatype=datatype,
            language=language,
        ),
        evidence_refs=_string_tuple(
            value["evidence_refs"],
            f"assertions[{index}].evidence_refs",
            TripleDeltaError,
        ),
    )


def _validated_authorized_refs(values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TripleDeltaError("authorized_evidence_refs must be an iterable of refs.")
    refs: list[str] = []
    for value in values:
        ref = _require_nonempty_string(
            value,
            "authorized_evidence_refs entry",
            TripleDeltaError,
        )
        refs.append(ref)
    if len(refs) != len(set(refs)):
        raise TripleDeltaError("authorized_evidence_refs contains duplicates.")
    return frozenset(refs)


def _validate_delta_metadata(
    delta: TripleDelta,
) -> None:
    if not any(
        (
            delta.assertions,
            delta.uncertainty,
            delta.unresolved_evidence_needs,
            delta.typed_context_refs,
        )
    ):
        raise TripleDeltaError("Delta must contain a result or unresolved record.")
    for context_ref in delta.typed_context_refs:
        _validate_typed_context_ref(context_ref)


def _validated_assertion(
    assertion: TripleAssertion,
    abox: ABoxSnapshot,
    tbox: TBoxSnapshot,
    authorized_refs: frozenset[str],
) -> tuple[tuple[URIRef, URIRef, URIRef | Literal], dict[str, object]]:
    _raise_if_primitive_symbol(assertion.subject, TripleDeltaError)
    _raise_if_prohibited_property(assertion.predicate, TripleDeltaError)
    _require_absolute_iri(assertion.subject, "assertion subject", TripleDeltaError)
    _require_absolute_iri(assertion.predicate, "assertion predicate", TripleDeltaError)
    if not assertion.subject.startswith(abox.namespace):
        raise TripleDeltaError(
            "Assertion subject must belong to the current interaction namespace."
        )
    if not assertion.evidence_refs:
        raise TripleDeltaError("Every factual assertion requires evidence_refs.")
    for evidence_ref in assertion.evidence_refs:
        if evidence_ref not in authorized_refs:
            raise TripleDeltaError(f"Assertion evidence_ref is not authorized: {evidence_ref}")

    predicate = URIRef(assertion.predicate)
    if predicate == RDF.type:
        object_node, object_record = _validated_type_object(assertion, tbox)
    elif assertion.predicate in tbox.object_properties:
        object_node, object_record = _validated_iri_object(assertion, abox)
    elif assertion.predicate in tbox.datatype_properties:
        object_node, object_record = _validated_literal_object(assertion)
    else:
        raise TripleDeltaError(
            f"Assertion predicate is not in the immutable vocabulary: {predicate}"
        )

    return (
        (URIRef(assertion.subject), predicate, object_node),
        {
            "subject": assertion.subject,
            "predicate": assertion.predicate,
            "object": object_record,
            "evidence_refs": list(assertion.evidence_refs),
        },
    )


def _validated_type_object(
    assertion: TripleAssertion,
    tbox: TBoxSnapshot,
) -> tuple[URIRef, dict[str, object]]:
    if assertion.object.kind != "iri" or not isinstance(assertion.object.value, str):
        raise TripleDeltaError("rdf:type object must be a declared class IRI.")
    class_iri = assertion.object.value
    _raise_if_primitive_symbol(class_iri, TripleDeltaError)
    _require_absolute_iri(class_iri, "rdf:type object", TripleDeltaError)
    if class_iri not in tbox.classes:
        raise TripleDeltaError(f"rdf:type class is not declared: {class_iri}")
    class_ref = URIRef(class_iri)
    if tbox.is_class_or_subclass(
        class_ref,
        URIRef(f"{tbox.ppr_namespace}resource"),
    ) or tbox.is_class_or_subclass(
        class_ref,
        URIRef(f"{tbox.ppr_namespace}capability"),
    ):
        raise TripleDeltaError("PA ABox cannot contain resource or capability individuals.")
    if _class_is_primitive(tbox, class_ref):
        raise TripleDeltaError("PA ABox cannot contain primitive individuals.")
    return class_ref, {"kind": "iri", "value": class_iri}


def _validated_iri_object(
    assertion: TripleAssertion,
    abox: ABoxSnapshot,
) -> tuple[URIRef, dict[str, object]]:
    if assertion.object.kind != "iri" or not isinstance(assertion.object.value, str):
        raise TripleDeltaError("Object property assertion requires an IRI object.")
    object_iri = assertion.object.value
    _raise_if_primitive_symbol(object_iri, TripleDeltaError)
    _require_absolute_iri(object_iri, "assertion object", TripleDeltaError)
    if not object_iri.startswith(abox.namespace):
        raise TripleDeltaError(
            "Object property target must belong to the current interaction namespace."
        )
    return URIRef(object_iri), {"kind": "iri", "value": object_iri}


def _validated_literal_object(
    assertion: TripleAssertion,
) -> tuple[Literal, dict[str, object]]:
    if assertion.object.kind != "literal":
        raise TripleDeltaError("Datatype property assertion requires a literal object.")
    datatype = assertion.object.datatype
    if datatype is not None:
        _require_absolute_iri(datatype, "literal datatype", TripleDeltaError)
    if assertion.object.language is not None and not isinstance(assertion.object.value, str):
        raise TripleDeltaError("Language-tagged literal value must be a string.")
    record: dict[str, object] = {
        "kind": "literal",
        "value": assertion.object.value,
    }
    if datatype is not None:
        record["datatype"] = datatype
    if assertion.object.language is not None:
        record["language"] = assertion.object.language
    return (
        Literal(
            assertion.object.value,
            datatype=URIRef(datatype) if datatype is not None else None,
            lang=assertion.object.language,
        ),
        record,
    )


def _validate_pa_abox_graph(
    graph: Graph,
    manifest: Mapping[str, object],
    tbox: TBoxSnapshot,
) -> None:
    namespace = str(manifest["interaction_namespace"])
    specification = URIRef(str(manifest["specification_iri"]))
    ppr = Namespace(tbox.ppr_namespace)
    if (specification, RDF.type, ppr.specification) not in graph:
        raise OntologyPersistenceError("ABox is missing its specification type.")
    requirement = str(manifest["product_requirement"])
    if list(graph.objects(specification, RDF.value)) != [Literal(requirement)]:
        raise OntologyPersistenceError(
            "ABox does not preserve the exact unresolved product requirement."
        )

    for triple in graph:
        _validate_pa_abox_triple(
            triple,
            namespace=namespace,
            specification=specification,
            requirement=requirement,
            tbox=tbox,
        )
    _validate_semantic_bridge(graph, specification, tbox)


def _validate_pa_abox_triple(
    triple: tuple[object, object, object],
    *,
    namespace: str,
    specification: URIRef,
    requirement: str,
    tbox: TBoxSnapshot,
) -> None:
    subject, predicate, object_node = triple
    ppr = Namespace(tbox.ppr_namespace)
    if not isinstance(subject, URIRef) or not str(subject).startswith(namespace):
        raise TripleDeltaError("ABox subjects must belong to the interaction namespace.")
    if not isinstance(predicate, URIRef):
        raise TripleDeltaError("ABox predicates must be IRIs.")
    _raise_if_primitive_symbol(str(subject), TripleDeltaError)
    _raise_if_prohibited_property(str(predicate), TripleDeltaError)

    if predicate == RDF.type:
        if not isinstance(object_node, URIRef) or str(object_node) not in tbox.classes:
            raise TripleDeltaError("ABox rdf:type object is not a declared class.")
        if tbox.is_class_or_subclass(
            object_node,
            ppr.resource,
        ) or tbox.is_class_or_subclass(
            object_node,
            ppr.capability,
        ):
            raise TripleDeltaError("PA ABox cannot contain resource or capability individuals.")
        if _class_is_primitive(tbox, object_node):
            raise TripleDeltaError("PA ABox cannot contain primitive individuals.")
        return
    if predicate == RDF.value:
        if subject != specification or object_node != Literal(requirement):
            raise TripleDeltaError("rdf:value is reserved for the exact unresolved specification.")
        return
    if str(predicate) in tbox.object_properties:
        if _property_implies_resource_catalog_fact(tbox, predicate):
            raise TripleDeltaError("PA ABox cannot contain resource-catalog assertions.")
        if not isinstance(object_node, URIRef) or not str(object_node).startswith(namespace):
            raise TripleDeltaError("ABox object property target must be a runtime interaction IRI.")
        _raise_if_primitive_symbol(str(object_node), TripleDeltaError)
        return
    if str(predicate) in tbox.datatype_properties:
        if _property_implies_resource_catalog_fact(tbox, predicate):
            raise TripleDeltaError("PA ABox cannot contain resource-catalog assertions.")
        if not isinstance(object_node, Literal):
            raise TripleDeltaError("ABox datatype property target must be a literal.")
        return
    raise TripleDeltaError(f"ABox predicate is outside the immutable vocabulary: {predicate}")


def _validate_semantic_bridge(
    graph: Graph,
    specification: URIRef,
    tbox: TBoxSnapshot,
) -> None:
    ppr = Namespace(tbox.ppr_namespace)
    specification_instances = set(graph.subjects(RDF.type, ppr.specification))
    if specification_instances != {specification}:
        raise TripleDeltaError(
            "The interaction ABox must contain exactly its initialized specification."
        )
    for specification_type in graph.objects(specification, RDF.type):
        if not isinstance(specification_type, URIRef) or not tbox.is_class_or_subclass(
            specification_type,
            ppr.specification,
        ):
            raise TripleDeltaError("The initialized specification cannot have an unrelated type.")
    defined_features: set[URIRef] = set()
    for defining_specification, defined_feature in graph.subject_objects(ppr.defines):
        if defining_specification != specification:
            raise TripleDeltaError("defines subject must be the initialized specification.")
        if not isinstance(defined_feature, URIRef) or not _instance_is_type(
            graph,
            defined_feature,
            ppr.feature,
            tbox,
        ):
            raise TripleDeltaError(
                "defines must target an interaction individual typed as feature."
            )
        defined_features.add(defined_feature)
    for process, realized_feature in graph.subject_objects(ppr.realizes):
        if not _instance_is_type(graph, process, ppr.process, tbox):
            raise TripleDeltaError("realizes subject must be typed as process.")
        if not isinstance(realized_feature, URIRef) or not _instance_is_type(
            graph,
            realized_feature,
            ppr.feature,
            tbox,
        ):
            raise TripleDeltaError("realizes object must be typed as feature.")
        if realized_feature not in defined_features:
            raise TripleDeltaError(
                "Every realized feature must be defined by the current specification."
            )
    for subject, predicate, _object_node in graph:
        if (
            _local_name(str(predicate)) == "consistsOf"
            and isinstance(subject, URIRef)
            and _instance_is_type(graph, subject, ppr.process, tbox)
        ):
            raise TripleDeltaError("PA ABox cannot encode process composition recipes.")


def _load_persisted_abox(
    interaction_root: Path,
    tbox: TBoxSnapshot,
) -> tuple[ABoxSnapshot, dict[str, object], dict[str, object]]:
    ontology_root = interaction_root / _ONTOLOGY_ROOT
    manifest_path = ontology_root / _MANIFEST_NAME
    provenance_path = ontology_root / _PROVENANCE_NAME
    abox_path = ontology_root / _ABOX_NAME
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance_value = json.loads(provenance_path.read_text(encoding="utf-8"))
        graph = Graph()
        graph.parse(source=str(abox_path), format="turtle")
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        SyntaxError,
        ParserError,
        PluginException,
    ) as exc:
        raise OntologyPersistenceError("Persisted ontology context could not be read.") from exc
    if not isinstance(manifest_value, dict) or set(manifest_value) != _MANIFEST_KEYS:
        raise OntologyPersistenceError("ABox manifest shape is invalid.")
    manifest: dict[str, object] = manifest_value
    _validate_manifest(manifest, tbox, ontology_root)
    if (
        not isinstance(provenance_value, dict)
        or set(provenance_value) != _PROVENANCE_KEYS
        or provenance_value.get("schema_version") != _SCHEMA_VERSION
        or provenance_value.get("tbox_fingerprint") != tbox.fingerprint
        or not isinstance(provenance_value.get("assertions"), list)
    ):
        raise OntologyPersistenceError("Assertion provenance shape is invalid.")
    provenance: dict[str, object] = provenance_value
    expected_provenance_count = 2 + int(manifest["accepted_assertion_count"])
    if len(provenance["assertions"]) != expected_provenance_count:
        raise OntologyPersistenceError("Assertion provenance count is invalid.")
    _validate_pa_abox_graph(graph, manifest, tbox)
    return _snapshot_from_graph(interaction_root, graph, manifest), manifest, provenance


def _validate_manifest(
    manifest: Mapping[str, object],
    tbox: TBoxSnapshot,
    ontology_root: Path,
) -> None:
    if manifest["schema_version"] != _SCHEMA_VERSION:
        raise OntologyPersistenceError("Unsupported ABox manifest schema version.")
    if manifest["status"] != "unresolved":
        raise OntologyPersistenceError("Phase 4.0 ABox status must remain unresolved.")
    if manifest["tbox_fingerprint"] != tbox.fingerprint:
        raise OntologyPersistenceError("ABox TBox fingerprint does not match.")
    namespace = manifest["interaction_namespace"]
    specification_iri = manifest["specification_iri"]
    product_requirement = manifest["product_requirement"]
    if not isinstance(namespace, str) or not namespace.endswith("/"):
        raise OntologyPersistenceError("ABox interaction namespace is invalid.")
    _require_absolute_iri(namespace, "interaction namespace", OntologyPersistenceError)
    if not isinstance(specification_iri, str) or not specification_iri.startswith(namespace):
        raise OntologyPersistenceError("ABox specification IRI is invalid.")
    if not isinstance(product_requirement, str) or not product_requirement.strip():
        raise OntologyPersistenceError("ABox product requirement is invalid.")
    for key in ("delta_count", "accepted_assertion_count"):
        value = manifest[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise OntologyPersistenceError(f"ABox manifest {key} is invalid.")
    delta_count = int(manifest["delta_count"])
    expected_delta_names = {f"delta_{number:04d}.json" for number in range(1, delta_count + 1)}
    actual_delta_names = {path.name for path in ontology_root.glob("delta_*.json")}
    if actual_delta_names != expected_delta_names:
        raise OntologyPersistenceError("Persisted delta sequence is invalid.")


def _persist_merge(
    abox: ABoxSnapshot,
    graph: Graph,
    manifest: Mapping[str, object],
    provenance: Mapping[str, object],
    delta_path: Path,
    delta_record: Mapping[str, object],
) -> None:
    try:
        temporary_root = Path(tempfile.mkdtemp(prefix=".merge-", dir=abox.ontology_root))
    except OSError as exc:
        raise OntologyPersistenceError("Merge staging directory failed.") from exc
    try:
        staged_abox = temporary_root / _ABOX_NAME
        staged_manifest = temporary_root / _MANIFEST_NAME
        staged_provenance = temporary_root / _PROVENANCE_NAME
        staged_delta = temporary_root / delta_path.name
        graph.serialize(destination=staged_abox, format="turtle")
        _write_json(staged_manifest, manifest)
        _write_json(staged_provenance, provenance)
        _write_json(staged_delta, delta_record)
        os.replace(staged_abox, abox.abox_path)
        os.replace(staged_provenance, abox.provenance_path)
        os.replace(staged_manifest, abox.manifest_path)
        if delta_path.exists():
            raise OntologyPersistenceError(f"Delta record already exists: {delta_path.name}")
        os.replace(staged_delta, delta_path)
    except OntologyContextError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise OntologyPersistenceError("Validated delta persistence failed.") from exc
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _write_json(path: Path, value: object) -> None:
    serialized = json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.write_text(serialized + "\n", encoding="utf-8")


def _instance_is_type(
    abox_graph: Graph,
    instance: URIRef,
    target_class: URIRef,
    tbox: TBoxSnapshot,
) -> bool:
    return any(
        isinstance(actual_class, URIRef) and tbox.is_class_or_subclass(actual_class, target_class)
        for actual_class in abox_graph.objects(instance, RDF.type)
    )


def _class_is_primitive(tbox: TBoxSnapshot, candidate_class: URIRef) -> bool:
    primitive_roots = (
        URIRef(class_iri)
        for class_iri in tbox.classes
        if _local_name(class_iri).startswith("primitive")
    )
    return any(
        tbox.is_class_or_subclass(candidate_class, primitive_root)
        for primitive_root in primitive_roots
    )


def _property_implies_resource_catalog_fact(
    tbox: TBoxSnapshot,
    property_iri: URIRef,
) -> bool:
    ppr = Namespace(tbox.ppr_namespace)
    prohibited_roots = (ppr.resource, ppr.capability)
    declared_types = (
        class_iri
        for relation in (RDFS.domain, RDFS.range)
        for class_iri in tbox.graph.objects(property_iri, relation)
        if isinstance(class_iri, URIRef)
    )
    return any(
        tbox.is_class_or_subclass(declared_type, prohibited_root)
        for declared_type in declared_types
        for prohibited_root in prohibited_roots
    )


def _string_tuple(
    value: object,
    label: str,
    error_type: type[OntologyContextError],
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise error_type(f"{label} must be an array of strings.")
    values = tuple(_require_nonempty_string(item, f"{label} entry", error_type) for item in value)
    if len(values) != len(set(values)):
        raise error_type(f"{label} contains duplicate fixed symbols.")
    return values


def _object_sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TripleDeltaError(f"{label} must be an array.")
    return value


def _validated_object_scalar(value: object, index: int) -> str | int | float | bool:
    if not isinstance(value, (str, int, float, bool)) or value is None:
        raise TripleDeltaError(f"assertions[{index}].object.value must be a JSON scalar.")
    if isinstance(value, float) and not math.isfinite(value):
        raise TripleDeltaError(f"assertions[{index}].object.value must be finite.")
    return value


def _require_json_value(value: object, label: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TripleDeltaError(f"{label} must contain JSON values.") from exc


def _require_nonempty_string(
    value: object,
    label: str,
    error_type: type[OntologyContextError],
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{label} must be a non-empty string.")
    return value


def _require_absolute_iri(
    value: str,
    label: str,
    error_type: type[OntologyContextError],
) -> None:
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise error_type(f"{label} must be an absolute IRI without whitespace.")
    if not urlsplit(value).scheme:
        raise error_type(f"{label} must be an absolute IRI.")


def _local_name(value: str) -> str:
    fragment = value.rsplit("#", maxsplit=1)[-1]
    path_part = fragment.rsplit("/", maxsplit=1)[-1]
    return path_part.rsplit(":", maxsplit=1)[-1]


def _raise_if_prohibited_property(
    value: str,
    error_type: type[OntologyContextError],
) -> None:
    if _local_name(value) in _PROHIBITED_PA_PROPERTIES:
        raise error_type(f"PA ABox property is prohibited in Phase 4.0: {value}")


def _raise_if_primitive_symbol(
    value: str,
    error_type: type[OntologyContextError],
) -> None:
    local_name = _local_name(value)
    if local_name.startswith("primitive"):
        raise error_type(f"PA ABox cannot contain primitive assertion: {value}")


def _validate_typed_context_ref(value: str) -> None:
    relative_path = Path(value)
    if (
        not value.startswith(_TYPED_CONTEXT_PREFIX)
        or not value.endswith(".json")
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or "\\" in value
    ):
        raise TripleDeltaError(
            "typed_context_ref must name a JSON record under products/grounding/."
        )
