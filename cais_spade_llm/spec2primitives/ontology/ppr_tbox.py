"""Load and validate the shared immutable PPR TBox."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from xml.sax import SAXException

from rdflib import Graph, Literal, URIRef
from rdflib.compare import to_isomorphic
from rdflib.exceptions import ParserError
from rdflib.namespace import OWL, RDF, RDFS
from rdflib.plugin import PluginException

_REQUIRED_CLASSES = (
    "specification",
    "product",
    "feature",
    "state",
    "process",
    "processExecution",
    "resource",
    "capability",
    "Assembly",
    "Part",
    "AssemblyFeature",
    "AssemblyFeatureAssociation",
)
_REQUIRED_OBJECT_PROPERTIES = (
    "defines",
    "realizes",
    "hascurrentstate",
    "hasdesiredstate",
    "capableOf",
    "hasProcessExecution",
    "runsProcess",
    "runsOnResource",
    "hasPart",
    "hasAssemblyFeature",
    "hasAssemblyFeatureAssociation",
    "relatesAssemblyFeature",
)


class OntologyContextError(ValueError):
    """Raised when an ontology-context operation violates its profile."""


class TBoxLoadError(OntologyContextError):
    """Raised when the caller-supplied TBox cannot be parsed."""


class TBoxProfileError(OntologyContextError):
    """Raised when a parsed graph is not an approved schema-only TBox."""


@dataclass(frozen=True)
class TBoxSnapshot:
    """Validated immutable-by-contract PPR TBox snapshot."""

    graph: Graph = field(repr=False, compare=False)
    source_path: Path
    ppr_namespace: str
    fingerprint: str
    classes: frozenset[str]
    object_properties: frozenset[str]
    datatype_properties: frozenset[str]

    def assert_unchanged(self) -> None:
        """Raise if the validated graph has changed since it was loaded."""
        if _graph_fingerprint(self.graph) != self.fingerprint:
            raise TBoxProfileError("TBoxSnapshot graph changed after validation.")

    def is_class_or_subclass(
        self,
        candidate_class: str | URIRef,
        target_class: str | URIRef,
    ) -> bool:
        """Return whether a class equals or specializes another named class."""
        self.assert_unchanged()
        candidate = URIRef(candidate_class)
        target = URIRef(target_class)
        pending = [candidate]
        visited: set[URIRef] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(
                superclass
                for superclass in (
                    *self.graph.objects(current, RDFS.subClassOf),
                    *self.graph.objects(current, OWL.equivalentClass),
                    *self.graph.subjects(OWL.equivalentClass, current),
                )
                if isinstance(superclass, URIRef)
            )
        return False


def load_ppr_tbox(tbox_path: Path, *, ppr_namespace: str) -> TBoxSnapshot:
    """Parse and validate a caller-supplied schema-only PPR TBox.

    Args:
        tbox_path: RDF/XML, Turtle, or another RDFLib-supported RDF document.
        ppr_namespace: Exact namespace containing the required PPR symbols.

    Returns:
        A validated TBox snapshot with a stable graph fingerprint.

    Raises:
        TBoxLoadError: If the source cannot be parsed.
        TBoxProfileError: If the source is not a schema-only PPR graph.
    """
    source_path = Path(tbox_path).resolve()
    namespace = _validated_namespace(ppr_namespace)
    if not source_path.is_file():
        raise TBoxLoadError(f"TBox source is not a file: {source_path}")

    graph = Graph()
    try:
        graph.parse(source=str(source_path))
    except (OSError, ValueError, SyntaxError, SAXException, ParserError, PluginException) as exc:
        raise TBoxLoadError(f"TBox could not be parsed: {source_path}") from exc

    classes = _declared_iris(graph, (OWL.Class, RDFS.Class))
    object_properties = _declared_iris(graph, (OWL.ObjectProperty,))
    datatype_properties = _declared_iris(graph, (OWL.DatatypeProperty,))
    _validate_required_symbols(
        namespace,
        classes=classes,
        object_properties=object_properties,
    )
    _validate_feature_state_slice(graph, namespace)
    _validate_process_execution_slice(graph, namespace)
    _validate_assembly_slice(graph, namespace)
    _validate_schema_only_profile(
        graph,
        namespace,
        classes=classes,
        object_properties=object_properties,
        datatype_properties=datatype_properties,
    )
    return TBoxSnapshot(
        graph=graph,
        source_path=source_path,
        ppr_namespace=namespace,
        fingerprint=_graph_fingerprint(graph),
        classes=frozenset(str(value) for value in classes),
        object_properties=frozenset(str(value) for value in object_properties),
        datatype_properties=frozenset(str(value) for value in datatype_properties),
    )


def _validated_namespace(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TBoxProfileError("ppr_namespace must be an exact non-empty IRI.")
    _require_absolute_iri(value, "ppr_namespace")
    if not value.endswith(("#", "/")):
        raise TBoxProfileError("ppr_namespace must end with '#' or '/'.")
    return value


def _declared_iris(graph: Graph, declarations: Sequence[URIRef]) -> set[URIRef]:
    return {
        subject
        for declaration in declarations
        for subject in graph.subjects(RDF.type, declaration)
        if isinstance(subject, URIRef)
    }


def _validate_required_symbols(
    namespace: str,
    *,
    classes: set[URIRef],
    object_properties: set[URIRef],
) -> None:
    missing_classes = [
        value for value in _REQUIRED_CLASSES if URIRef(f"{namespace}{value}") not in classes
    ]
    missing_properties = [
        value
        for value in _REQUIRED_OBJECT_PROPERTIES
        if URIRef(f"{namespace}{value}") not in object_properties
    ]
    if missing_classes or missing_properties:
        details: list[str] = []
        if missing_classes:
            details.append(f"classes={missing_classes}")
        if missing_properties:
            details.append(f"object_properties={missing_properties}")
        raise TBoxProfileError("TBox is missing required PPR symbols: " + ", ".join(details))


def _validate_process_execution_slice(graph: Graph, namespace: str) -> None:
    process_execution = URIRef(f"{namespace}processExecution")
    expected_profiles = {
        URIRef(f"{namespace}hasProcessExecution"): (
            URIRef(f"{namespace}specification"),
            process_execution,
            False,
        ),
        URIRef(f"{namespace}runsProcess"): (
            process_execution,
            URIRef(f"{namespace}process"),
            True,
        ),
        URIRef(f"{namespace}runsOnResource"): (
            process_execution,
            URIRef(f"{namespace}resource"),
            True,
        ),
    }
    for property_iri, profile in expected_profiles.items():
        expected_domain, expected_range, expected_functional = profile
        if set(graph.objects(property_iri, RDFS.domain)) != {expected_domain}:
            raise TBoxProfileError(f"TBox property has an invalid domain: {property_iri}")
        if set(graph.objects(property_iri, RDFS.range)) != {expected_range}:
            raise TBoxProfileError(f"TBox property has an invalid range: {property_iri}")
        is_functional = (property_iri, RDF.type, OWL.FunctionalProperty) in graph
        if is_functional != expected_functional:
            raise TBoxProfileError(
                f"TBox property has an invalid functional profile: {property_iri}"
            )


def _validate_feature_state_slice(graph: Graph, namespace: str) -> None:
    """Require exact feature-owned current and desired state properties."""
    feature = URIRef(f"{namespace}feature")
    state = URIRef(f"{namespace}state")
    for symbol in ("hascurrentstate", "hasdesiredstate"):
        property_iri = URIRef(f"{namespace}{symbol}")
        if set(graph.objects(property_iri, RDFS.domain)) != {feature}:
            raise TBoxProfileError(f"TBox property has an invalid domain: {property_iri}")
        if set(graph.objects(property_iri, RDFS.range)) != {state}:
            raise TBoxProfileError(f"TBox property has an invalid range: {property_iri}")
        if (property_iri, RDF.type, OWL.FunctionalProperty) not in graph:
            raise TBoxProfileError(
                f"TBox property has an invalid functional profile: {property_iri}"
            )


def _validate_assembly_slice(graph: Graph, namespace: str) -> None:
    """Require the exact lean PPR assembly-extension profile."""
    product = URIRef(f"{namespace}product")
    feature = URIRef(f"{namespace}feature")
    assembly = URIRef(f"{namespace}Assembly")
    part = URIRef(f"{namespace}Part")
    assembly_feature = URIRef(f"{namespace}AssemblyFeature")
    association = URIRef(f"{namespace}AssemblyFeatureAssociation")
    expected_superclasses = {
        assembly: product,
        part: product,
        assembly_feature: feature,
        association: feature,
    }
    for class_iri, superclass_iri in expected_superclasses.items():
        if superclass_iri not in set(graph.objects(class_iri, RDFS.subClassOf)):
            raise TBoxProfileError(
                f"TBox assembly class has an invalid superclass: {class_iri}"
            )

    expected_properties = {
        URIRef(f"{namespace}hasPart"): (assembly, product),
        URIRef(f"{namespace}hasAssemblyFeature"): (product, assembly_feature),
        URIRef(f"{namespace}hasAssemblyFeatureAssociation"): (assembly, association),
        URIRef(f"{namespace}relatesAssemblyFeature"): (association, assembly_feature),
    }
    for property_iri, (expected_domain, expected_range) in expected_properties.items():
        if set(graph.objects(property_iri, RDFS.domain)) != {expected_domain}:
            raise TBoxProfileError(
                f"TBox assembly property has an invalid domain: {property_iri}"
            )
        if set(graph.objects(property_iri, RDFS.range)) != {expected_range}:
            raise TBoxProfileError(
                f"TBox assembly property has an invalid range: {property_iri}"
            )
        if (property_iri, RDF.type, OWL.FunctionalProperty) in graph:
            raise TBoxProfileError(
                f"TBox assembly property cannot be functional: {property_iri}"
            )

    relates = URIRef(f"{namespace}relatesAssemblyFeature")
    matching_restrictions = []
    for restriction in graph.objects(association, RDFS.subClassOf):
        cardinalities = list(graph.objects(restriction, OWL.qualifiedCardinality))
        if (
            (restriction, RDF.type, OWL.Restriction) in graph
            and (restriction, OWL.onProperty, relates) in graph
            and (restriction, OWL.onClass, assembly_feature) in graph
            and len(cardinalities) == 1
            and cardinalities[0].toPython() == 2
        ):
            matching_restrictions.append(restriction)
    if len(matching_restrictions) != 1:
        raise TBoxProfileError(
            "TBox AssemblyFeatureAssociation must relate exactly two AssemblyFeature values."
        )


def _validate_schema_only_profile(
    graph: Graph,
    namespace: str,
    *,
    classes: set[URIRef],
    object_properties: set[URIRef],
    datatype_properties: set[URIRef],
) -> None:
    _validate_no_individual_constructs(graph)
    _validate_no_recipe_axioms(graph, namespace)
    _validate_schema_assertions(
        graph,
        classes=classes,
        object_properties=object_properties,
        datatype_properties=datatype_properties,
    )


def _validate_no_individual_constructs(graph: Graph) -> None:
    if any(graph.subjects(RDF.type, OWL.NamedIndividual)):
        raise TBoxProfileError("Production TBox contains named individuals.")
    if any(graph.triples((None, RDF.value, None))):
        raise TBoxProfileError("Production TBox contains an RDF value assertion.")
    if any(not isinstance(value, Literal) for value in graph.objects(None, OWL.hasValue)):
        raise TBoxProfileError("Production TBox contains an individual-valued restriction.")
    for list_head in graph.objects(None, OWL.oneOf):
        try:
            members = tuple(graph.items(list_head))
        except ValueError as exc:
            raise TBoxProfileError("Production TBox contains an invalid enumeration.") from exc
        if any(not isinstance(member, Literal) for member in members):
            raise TBoxProfileError("Production TBox contains an individual enumeration.")


def _validate_no_recipe_axioms(graph: Graph, namespace: str) -> None:
    recipe_properties = {
        URIRef(f"{namespace}requires"),
        URIRef(f"{namespace}precedes"),
    }
    for recipe_property in recipe_properties:
        if any(graph.subjects(OWL.onProperty, recipe_property)):
            raise TBoxProfileError(
                "Production TBox contains a requires/precedes recipe restriction."
            )
        if any(graph.triples((None, recipe_property, None))):
            raise TBoxProfileError("Production TBox contains a requires/precedes task assertion.")
    for chain_head in graph.objects(None, OWL.propertyChainAxiom):
        try:
            chain_members = set(graph.items(chain_head))
        except ValueError as exc:
            raise TBoxProfileError("Production TBox contains an invalid property chain.") from exc
        if recipe_properties.intersection(chain_members):
            raise TBoxProfileError("Production TBox contains a requires/precedes property recipe.")


def _validate_schema_assertions(
    graph: Graph,
    *,
    classes: set[URIRef],
    object_properties: set[URIRef],
    datatype_properties: set[URIRef],
) -> None:
    annotation_properties = _declared_iris(graph, (OWL.AnnotationProperty,))
    instance_properties = (
        object_properties
        | datatype_properties
        | _declared_iris(
            graph,
            (RDF.Property,),
        )
    )
    declared_properties = instance_properties | annotation_properties
    schema_entities = (
        classes
        | declared_properties
        | _declared_iris(graph, (RDFS.Datatype,))
        | set(graph.subjects(RDF.type, OWL.Ontology))
    )
    for subject, instance_class in graph.subject_objects(RDF.type):
        if instance_class in classes or instance_class == OWL.NamedIndividual:
            raise TBoxProfileError("Production TBox contains an RDF instance assertion.")
        if isinstance(subject, URIRef) and subject not in schema_entities:
            raise TBoxProfileError("Production TBox contains an implicit named individual.")
    for property_iri in instance_properties:
        if any(graph.triples((None, property_iri, None))):
            raise TBoxProfileError("Production TBox contains a property instance assertion.")
    for subject, predicate, _ in graph:
        if isinstance(subject, URIRef) and subject not in schema_entities:
            raise TBoxProfileError("Production TBox contains an undeclared named instance.")
        if not _is_schema_predicate(predicate, annotation_properties):
            raise TBoxProfileError("Production TBox contains a non-schema property assertion.")


def _graph_fingerprint(graph: Graph) -> str:
    digest = to_isomorphic(graph).graph_digest()
    return hashlib.sha256(str(digest).encode("ascii")).hexdigest()


def _is_schema_predicate(
    predicate: URIRef,
    annotation_properties: set[URIRef],
) -> bool:
    predicate_value = str(predicate)
    return predicate in annotation_properties or predicate_value.startswith(
        (str(RDF), str(RDFS), str(OWL))
    )


def _require_absolute_iri(value: str, label: str) -> None:
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise TBoxProfileError(f"{label} must be an absolute IRI without whitespace.")
    if not urlsplit(value).scheme:
        raise TBoxProfileError(f"{label} must be an absolute IRI.")
