"""Map directly supported PA statements through the authoritative ABox validator."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdflib import RDF, RDFS, URIRef

from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingSession,
    GroundingStatement,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    MergeResult,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.ontology import TBoxSnapshot

_PRODUCER = "ontology_grounding"
_PROPOSAL_ROOT = Path("products/grounding/ontology_grounding")
_PROPOSAL_KEYS = {
    "individuals",
    "relations",
    "literal_facts",
    "unrepresented_statement_ids",
}
_INDIVIDUAL_KEYS = {"individual_index", "class_iri", "statement_ids"}
_RELATION_KEYS = {
    "subject_kind",
    "subject_individual_index",
    "subject_iri",
    "predicate_iri",
    "object_kind",
    "object_individual_index",
    "object_iri",
    "statement_ids",
}
_LITERAL_KEYS = {
    "subject_kind",
    "subject_individual_index",
    "subject_iri",
    "predicate_iri",
    "value",
    "datatype",
    "language",
    "statement_ids",
}
_BLOCKED_CLASS_NAMES = frozenset(
    {"capability", "primitive", "recipe", "resource", "specification"}
)
_BLOCKED_PROPERTY_NAMES = frozenset(
    {"capableOf", "precedes", "provides", "requires"}
)


class OntologyGroundingError(ValueError):
    """Raised when an untrusted late ontology proposal cannot be accepted."""


@dataclass(frozen=True)
class OntologyGroundingProposal:
    """Hold one validated, still-untrusted statement-to-ontology mapping."""

    individuals: tuple[Mapping[str, object], ...]
    relations: tuple[Mapping[str, object], ...]
    literal_facts: tuple[Mapping[str, object], ...]
    unrepresented_statement_ids: tuple[str, ...]


@dataclass(frozen=True)
class OntologyGroundingResult:
    """Return accepted assertions and the persisted late-mapping proposal."""

    merge: MergeResult
    proposal_path: Path
    proposal: OntologyGroundingProposal


async def propose_and_validate_ontology_grounding(
    product_agent: ProductAgentContextRuntime,
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    session: GroundingSession,
) -> OntologyGroundingResult:
    """Map directly supported session statements and atomically validate the delta."""
    root = Path(interaction_root).resolve()
    if abox.interaction_root != root or abox.tbox_fingerprint != tbox.fingerprint:
        raise OntologyGroundingError(
            "Ontology grounding inputs do not share one interaction and TBox."
        )
    if (
        session.requirement_text != abox.product_requirement
        or session.status != "ready_for_ontology"
        or session.decision.decision_type != "ready_for_ontology"
    ):
        raise OntologyGroundingError(
            "Late ontology mapping requires the ready GroundingSession."
        )
    direct_statements = tuple(
        statement
        for statement in session.statements
        if statement.status == "directly_stated"
    )
    if not direct_statements:
        raise OntologyGroundingError(
            "Late ontology mapping requires directly supported statements."
        )
    allowed_classes = _allowed_classes(tbox)
    allowed_object_properties = _allowed_object_properties(tbox)
    allowed_datatype_properties = _allowed_datatype_properties(tbox)
    existing_individuals = _existing_individuals(abox)
    property_signatures = _property_signatures(
        tbox,
        allowed_object_properties | allowed_datatype_properties,
    )
    prompt_input = {
        "directly_supported_statements": [
            statement.to_record() for statement in direct_statements
        ],
        "initialized_specification_iri": abox.specification_iri,
        "allowed_classes": sorted(allowed_classes),
        "allowed_object_properties": sorted(allowed_object_properties),
        "allowed_datatype_properties": sorted(allowed_datatype_properties),
        "property_signatures": property_signatures,
        "current_individuals": existing_individuals,
    }
    prompt = (
        "Create one untrusted late semantic mapping from only the supplied directly "
        "supported statement IDs. Every proposed fact must cite statement_ids. "
        "The initialized specification IRI is controller-owned: never create, rename, "
        "or type it. The controller assigns IRIs to new individuals. Refer to current "
        "individuals only by their exact supplied IRIs. Respect every property domain "
        "and range. Do not add primitive, resource, capability, recipe, requires, "
        "precedes, resource-selection, or composition facts. Do not map inferred or "
        "missing information. Put every directly supported statement that cannot be "
        "represented safely into unrepresented_statement_ids. Do not force a relation; "
        "in particular, use realizes only when cited statements support a correctly "
        "typed process and feature.\n\n"
        f"Late mapping input:\n{json.dumps(prompt_input, indent=2, ensure_ascii=False)}"
    )
    output = await product_agent.ask_llm_structured(
        prompt,
        response_format=_proposal_response_format(
            statement_ids=[item.statement_id for item in direct_statements],
            classes=sorted(allowed_classes),
            existing_iris=sorted(existing_individuals),
            object_properties=sorted(allowed_object_properties),
            datatype_properties=sorted(allowed_datatype_properties),
        ),
    )
    proposal_number = _next_proposal_number(root)
    proposal_path = root / _PROPOSAL_ROOT / f"proposal_{proposal_number:04d}.json"
    proposal: OntologyGroundingProposal | None = None
    delta: Mapping[str, object] | None = None
    try:
        if not isinstance(output, Mapping) or set(output) != {
            "ontology_grounding_proposal"
        }:
            raise OntologyGroundingError(
                "ProductAgent returned an invalid OntologyGroundingProposal envelope."
            )
        proposal = _validated_proposal(
            output["ontology_grounding_proposal"],
            direct_statements=direct_statements,
            allowed_classes=allowed_classes,
            allowed_object_properties=allowed_object_properties,
            allowed_datatype_properties=allowed_datatype_properties,
            existing_individuals=existing_individuals,
            property_signatures=property_signatures,
            specification_iri=abox.specification_iri,
        )
        delta = _compile_proposal_delta(
            proposal,
            proposal_number=proposal_number,
            abox=abox,
            statements={item.statement_id: item for item in direct_statements},
        )
        authorized_refs = sorted(
            {
                source
                for statement in direct_statements
                for source in statement.sources
            }
        )
        merge = validate_and_merge_triple_delta(
            root,
            tbox,
            _PRODUCER,
            delta,
            authorized_evidence_refs=authorized_refs,
        )
    except (KeyError, OntologyGroundingError, TypeError, ValueError) as exc:
        _write_proposal_record(
            proposal_path,
            proposal_number=proposal_number,
            session=session,
            specification_iri=abox.specification_iri,
            output=output if isinstance(output, Mapping) else {"output": output},
            compiled_delta=delta,
            status="rejected",
            failure=f"{type(exc).__name__}: {exc}",
        )
        if isinstance(exc, OntologyGroundingError):
            raise
        raise OntologyGroundingError(
            f"OntologyGroundingProposal was rejected: {type(exc).__name__}: {exc}"
        ) from exc
    _write_proposal_record(
        proposal_path,
        proposal_number=proposal_number,
        session=session,
        specification_iri=abox.specification_iri,
        output=output,
        compiled_delta=delta,
        status="accepted",
        failure=None,
    )
    return OntologyGroundingResult(
        merge=merge,
        proposal_path=proposal_path,
        proposal=proposal,
    )


def _validated_proposal(  # noqa: C901, PLR0913
    value: object,
    *,
    direct_statements: Sequence[GroundingStatement],
    allowed_classes: frozenset[str],
    allowed_object_properties: frozenset[str],
    allowed_datatype_properties: frozenset[str],
    existing_individuals: Mapping[str, tuple[str, ...]],
    property_signatures: Mapping[str, Mapping[str, object]],
    specification_iri: str,
) -> OntologyGroundingProposal:
    if not isinstance(value, Mapping) or set(value) != _PROPOSAL_KEYS:
        raise OntologyGroundingError("OntologyGroundingProposal fields are invalid.")
    statement_ids = {item.statement_id for item in direct_statements}
    individuals = _mapping_list(value["individuals"], "individuals")
    relations = _mapping_list(value["relations"], "relations")
    literal_facts = _mapping_list(value["literal_facts"], "literal_facts")
    unrepresented = _string_list(
        value["unrepresented_statement_ids"],
        "unrepresented_statement_ids",
        allow_empty=True,
    )
    if len(set(unrepresented)) != len(unrepresented):
        raise OntologyGroundingError(
            "Ontology proposal unrepresented statement IDs must be unique."
        )
    if not set(unrepresented).issubset(statement_ids):
        raise OntologyGroundingError(
            "Ontology proposal reports an unknown statement as unrepresented."
        )
    validated_individuals: list[Mapping[str, object]] = []
    new_types: dict[int, str] = {}
    used_statement_ids: set[str] = set()
    for item in individuals:
        if set(item) != _INDIVIDUAL_KEYS:
            raise OntologyGroundingError("Proposal individual fields are invalid.")
        index = _positive_integer(item["individual_index"], "individual_index")
        class_iri = item["class_iri"]
        if not isinstance(class_iri, str) or class_iri not in allowed_classes:
            raise OntologyGroundingError("Proposal individual class is not allowed.")
        if index in new_types:
            raise OntologyGroundingError("Proposal individual indices must be unique.")
        cited = _statement_refs(item["statement_ids"], statement_ids)
        used_statement_ids.update(cited)
        new_types[index] = class_iri
        validated_individuals.append(dict(item))

    validated_relations: list[Mapping[str, object]] = []
    for item in relations:
        if set(item) != _RELATION_KEYS:
            raise OntologyGroundingError("Proposal relation fields are invalid.")
        predicate = item["predicate_iri"]
        if not isinstance(predicate, str) or predicate not in allowed_object_properties:
            raise OntologyGroundingError("Proposal object property is not allowed.")
        subject_types = _subject_types(
            item,
            new_types=new_types,
            existing_individuals=existing_individuals,
            specification_iri=specification_iri,
        )
        object_types = _object_types(
            item,
            new_types=new_types,
            existing_individuals=existing_individuals,
        )
        _validate_property_signature(
            predicate,
            subject_types=subject_types,
            object_types=object_types,
            property_signatures=property_signatures,
        )
        cited = _statement_refs(item["statement_ids"], statement_ids)
        used_statement_ids.update(cited)
        validated_relations.append(dict(item))

    validated_literals: list[Mapping[str, object]] = []
    for item in literal_facts:
        if set(item) != _LITERAL_KEYS:
            raise OntologyGroundingError("Proposal literal fact fields are invalid.")
        predicate = item["predicate_iri"]
        if not isinstance(predicate, str) or predicate not in allowed_datatype_properties:
            raise OntologyGroundingError("Proposal datatype property is not allowed.")
        _subject_types(
            item,
            new_types=new_types,
            existing_individuals=existing_individuals,
            specification_iri=specification_iri,
        )
        if isinstance(item["value"], (dict, list)) or item["value"] is None:
            raise OntologyGroundingError("Proposal literal value is invalid.")
        for field in ("datatype", "language"):
            if item[field] is not None and not isinstance(item[field], str):
                raise OntologyGroundingError(f"Proposal literal {field} is invalid.")
        cited = _statement_refs(item["statement_ids"], statement_ids)
        used_statement_ids.update(cited)
        validated_literals.append(dict(item))

    if used_statement_ids & set(unrepresented):
        raise OntologyGroundingError(
            "A statement cannot be both mapped and unrepresented."
        )
    if used_statement_ids | set(unrepresented) != statement_ids:
        raise OntologyGroundingError(
            "Every directly supported statement must be mapped or reported unrepresented."
        )
    return OntologyGroundingProposal(
        individuals=tuple(validated_individuals),
        relations=tuple(validated_relations),
        literal_facts=tuple(validated_literals),
        unrepresented_statement_ids=tuple(unrepresented),
    )


def _compile_proposal_delta(
    proposal: OntologyGroundingProposal,
    *,
    proposal_number: int,
    abox: ABoxSnapshot,
    statements: Mapping[str, GroundingStatement],
) -> dict[str, object]:
    individual_iris = {
        int(item["individual_index"]): (
            f"{abox.namespace}grounding_{proposal_number:04d}_"
            f"individual_{int(item['individual_index']):04d}"
        )
        for item in proposal.individuals
    }
    assertions: list[dict[str, object]] = []
    for item in proposal.individuals:
        assertions.append(
            {
                "subject": individual_iris[int(item["individual_index"])],
                "predicate": str(RDF.type),
                "object": {"kind": "iri", "value": item["class_iri"]},
                "evidence_refs": _statement_sources(item["statement_ids"], statements),
            }
        )
    for item in proposal.relations:
        assertions.append(
            {
                "subject": _compiled_subject(item, abox, individual_iris),
                "predicate": item["predicate_iri"],
                "object": {
                    "kind": "iri",
                    "value": _compiled_object(item, individual_iris),
                },
                "evidence_refs": _statement_sources(item["statement_ids"], statements),
            }
        )
    for item in proposal.literal_facts:
        object_value: dict[str, object] = {
            "kind": "literal",
            "value": item["value"],
        }
        if item["datatype"] is not None:
            object_value["datatype"] = item["datatype"]
        if item["language"] is not None:
            object_value["language"] = item["language"]
        assertions.append(
            {
                "subject": _compiled_subject(item, abox, individual_iris),
                "predicate": item["predicate_iri"],
                "object": object_value,
                "evidence_refs": _statement_sources(item["statement_ids"], statements),
            }
        )
    return {
        "assertions": assertions,
        "uncertainty": [],
        "unresolved_evidence_needs": [
            {
                "description": (
                    "The current TBox cannot represent directly supported statement "
                    f"{statement_id}: {statements[statement_id].text}"
                ),
                "evidence_refs": list(statements[statement_id].sources),
            }
            for statement_id in proposal.unrepresented_statement_ids
        ],
        "typed_context_refs": [],
    }


def _proposal_response_format(  # noqa: PLR0913
    *,
    statement_ids: Sequence[str],
    classes: Sequence[str],
    existing_iris: Sequence[str],
    object_properties: Sequence[str],
    datatype_properties: Sequence[str],
) -> dict[str, Any]:
    statement_refs = {
        "type": "array",
        "items": {"type": "string", "enum": list(statement_ids)},
        "minItems": 1,
    }
    subject_properties = {
        "subject_kind": {
            "type": "string",
            "enum": ["specification", "new_individual", "existing_individual"],
        },
        "subject_individual_index": {"type": ["integer", "null"], "minimum": 1},
        "subject_iri": {
            "type": ["string", "null"],
            "enum": [None, *existing_iris],
        },
    }
    individual = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_INDIVIDUAL_KEYS),
        "properties": {
            "individual_index": {"type": "integer", "minimum": 1},
            "class_iri": {
                "type": "string",
                "enum": list(classes) or ["no_class_available"],
            },
            "statement_ids": statement_refs,
        },
    }
    relation = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_RELATION_KEYS),
        "properties": {
            **subject_properties,
            "predicate_iri": {
                "type": "string",
                "enum": list(object_properties) or ["no_object_property_available"],
            },
            "object_kind": {
                "type": "string",
                "enum": ["new_individual", "existing_individual"],
            },
            "object_individual_index": {
                "type": ["integer", "null"],
                "minimum": 1,
            },
            "object_iri": {
                "type": ["string", "null"],
                "enum": [None, *existing_iris],
            },
            "statement_ids": statement_refs,
        },
    }
    literal = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_LITERAL_KEYS),
        "properties": {
            **subject_properties,
            "predicate_iri": {
                "type": "string",
                "enum": list(datatype_properties)
                or ["no_datatype_property_available"],
            },
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                ]
            },
            "datatype": {"type": ["string", "null"]},
            "language": {"type": ["string", "null"]},
            "statement_ids": statement_refs,
        },
    }
    return {
        "name": "spec2primitives_ontology_grounding_proposal",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["ontology_grounding_proposal"],
            "properties": {
                "ontology_grounding_proposal": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_PROPOSAL_KEYS),
                    "properties": {
                        "individuals": {
                            "type": "array",
                            "items": individual,
                            **({} if classes else {"maxItems": 0}),
                        },
                        "relations": {
                            "type": "array",
                            "items": relation,
                            **({} if object_properties else {"maxItems": 0}),
                        },
                        "literal_facts": {
                            "type": "array",
                            "items": literal,
                            **({} if datatype_properties else {"maxItems": 0}),
                        },
                        "unrepresented_statement_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(statement_ids),
                            },
                        },
                    },
                }
            },
        },
    }


def _allowed_classes(tbox: TBoxSnapshot) -> frozenset[str]:
    blocked_roots = {
        URIRef(f"{tbox.ppr_namespace}resource"),
        URIRef(f"{tbox.ppr_namespace}capability"),
    }
    allowed: set[str] = set()
    for class_iri in tbox.classes:
        class_ref = URIRef(class_iri)
        local_name = _local_name(class_iri)
        if local_name in _BLOCKED_CLASS_NAMES:
            continue
        if any(tbox.is_class_or_subclass(class_ref, root) for root in blocked_roots):
            continue
        allowed.add(class_iri)
    return frozenset(allowed)


def _allowed_object_properties(tbox: TBoxSnapshot) -> frozenset[str]:
    return frozenset(
        item
        for item in tbox.object_properties
        if _local_name(item) not in _BLOCKED_PROPERTY_NAMES
    )


def _allowed_datatype_properties(tbox: TBoxSnapshot) -> frozenset[str]:
    return frozenset(
        item
        for item in tbox.datatype_properties
        if _local_name(item) not in _BLOCKED_PROPERTY_NAMES
    )


def _existing_individuals(abox: ABoxSnapshot) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for subject in sorted(set(abox.graph.subjects()), key=str):
        if not isinstance(subject, URIRef) or str(subject) == abox.specification_iri:
            continue
        types = tuple(
            sorted(
                str(value)
                for value in abox.graph.objects(subject, RDF.type)
                if isinstance(value, URIRef)
            )
        )
        result[str(subject)] = types
    return result


def _property_signatures(
    tbox: TBoxSnapshot,
    properties: frozenset[str],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for property_iri in sorted(properties):
        ref = URIRef(property_iri)
        domains = sorted(
            str(item)
            for item in tbox.graph.objects(ref, RDFS.domain)
            if isinstance(item, URIRef)
        )
        ranges = sorted(
            str(item)
            for item in tbox.graph.objects(ref, RDFS.range)
            if isinstance(item, URIRef)
        )
        local_name = _local_name(property_iri)
        if local_name == "defines":
            domains = domains or [f"{tbox.ppr_namespace}specification"]
            ranges = ranges or [f"{tbox.ppr_namespace}feature"]
        elif local_name == "realizes":
            domains = domains or [f"{tbox.ppr_namespace}process"]
            ranges = ranges or [f"{tbox.ppr_namespace}feature"]
        result[property_iri] = {
            "domain_classes": domains,
            "range_classes": ranges,
        }
    return result


def _subject_types(
    item: Mapping[str, object],
    *,
    new_types: Mapping[int, str],
    existing_individuals: Mapping[str, tuple[str, ...]],
    specification_iri: str,
) -> tuple[str, ...]:
    kind = item["subject_kind"]
    index = item["subject_individual_index"]
    iri = item["subject_iri"]
    if kind == "specification":
        if index is not None or iri is not None:
            raise OntologyGroundingError(
                "The specification subject cannot contain an index or IRI."
            )
        return ("specification",)
    if kind == "new_individual":
        if iri is not None:
            raise OntologyGroundingError("A new subject cannot contain an existing IRI.")
        validated_index = _positive_integer(index, "subject_individual_index")
        class_iri = new_types.get(validated_index)
        if class_iri is None:
            raise OntologyGroundingError("Proposal subject individual is undeclared.")
        return (class_iri,)
    if kind == "existing_individual":
        if index is not None or not isinstance(iri, str) or iri == specification_iri:
            raise OntologyGroundingError("Existing subject reference is invalid.")
        types = existing_individuals.get(iri)
        if types is None:
            raise OntologyGroundingError("Existing subject is not in the current ABox.")
        return types
    raise OntologyGroundingError("Proposal subject_kind is invalid.")


def _object_types(
    item: Mapping[str, object],
    *,
    new_types: Mapping[int, str],
    existing_individuals: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    kind = item["object_kind"]
    index = item["object_individual_index"]
    iri = item["object_iri"]
    if kind == "new_individual":
        if iri is not None:
            raise OntologyGroundingError("A new object cannot contain an existing IRI.")
        validated_index = _positive_integer(index, "object_individual_index")
        class_iri = new_types.get(validated_index)
        if class_iri is None:
            raise OntologyGroundingError("Proposal object individual is undeclared.")
        return (class_iri,)
    if kind == "existing_individual":
        if index is not None or not isinstance(iri, str):
            raise OntologyGroundingError("Existing object reference is invalid.")
        types = existing_individuals.get(iri)
        if types is None:
            raise OntologyGroundingError("Existing object is not in the current ABox.")
        return types
    raise OntologyGroundingError("Proposal object_kind is invalid.")


def _validate_property_signature(
    predicate: str,
    *,
    subject_types: Sequence[str],
    object_types: Sequence[str],
    property_signatures: Mapping[str, Mapping[str, object]],
) -> None:
    signature = property_signatures[predicate]
    domains = set(
        _string_list(signature["domain_classes"], "domain_classes", allow_empty=True)
    )
    ranges = set(
        _string_list(signature["range_classes"], "range_classes", allow_empty=True)
    )
    normalized_subject_types = set(subject_types)
    if "specification" in normalized_subject_types:
        normalized_subject_types = {
            item for item in domains if _local_name(item) == "specification"
        } or {"specification"}
    if domains and not normalized_subject_types.intersection(domains):
        raise OntologyGroundingError("Proposal relation subject violates property domain.")
    if ranges and not set(object_types).intersection(ranges):
        raise OntologyGroundingError("Proposal relation object violates property range.")


def _compiled_subject(
    item: Mapping[str, object],
    abox: ABoxSnapshot,
    individual_iris: Mapping[int, str],
) -> str:
    if item["subject_kind"] == "specification":
        return abox.specification_iri
    if item["subject_kind"] == "existing_individual":
        return str(item["subject_iri"])
    return individual_iris[int(item["subject_individual_index"])]


def _compiled_object(
    item: Mapping[str, object],
    individual_iris: Mapping[int, str],
) -> str:
    if item["object_kind"] == "existing_individual":
        return str(item["object_iri"])
    return individual_iris[int(item["object_individual_index"])]


def _statement_refs(value: object, allowed: set[str]) -> list[str]:
    refs = _string_list(value, "statement_ids")
    if len(set(refs)) != len(refs) or not set(refs).issubset(allowed):
        raise OntologyGroundingError("Proposal cites an unknown statement ID.")
    return refs


def _statement_sources(
    value: object,
    statements: Mapping[str, GroundingStatement],
) -> list[str]:
    result: set[str] = set()
    for statement_id in _string_list(value, "statement_ids"):
        statement = statements.get(statement_id)
        if statement is None or statement.status != "directly_stated":
            raise OntologyGroundingError(
                "Ontology assertion cites a non-direct statement."
            )
        result.update(statement.sources)
    return sorted(result)


def _mapping_list(value: object, field: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise OntologyGroundingError(f"{field} must be a list of objects.")
    return list(value)


def _string_list(
    value: object,
    field: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise OntologyGroundingError(f"{field} must be a string list.")
    return list(value)


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OntologyGroundingError(f"{field} must be a positive integer.")
    return value


def _local_name(iri: str) -> str:
    return iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _next_proposal_number(root: Path) -> int:
    return len(tuple((root / _PROPOSAL_ROOT).glob("proposal_*.json"))) + 1


def _write_proposal_record(  # noqa: PLR0913
    path: Path,
    *,
    proposal_number: int,
    session: GroundingSession,
    specification_iri: str,
    output: Mapping[str, object],
    compiled_delta: Mapping[str, object] | None,
    status: str,
    failure: str | None,
) -> None:
    record = {
        "schema_version": 2,
        "record_type": "OntologyGroundingProposal",
        "proposal_number": proposal_number,
        "session_revision": session.revision,
        "session_fingerprint": session.fingerprint,
        "initialized_specification_iri": specification_iri,
        "direct_statement_ids": [
            item.statement_id
            for item in session.statements
            if item.status == "directly_stated"
        ],
        "output": dict(output),
        "compiled_delta": None if compiled_delta is None else dict(compiled_delta),
        "status": status,
        "failure": failure,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise OntologyGroundingError(
            f"Ontology proposal already exists: {path.name}."
        ) from exc
