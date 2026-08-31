"""Map directly supported PA statements through the authoritative ABox validator."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdflib import RDF, RDFS, URIRef

from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    MergeResult,
    validate_and_merge_triple_delta,
    validate_triple_delta,
)
from cais_spade_llm.spec2primitives.ontology import (
    PredefinedWorkcellSnapshot,
    TBoxSnapshot,
)

_PRODUCER = "ontology_grounding"
_PROPOSAL_ROOT = Path("products/grounding/ontology_grounding")
_PROPOSAL_KEYS = {
    "individuals",
    "relations",
    "literal_facts",
    "context_summary",
    "evidence_refs",
    "missing_information",
}
_INDIVIDUAL_KEYS = {
    "individual_index",
    "class_iri",
    "grounded_meaning",
    "evidence_refs",
}
_RELATION_KEYS = {
    "subject_kind",
    "subject_individual_index",
    "subject_iri",
    "predicate_iri",
    "object_kind",
    "object_individual_index",
    "object_iri",
    "evidence_refs",
}
_LITERAL_KEYS = {
    "subject_kind",
    "subject_individual_index",
    "subject_iri",
    "predicate_iri",
    "value",
    "datatype",
    "language",
}
_BLOCKED_CLASS_NAMES = frozenset(
    {
        "capability",
        "primitive",
        "processExecution",
        "recipe",
        "resource",
        "specification",
    }
)
_BLOCKED_PROPERTY_NAMES = frozenset(
    {
        "capableOf",
        "hasProcessExecution",
        "precedes",
        "provides",
        "requires",
        "runsOnResource",
        "runsProcess",
    }
)


class OntologyGroundingError(ValueError):
    """Raised when an untrusted late ontology proposal cannot be accepted."""


@dataclass(frozen=True)
class OntologyGroundingProposal:
    """Hold one validated ontology projection and final cited context."""

    individuals: tuple[Mapping[str, object], ...]
    relations: tuple[Mapping[str, object], ...]
    literal_facts: tuple[Mapping[str, object], ...]
    context_summary: str
    evidence_refs: tuple[str, ...]
    missing_information: tuple[str, ...]


@dataclass(frozen=True)
class OntologyGroundingResult:
    """Return accepted assertions and the persisted late-mapping proposal."""

    merge: MergeResult
    proposal_path: Path
    proposal: OntologyGroundingProposal


@dataclass(frozen=True)
class OntologyGroundingCandidate:
    """Hold one valid but uncommitted PA ontology proposal."""

    provisional_abox: ABoxSnapshot
    proposal_path: Path
    proposal_number: int
    output: Mapping[str, object]
    compiled_delta: Mapping[str, object]
    proposal: OntologyGroundingProposal


@dataclass(frozen=True)
class OntologyGroundingInterruption:
    """Return a direct ProductAgent clarification or insufficiency result."""

    kind: str
    message: str


async def propose_and_validate_ontology_grounding(  # noqa: PLR0913
    product_agent: ProductAgentContextRuntime,
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
    evidence_catalog: Sequence[Mapping[str, object]],
    authorized_evidence_refs: set[str],
    tools: list[dict[str, Any]],
    tool_executor: Callable[
        [str, Mapping[str, object]], Awaitable[Mapping[str, object]]
    ],
    max_tool_rounds: int,
    required_output_projection: Mapping[str, object],
    validation_gap: Mapping[str, object] | None = None,
) -> OntologyGroundingCandidate | OntologyGroundingInterruption:
    """Let PA investigate and return one valid, uncommitted semantic candidate."""
    root = Path(interaction_root).resolve()
    if abox.interaction_root != root or abox.tbox_fingerprint != tbox.fingerprint:
        raise OntologyGroundingError(
            "Ontology grounding inputs do not share one interaction and TBox."
        )
    if not isinstance(workcell, PredefinedWorkcellSnapshot):
        raise OntologyGroundingError(
            "Ontology grounding requires a PredefinedWorkcellSnapshot."
        )
    try:
        workcell.assert_unchanged()
    except (TypeError, ValueError) as exc:
        raise OntologyGroundingError(
            "Ontology grounding requires an immutable predefined Workcell."
        ) from exc
    if workcell.tbox_fingerprint != tbox.fingerprint:
        raise OntologyGroundingError(
            "Ontology grounding Workcell does not match the authoritative TBox."
        )
    allowed_classes = frozenset({f"{tbox.ppr_namespace}feature"})
    if not allowed_classes.issubset(_allowed_classes(tbox)):
        raise OntologyGroundingError(
            "The authoritative TBox does not expose the required feature class."
        )
    required_properties = {
        f"{tbox.ppr_namespace}defines",
        f"{tbox.ppr_namespace}realizes",
    }
    allowed_object_properties = frozenset(required_properties)
    if not allowed_object_properties.issubset(_allowed_object_properties(tbox)):
        raise OntologyGroundingError(
            "The authoritative TBox does not expose the task-grounding properties."
        )
    allowed_datatype_properties: frozenset[str] = frozenset()
    existing_individuals = _existing_individuals(abox)
    existing_individuals[workcell.process_iri] = (
        f"{tbox.ppr_namespace}process",
    )
    property_signatures = _property_signatures(
        tbox,
        allowed_object_properties | allowed_datatype_properties,
    )
    prompt_input = {
        "exact_requirement": abox.product_requirement,
        "approved_evidence_catalog": [dict(item) for item in evidence_catalog],
        "required_output_projection": dict(required_output_projection),
        "initialized_specification_iri": abox.specification_iri,
        "allowed_classes": sorted(allowed_classes),
        "allowed_object_properties": sorted(allowed_object_properties),
        "allowed_datatype_properties": sorted(allowed_datatype_properties),
        "property_signatures": property_signatures,
        "current_individuals": existing_individuals,
        "predefined_process": {
            "process_symbol": workcell.process_symbol,
            "process_iri": workcell.process_iri,
        },
        "current_grounding_goal": {
            "required_meaning": "evidence-supported requirement features",
            "primary_execution_target_count": 1,
            "supporting_feature_count": "zero_or_more",
        },
    }
    if validation_gap is not None:
        prompt_input["current_validation_gap"] = dict(validation_gap)
    base_prompt = (
        "Investigate the requirement using only the exact requirement and approved "
        "evidence returned by the controlled retrieve tool. Before asking a user "
        "clarification, retrieve and consider every approved catalog entry plausibly "
        "relevant to that ambiguity. You may otherwise retrieve zero, one, or multiple "
        "catalog entries in any order. Catalog metadata is discovery-only; only "
        "evidence refs returned by retrieve may support assertions. Never use hidden "
        "case knowledge, evaluator information, or an interpretation supplied by these "
        "instructions. After tool "
        "use, return exactly one result containing proposal fields, one "
        "clarification_question, or one insufficient_evidence result. The supplied "
        "required_output_projection is mandatory and system-authorized: never ask "
        "the user whether it should be satisfied or whether approved evidence should "
        "be retrieved for it. Ask a clarification only for requirement meaning that "
        "the approved evidence cannot resolve. For a "
        "proposal, create one concise context summary. Ground the smallest set of "
        "distinct, evidence-supported requirement-level "
        "features needed for current_grounding_goal. Choose the feature count from "
        "the evidence. Give every feature one concise grounded_meaning and direct "
        "evidence_refs. Every proposed feature must have exactly one defines relation "
        "from the initialized specification. Exactly one proposed feature must also "
        "have one realizes relation from the predefined process. Use the supplied "
        "property signatures and current individuals to identify that current "
        "execution target; "
        "any other grounded features are supporting specification context. Give every "
        "relation its direct evidence_refs and cite only refs returned by retrieve "
        "or requirement_0001. Keep "
        "useful details that the current "
        "ontology cannot represent in context_summary and missing_information. The "
        "initialized specification IRI is controller-owned: never create, rename, or "
        "type it. The controller assigns IRIs to new individuals. Refer to current "
        "individuals only by their exact supplied IRIs. Respect every property domain "
        "and range. Do not add primitive, resource, capability, recipe, requires, "
        "precedes, process-execution, resource-selection, or composition facts. Do "
        "not create a process individual. Use missing_information only for "
        "evidence-backed product or scene facts that remain unknown after considering "
        "the retrieved evidence. Do not report information as missing solely because "
        "it is represented by a typed context record instead of an RDF predicate. "
        "Information outside the allowed ontology projection does not block a valid "
        "proposal unless the required ontology-level meaning is unsupported. "
        "Do not force a relation "
        "that the available evidence does not support.\n\n"
        f"Grounding input:\n{json.dumps(prompt_input, indent=2, ensure_ascii=False)}"
    )
    response_format = _proposal_response_format(
        classes=sorted(allowed_classes),
        existing_iris=sorted(existing_individuals),
        object_properties=sorted(allowed_object_properties),
        datatype_properties=sorted(allowed_datatype_properties),
    )
    proposal_number = _next_proposal_number(root)
    proposal_path = root / _PROPOSAL_ROOT / f"proposal_{proposal_number:04d}.json"
    transport_output = await product_agent.ask_llm_structured(
        base_prompt,
        response_format=response_format,
        tools=tools,
        tool_executor=tool_executor,
        max_tool_rounds=max_tool_rounds,
    )
    if (
        not isinstance(transport_output, Mapping)
        or set(transport_output) != {"result"}
        or not isinstance(transport_output["result"], Mapping)
    ):
        raise OntologyGroundingError("ProductAgent grounding result wrapper is invalid.")
    output = transport_output["result"]
    if isinstance(output, Mapping) and set(output) == {"clarification_question"}:
        message = output["clarification_question"]
        if not isinstance(message, str) or not message.strip():
            raise OntologyGroundingError("clarification_question is invalid.")
        return OntologyGroundingInterruption("clarification_question", message)
    if isinstance(output, Mapping) and set(output) == {"insufficient_evidence"}:
        message = output["insufficient_evidence"]
        if not isinstance(message, str) or not message.strip():
            raise OntologyGroundingError("insufficient_evidence is invalid.")
        return OntologyGroundingInterruption("insufficient_evidence", message)

    try:
        proposal = _validated_proposal(
            output,
            authorized_evidence_refs=authorized_evidence_refs,
            allowed_classes=allowed_classes,
            allowed_object_properties=allowed_object_properties,
            allowed_datatype_properties=allowed_datatype_properties,
            existing_individuals=existing_individuals,
            property_signatures=property_signatures,
            specification_iri=abox.specification_iri,
        )
        _validate_provisional_task_proposal(proposal, tbox=tbox, workcell=workcell)
        delta = _compile_proposal_delta(
            proposal,
            proposal_number=proposal_number,
            abox=abox,
        )
        validated_delta = validate_triple_delta(
            root,
            tbox,
            delta,
            authorized_evidence_refs=sorted(authorized_evidence_refs),
            authorized_external_process_iris={workcell.process_iri},
        )
    except (KeyError, OntologyGroundingError, TypeError, ValueError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        _write_proposal_record(
            proposal_path,
            proposal_number=proposal_number,
            specification_iri=abox.specification_iri,
            output=output if isinstance(output, Mapping) else {"output": output},
            compiled_delta=None,
            status="rejected",
            failure=failure,
        )
        raise OntologyGroundingError(
            f"OntologyGroundingProposal is invalid: {failure}"
        ) from exc

    return OntologyGroundingCandidate(
        provisional_abox=validated_delta.abox,
        proposal_path=proposal_path,
        proposal_number=proposal_number,
        output=dict(output),
        compiled_delta=dict(delta),
        proposal=proposal,
    )


def commit_ontology_grounding_candidate(
    candidate: OntologyGroundingCandidate,
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
    authorized_evidence_refs: set[str],
) -> OntologyGroundingResult:
    """Revalidate and persist one evidence-ready ontology candidate."""
    root = Path(interaction_root).resolve()
    if (
        abox.interaction_root != root
        or candidate.provisional_abox.interaction_root != root
        or abox.namespace != candidate.provisional_abox.namespace
        or abox.product_requirement != candidate.provisional_abox.product_requirement
        or abox.tbox_fingerprint != tbox.fingerprint
        or candidate.provisional_abox.tbox_fingerprint != tbox.fingerprint
        or candidate.proposal_path
        != root / _PROPOSAL_ROOT / f"proposal_{candidate.proposal_number:04d}.json"
        or candidate.proposal_path.exists()
    ):
        raise OntologyGroundingError(
            "Ontology grounding candidate no longer matches this interaction."
        )
    merge = validate_and_merge_triple_delta(
        root,
        tbox,
        _PRODUCER,
        candidate.compiled_delta,
        authorized_evidence_refs=sorted(authorized_evidence_refs),
        authorized_external_process_iris={workcell.process_iri},
    )
    _write_proposal_record(
        candidate.proposal_path,
        proposal_number=candidate.proposal_number,
        specification_iri=abox.specification_iri,
        output=candidate.output,
        compiled_delta=candidate.compiled_delta,
        status="accepted",
        failure=None,
    )
    return OntologyGroundingResult(
        merge=merge,
        proposal_path=candidate.proposal_path,
        proposal=candidate.proposal,
    )


def _validate_provisional_task_proposal(
    proposal: OntologyGroundingProposal,
    *,
    tbox: TBoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
) -> None:
    """Require connected features and one graph-derived execution target."""
    feature_iri = f"{tbox.ppr_namespace}feature"
    defines_iri = f"{tbox.ppr_namespace}defines"
    realizes_iri = f"{tbox.ppr_namespace}realizes"
    if not proposal.individuals or any(
        item["class_iri"] != feature_iri for item in proposal.individuals
    ):
        raise OntologyGroundingError(
            "The provisional task proposal requires at least one feature."
        )
    if proposal.literal_facts:
        raise OntologyGroundingError(
            "The provisional task proposal does not allow literal facts."
        )

    feature_indices = {
        int(item["individual_index"]) for item in proposal.individuals
    }
    defined_indices: list[int] = []
    realized_indices: list[int] = []
    for relation in proposal.relations:
        predicate = relation["predicate_iri"]
        object_index = relation["object_individual_index"]
        if (
            relation["object_kind"] != "new_individual"
            or not isinstance(object_index, int)
            or isinstance(object_index, bool)
            or object_index not in feature_indices
            or relation["object_iri"] is not None
        ):
            raise OntologyGroundingError(
                "Every task-grounding relation must target a proposed feature."
            )
        if predicate == defines_iri:
            if (
                relation["subject_kind"] != "specification"
                or relation["subject_individual_index"] is not None
                or relation["subject_iri"] is not None
            ):
                raise OntologyGroundingError(
                    "Every grounded feature must be defined by the specification."
                )
            defined_indices.append(object_index)
            continue
        if predicate == realizes_iri:
            if (
                relation["subject_kind"] != "existing_individual"
                or relation["subject_individual_index"] is not None
                or relation["subject_iri"] != workcell.process_iri
            ):
                raise OntologyGroundingError(
                    "The execution target must be realized by the predefined process."
                )
            realized_indices.append(object_index)
            continue
        raise OntologyGroundingError(
            "The provisional task proposal contains an unrelated relation."
        )

    if len(defined_indices) != len(set(defined_indices)):
        raise OntologyGroundingError(
            "Each grounded feature must have exactly one specification definition."
        )
    if set(defined_indices) != feature_indices:
        raise OntologyGroundingError(
            "Every grounded feature must be defined by the specification."
        )
    if len(realized_indices) != 1:
        raise OntologyGroundingError(
            "The provisional task proposal requires exactly one graph-derived "
            "execution target."
        )


def _validated_proposal(  # noqa: C901, PLR0913
    value: object,
    *,
    authorized_evidence_refs: set[str],
    allowed_classes: frozenset[str],
    allowed_object_properties: frozenset[str],
    allowed_datatype_properties: frozenset[str],
    existing_individuals: Mapping[str, tuple[str, ...]],
    property_signatures: Mapping[str, Mapping[str, object]],
    specification_iri: str,
) -> OntologyGroundingProposal:
    if not isinstance(value, Mapping) or set(value) != _PROPOSAL_KEYS:
        raise OntologyGroundingError("OntologyGroundingProposal fields are invalid.")
    context_summary = value["context_summary"]
    if not isinstance(context_summary, str) or not context_summary.strip():
        raise OntologyGroundingError("Proposal context_summary is invalid.")
    evidence_refs = _string_list(value["evidence_refs"], "evidence_refs")
    if (
        len(set(evidence_refs)) != len(evidence_refs)
        or not set(evidence_refs).issubset(authorized_evidence_refs)
    ):
        raise OntologyGroundingError(
            "Proposal cites duplicate or unauthorized evidence."
        )
    missing_information = _string_list(
        value["missing_information"],
        "missing_information",
        allow_empty=True,
    )
    if len(set(missing_information)) != len(missing_information):
        raise OntologyGroundingError(
            "Proposal missing_information values must be unique."
        )

    individuals = _mapping_list(value["individuals"], "individuals")
    relations = _mapping_list(value["relations"], "relations")
    literal_facts = _mapping_list(value["literal_facts"], "literal_facts")
    if not individuals and not relations and not literal_facts:
        raise OntologyGroundingError(
            "A grounding proposal must contain at least one ontology fact."
        )
    validated_individuals: list[Mapping[str, object]] = []
    new_types: dict[int, str] = {}
    for item in individuals:
        if set(item) != _INDIVIDUAL_KEYS:
            raise OntologyGroundingError("Proposal individual fields are invalid.")
        index = _positive_integer(item["individual_index"], "individual_index")
        class_iri = item["class_iri"]
        if not isinstance(class_iri, str) or class_iri not in allowed_classes:
            raise OntologyGroundingError("Proposal individual class is not allowed.")
        if index in new_types:
            raise OntologyGroundingError("Proposal individual indices must be unique.")
        grounded_meaning = item["grounded_meaning"]
        if not isinstance(grounded_meaning, str) or not grounded_meaning.strip():
            raise OntologyGroundingError(
                "Proposal individual grounded_meaning is invalid."
            )
        item_evidence_refs = _validated_evidence_refs(
            item["evidence_refs"],
            "individual evidence_refs",
            authorized_evidence_refs=authorized_evidence_refs,
        )
        new_types[index] = class_iri
        validated_item = dict(item)
        validated_item["evidence_refs"] = item_evidence_refs
        validated_individuals.append(validated_item)

    validated_relations: list[Mapping[str, object]] = []
    for item in relations:
        if set(item) != _RELATION_KEYS:
            raise OntologyGroundingError("Proposal relation fields are invalid.")
        predicate = item["predicate_iri"]
        if not isinstance(predicate, str) or predicate not in allowed_object_properties:
            raise OntologyGroundingError("Proposal object property is not allowed.")
        relation_evidence_refs = _validated_evidence_refs(
            item["evidence_refs"],
            "relation evidence_refs",
            authorized_evidence_refs=authorized_evidence_refs,
        )
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
        validated_item = dict(item)
        validated_item["evidence_refs"] = relation_evidence_refs
        validated_relations.append(validated_item)

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
        validated_literals.append(dict(item))

    return OntologyGroundingProposal(
        individuals=tuple(validated_individuals),
        relations=tuple(validated_relations),
        literal_facts=tuple(validated_literals),
        context_summary=context_summary,
        evidence_refs=tuple(evidence_refs),
        missing_information=tuple(missing_information),
    )


def _compile_proposal_delta(
    proposal: OntologyGroundingProposal,
    *,
    proposal_number: int,
    abox: ABoxSnapshot,
) -> dict[str, object]:
    del proposal_number
    individual_iris = {
        int(item["individual_index"]): (
            f"{abox.namespace}feature_{int(item['individual_index']):04d}"
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
                "evidence_refs": list(item["evidence_refs"]),
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
                "evidence_refs": list(item["evidence_refs"]),
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
                "evidence_refs": list(proposal.evidence_refs),
            }
        )
    return {
        "assertions": assertions,
        "uncertainty": [
            {
                "description": item,
                "evidence_refs": list(proposal.evidence_refs),
            }
            for item in proposal.missing_information
        ],
        "unresolved_evidence_needs": [],
        "typed_context_refs": [],
    }


def _proposal_response_format(  # noqa: PLR0913
    *,
    classes: Sequence[str],
    existing_iris: Sequence[str],
    object_properties: Sequence[str],
    datatype_properties: Sequence[str],
) -> dict[str, Any]:
    # Nullable reference fields alone permit contradictory kind/reference
    # combinations, so strict output enumerates only system-valid combinations.
    subject_property_variants = (
        {
            "subject_kind": {"type": "string", "enum": ["specification"]},
            "subject_individual_index": {"type": "null"},
            "subject_iri": {"type": "null"},
        },
        {
            "subject_kind": {"type": "string", "enum": ["new_individual"]},
            "subject_individual_index": {"type": "integer", "minimum": 1},
            "subject_iri": {"type": "null"},
        },
        {
            "subject_kind": {"type": "string", "enum": ["existing_individual"]},
            "subject_individual_index": {"type": "null"},
            "subject_iri": {"type": "string", "enum": list(existing_iris)},
        },
    )
    object_property_variants = (
        {
            "object_kind": {"type": "string", "enum": ["new_individual"]},
            "object_individual_index": {"type": "integer", "minimum": 1},
            "object_iri": {"type": "null"},
        },
        {
            "object_kind": {"type": "string", "enum": ["existing_individual"]},
            "object_individual_index": {"type": "null"},
            "object_iri": {"type": "string", "enum": list(existing_iris)},
        },
    )
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
            "grounded_meaning": {"type": "string", "minLength": 1},
            "evidence_refs": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
        },
    }
    relation = {
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_RELATION_KEYS),
                "properties": {
                    **subject_properties,
                    "predicate_iri": {
                        "type": "string",
                        "enum": list(object_properties)
                        or ["no_object_property_available"],
                    },
                    **object_reference_properties,
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                    },
                },
            }
            for subject_properties in subject_property_variants
            for object_reference_properties in object_property_variants
        ]
    }
    literal = {
        "anyOf": [
            {
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
                },
            }
            for subject_properties in subject_property_variants
        ]
    }
    # The structured-output API requires an object root, so the semantic union
    # lives under result and is unwrapped before deterministic system validation.
    # The API also rejects uniqueItems; _validated_proposal keeps uniqueness
    # deterministic after parsing instead of weakening the contract.
    return {
        "name": "spec2primitives_grounding_result",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["result"],
            "properties": {
                "result": {
                    "anyOf": [
                        {
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
                                "context_summary": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "evidence_refs": {
                                    "type": "array",
                                    "items": {"type": "string", "minLength": 1},
                                    "minItems": 1,
                                },
                                "missing_information": {
                                    "type": "array",
                                    "items": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["clarification_question"],
                            "properties": {
                                "clarification_question": {
                                    "type": "string",
                                    "minLength": 1,
                                }
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["insufficient_evidence"],
                            "properties": {
                                "insufficient_evidence": {
                                    "type": "string",
                                    "minLength": 1,
                                }
                            },
                        },
                    ],
                },
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


def _validated_evidence_refs(
    value: object,
    field: str,
    *,
    authorized_evidence_refs: set[str],
) -> list[str]:
    """Return one unique list of citations within the authorized evidence set."""
    evidence_refs = _string_list(value, field)
    if (
        len(set(evidence_refs)) != len(evidence_refs)
        or not set(evidence_refs).issubset(authorized_evidence_refs)
    ):
        raise OntologyGroundingError(
            f"{field} cites duplicate or unauthorized evidence."
        )
    return evidence_refs


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
    specification_iri: str,
    output: Mapping[str, object],
    compiled_delta: Mapping[str, object] | None,
    status: str,
    failure: str | None,
) -> None:
    record = {
        "schema_version": 5,
        "record_type": "OntologyGroundingProposal",
        "proposal_number": proposal_number,
        "initialized_specification_iri": specification_iri,
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
