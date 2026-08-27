"""Define injected ontology and controlled PA grounding boundaries.

The main PA UI supplies the production grounding runtime only when an
authoritative schema-only TBox and model configuration are present. Tests may
inject controlled implementations without widening ProductAgent's public
surface.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from rdflib import Literal, URIRef

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingContractError,
    GroundingProducerDescriptor,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import ABoxSnapshot
from cais_spade_llm.spec2primitives.ontology import (
    OntologyContextError,
    TBoxSnapshot,
    load_ppr_tbox,
)

if TYPE_CHECKING:
    from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
        ProductAgentContextRuntime,
    )

_EVIDENCE_TYPES = frozenset({"document", "CAD", "observation"})
_PRODUCER_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")


@dataclass(frozen=True)
class PAOntologyConfig:
    """Locate the exact schema-only PPR TBox used by one PA interaction."""

    tbox_path: Path
    ppr_namespace: str

    def load_tbox(self) -> TBoxSnapshot:
        """Load and profile-validate the configured immutable TBox."""
        return load_ppr_tbox(
            Path(self.tbox_path),
            ppr_namespace=self.ppr_namespace,
        )


class ProductContextGroundingRuntime(Protocol):
    """Expose only controlled Phase 4 interpretation and assessment calls."""

    def grounding_producer_descriptors(
        self,
    ) -> Sequence[GroundingProducerDescriptor | Mapping[str, object]]:
        """Return the application-owned output-capable producer registry."""
        ...

    async def initial_product_context_decision(
        self,
        product_agent: ProductAgentContextRuntime,
        *,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        product_context: Mapping[str, object],
        max_pa_turns: int,
    ) -> Mapping[str, object]:
        """Optionally make the first request from a TaskTransitionDraft."""
        ...

    async def interpret_served_context(
        self,
        *,
        producer: str,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        served_context: Mapping[str, object],
        operation_number: int,
    ) -> Mapping[str, object]:
        """Interpret one served result without mutating the interaction ABox."""
        ...

    async def assess_product_context(  # noqa: PLR0913
        self,
        product_agent: ProductAgentContextRuntime,
        *,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        abox_view: Mapping[str, object],
        attempted_evidence: tuple[str, ...],
        clarification_history: tuple[Mapping[str, object], ...] = (),
        turn_number: int,
        max_pa_turns: int,
    ) -> Mapping[str, object]:
        """Return one Phase 4.3-style decision over the updated ABox."""
        ...


def compact_abox_view(abox: ABoxSnapshot) -> dict[str, object]:
    """Return a JSON-safe product-context view without hidden inference."""
    assertions = []
    for subject, predicate, value in sorted(
        abox.graph,
        key=lambda triple: tuple(str(node) for node in triple),
    ):
        object_record: dict[str, object]
        if isinstance(value, URIRef):
            object_record = {"kind": "iri", "value": str(value)}
        elif isinstance(value, Literal):
            object_record = {
                "kind": "literal",
                "value": value.toPython(),
                "datatype": None if value.datatype is None else str(value.datatype),
                "language": value.language,
            }
        else:
            object_record = {"kind": "node", "value": str(value)}
        assertions.append(
            {
                "subject": str(subject),
                "predicate": str(predicate),
                "object": object_record,
            }
        )
    return {
        "status": "unresolved",
        "interaction_namespace": abox.namespace,
        "specification_iri": abox.specification_iri,
        "tbox_fingerprint": abox.tbox_fingerprint,
        "product_requirement": abox.product_requirement,
        "delta_count": abox.delta_count,
        "accepted_assertion_count": abox.accepted_assertion_count,
        "assertions": assertions,
    }


def validated_grounding_producer_descriptors(
    grounding_runtime: ProductContextGroundingRuntime,
) -> tuple[GroundingProducerDescriptor, ...]:
    """Validate the output-capable producer registry and evidence coverage."""
    values = grounding_runtime.grounding_producer_descriptors()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise OntologyContextError("grounding_producer_descriptors must be non-empty.")
    validated: list[GroundingProducerDescriptor] = []
    for value in values:
        try:
            descriptor = (
                value
                if isinstance(value, GroundingProducerDescriptor)
                else GroundingProducerDescriptor.from_mapping(value)
            )
        except (GroundingContractError, TypeError) as exc:
            raise OntologyContextError(
                f"Grounding producer descriptor is invalid: {exc}"
            ) from exc
        if not _PRODUCER_SYMBOL.fullmatch(descriptor.producer):
            raise OntologyContextError(
                "Grounding producer must be a fixed non-path identifier."
            )
        if not set(descriptor.evidence_types).issubset(_EVIDENCE_TYPES | {"existing_record"}):
            raise OntologyContextError(
                f"{descriptor.producer} advertises an invalid evidence type."
            )
        validated.append(descriptor)
    covered_evidence = {
        evidence_type
        for descriptor in validated
        for evidence_type in descriptor.evidence_types
        if evidence_type != "existing_record"
    }
    if covered_evidence != _EVIDENCE_TYPES:
        raise OntologyContextError(
            "Grounding producer descriptors must cover document, CAD, and observation."
        )
    return tuple(validated)


def producer_for_evidence_type(
    descriptors: Sequence[GroundingProducerDescriptor],
    evidence_type: str,
) -> str:
    """Resolve one unambiguous producer symbol for served raw evidence."""
    producers = {
        descriptor.producer
        for descriptor in descriptors
        if evidence_type in descriptor.evidence_types
    }
    if len(producers) != 1:
        raise OntologyContextError(
            f"Served {evidence_type} evidence must route to exactly one producer."
        )
    return next(iter(producers))
