"""Define injected ontology and controlled PA grounding boundaries.

The main PA UI intentionally supplies no complete grounding runtime until an
authoritative schema-only TBox and the correspondence, pose, and Phase 4.3
producers exist. Tests
and future adapters may inject implementations without widening ProductAgent's
public surface.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from rdflib import Literal, URIRef

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

    def grounding_producer_routes(self) -> Mapping[str, str]:
        """Map each evidence type to its application-owned producer symbol."""
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

    async def assess_product_context(
        self,
        product_agent: ProductAgentContextRuntime,
        *,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        abox_view: Mapping[str, object],
        attempted_evidence: tuple[str, ...],
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


def validated_grounding_producer_routes(
    grounding_runtime: ProductContextGroundingRuntime,
) -> dict[str, str]:
    """Validate the complete evidence-type routing table."""
    routes = grounding_runtime.grounding_producer_routes()
    if not isinstance(routes, Mapping) or set(routes) != _EVIDENCE_TYPES:
        raise OntologyContextError(
            "grounding_producer_routes must map document, CAD, and observation."
        )
    validated: dict[str, str] = {}
    for evidence_type in sorted(_EVIDENCE_TYPES):
        producer = routes[evidence_type]
        if not isinstance(producer, str) or not _PRODUCER_SYMBOL.fullmatch(producer):
            raise OntologyContextError(
                f"Producer for {evidence_type} must be a fixed non-path identifier."
            )
        validated[evidence_type] = producer
    return validated
