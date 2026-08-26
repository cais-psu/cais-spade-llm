"""Controlled ontology and grounding doubles for PA workflow tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rdflib import RDF

from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import ABoxSnapshot
from cais_spade_llm.spec2primitives.ontology import TBoxSnapshot

PPR_NAMESPACE = "http://PAonto.com#"
MINIMAL_TBOX_PATH = Path(__file__).parent / "fixtures/ontology/minimal_ppr_tbox.owl"


def ontology_config() -> PAOntologyConfig:
    """Return the schema-only test fixture configuration."""
    return PAOntologyConfig(
        tbox_path=MINIMAL_TBOX_PATH,
        ppr_namespace=PPR_NAMESPACE,
    )


class ControlledGroundingRuntime:
    """Produce deterministic evidence deltas and Phase 4.3 decisions."""

    def __init__(
        self,
        *,
        assessments: list[Mapping[str, object]] | None = None,
        interpretation_error: Exception | None = None,
        assessment_error: Exception | None = None,
        interpretation_override: Mapping[str, object] | None = None,
        routes: Mapping[str, str] | None = None,
    ) -> None:
        self.assessments = None if assessments is None else list(assessments)
        self.interpretation_error = interpretation_error
        self.assessment_error = assessment_error
        self.interpretation_override = interpretation_override
        self.routes = dict(routes or _default_routes())
        self.interpretation_calls: list[dict[str, object]] = []
        self.assessment_calls: list[dict[str, object]] = []

    def grounding_producer_routes(self) -> Mapping[str, str]:
        """Return the controlled evidence-type routing table."""
        return self.routes

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
        """Return one valid dynamic assertion for the served evidence."""
        self.interpretation_calls.append(
            {
                "producer": producer,
                "interaction_root": interaction_root,
                "served_context": served_context,
                "operation_number": operation_number,
                "delta_count": abox.delta_count,
            }
        )
        if self.interpretation_error is not None:
            raise self.interpretation_error
        if self.interpretation_override is not None:
            return self.interpretation_override

        evidence_ref = served_context.get("context_ref") or served_context.get("observation_ref")
        if not isinstance(evidence_ref, str):
            raise ValueError("Controlled served evidence has no exact ref.")
        class_iri = (
            f"{PPR_NAMESPACE}feature"
            if producer == "document_evidence"
            else f"{PPR_NAMESPACE}product"
        )
        subject = f"{abox.namespace}{producer}_{operation_number}"
        return {
            "assertions": [
                {
                    "subject": subject,
                    "predicate": str(RDF.type),
                    "object": {"kind": "iri", "value": class_iri},
                    "evidence_refs": [evidence_ref],
                }
            ],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [],
        }

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
        """Return a scripted decision or delegate through the PA protocol."""
        call = {
            "interaction_root": interaction_root,
            "tbox_fingerprint": tbox.fingerprint,
            "delta_count": abox.delta_count,
            "abox_view": abox_view,
            "attempted_evidence": attempted_evidence,
            "turn_number": turn_number,
            "max_pa_turns": max_pa_turns,
        }
        self.assessment_calls.append(call)
        if self.assessment_error is not None:
            raise self.assessment_error
        if self.assessments is not None:
            return self.assessments.pop(0)
        prompt = (
            "Assess only this ontology-backed product context. Do not inspect raw "
            "served evidence.\n"
            f"{json.dumps(call, default=str, ensure_ascii=False)}"
        )
        return await product_agent.ask_llm_structured(
            prompt,
            response_format={"name": "controlled_phase_4_3_test"},
        )


def request_context(
    context_ref: str,
    *,
    semantic_need: str = "additional product evidence",
) -> dict[str, object]:
    """Return one valid Phase 4.3 static-evidence request."""
    evidence_type = "document" if context_ref.endswith(".pdf") else "CAD"
    return {
        "unresolved_semantic_need": {
            "kind": "class",
            "symbol": (
                f"{PPR_NAMESPACE}feature"
                if evidence_type == "document"
                else f"{PPR_NAMESPACE}product"
            ),
            "description": semantic_need,
        },
        "needed_context": {
            "context_ref": context_ref,
            "request_live_observation": False,
            "clarification_question": None,
        },
        "context understanding complete": False,
    }


def request_live_observation(
    *,
    semantic_need: str = "fresh scene arrangement",
) -> dict[str, object]:
    """Return one valid Phase 4.3 live-observation request."""
    return {
        "unresolved_semantic_need": {
            "kind": "class",
            "symbol": f"{PPR_NAMESPACE}product",
            "description": semantic_need,
        },
        "needed_context": {
            "context_ref": None,
            "request_live_observation": True,
            "clarification_question": None,
        },
        "context understanding complete": False,
    }


def request_clarification(question: str) -> dict[str, object]:
    """Return one valid Phase 4.3 clarification decision."""
    return {
        "unresolved_semantic_need": {
            "kind": "user_intent",
            "symbol": "product_requirement",
            "description": "unresolved user intent",
        },
        "needed_context": {
            "context_ref": None,
            "request_live_observation": False,
            "clarification_question": question,
        },
        "context understanding complete": False,
    }


def complete_context() -> dict[str, object]:
    """Return one valid Phase 4.3 completion decision."""
    return {
        "unresolved_semantic_need": None,
        "needed_context": None,
        "context understanding complete": True,
    }


def _default_routes() -> dict[str, str]:
    return {
        "document": "document_evidence",
        "CAD": "rgb_d_cad_grounding",
        "observation": "rgb_d_cad_grounding",
    }
