"""Run the evidence-first document stages in an isolated diagnostic ABox."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import (
    OntologyGroundingError,
    OntologyGroundingInterruption,
    commit_ontology_grounding_candidate,
    propose_and_validate_ontology_grounding,
    reject_ontology_grounding_candidate,
    review_target_feature_semantics,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.config import DocumentVLMConfig
from cais_spade_llm.spec2primitives.ontology import (
    OntologyContextError,
    load_predefined_resource_registry,
    load_predefined_workcell,
)
from cais_spade_llm.spec2primitives.tools.document_evidence.interpreter import (
    DocumentInterpretationError,
    DocumentVisionRuntime,
    interpret_document_evidence,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    resolve_context_ref,
)

_DIAGNOSTIC_RECORD = Path("interaction_record/document_interpretation_diagnostic.json")


async def run_document_interpretation_diagnostic(  # noqa: PLR0913
    *,
    interaction_root: Path,
    product_requirement: str,
    context_ref: str,
    product_agent: ProductAgentContextRuntime,
    ontology_config: PAOntologyConfig,
    config: DocumentVLMConfig,
    vision_runtime: DocumentVisionRuntime,
) -> dict[str, object]:
    """Run one-page-set overview and direct ontology validation stages.

    The diagnostic is isolated from the main PA interaction. The selected
    document is an explicit operator input; no semantic target is supplied to
    the overview or late mapper.
    """
    root = Path(interaction_root).resolve()
    record_path = root / _DIAGNOSTIC_RECORD
    if not isinstance(product_requirement, str) or not product_requirement.strip():
        return _failure_result(
            product_requirement,
            context_ref,
            record_path,
            "invalid_product_requirement",
            "product_requirement must contain non-whitespace text.",
        )
    if record_path.exists():
        return _failure_result(
            product_requirement,
            context_ref,
            record_path,
            "diagnostic_exists",
            "The diagnostic interaction already has a result record.",
        )

    overview_stage: dict[str, object] | None = None
    proposal_stage: dict[str, object] | None = None
    try:
        tbox = ontology_config.load_tbox()
        abox = initialize_interaction_abox(root, product_requirement, tbox)
        resolved = resolve_context_ref({"context_ref": context_ref})
        served_context = resolved.get("served_context")
        if not isinstance(served_context, Mapping):
            raise DocumentInterpretationError("The selected approved document could not be served.")
        interpretation = await interpret_document_evidence(
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            served_context=served_context,
            operation_number=1,
            config=config,
            vision_runtime=vision_runtime,
        )
        page_refs = [
            f"{context_ref}#page={page_number}"
            for page_number in range(1, interpretation.page_count + 1)
        ]
        overview_merge = validate_and_merge_triple_delta(
            root,
            tbox,
            "document_evidence",
            interpretation.delta,
            authorized_evidence_refs=[context_ref, *page_refs],
        )
        overview_snapshot = _read_mapping(
            interpretation.overview_record_path,
            "DocumentOverviewRecord",
        )
        overview = overview_snapshot["overview"]
        if not isinstance(overview, Mapping):
            raise DocumentInterpretationError("DocumentOverviewRecord snapshot is invalid.")
        overview_ref = str(interpretation.overview_record_path.relative_to(root))
        overview_stage = {
            "status": "accepted",
            "cache_status": interpretation.cache_status,
            "record_ref": overview_ref,
            "summary": overview["summary"],
            "observations": overview["observations"],
            "uncertainty": overview["uncertainty"],
        }
        evidence_refs = [context_ref, overview_ref, *page_refs]

        async def no_retrieval(
            tool_name: str,
            arguments: Mapping[str, object],
        ) -> Mapping[str, object]:
            del tool_name, arguments
            return {
                "error": {
                    "reason": "diagnostic_evidence_fixed",
                    "message": "This diagnostic already supplied its selected document.",
                }
            }

        workcell = load_predefined_workcell(
            tbox,
            load_predefined_resource_registry(tbox),
        )
        evidence_catalog = [
            {
                "retrieval_state": "already_retrieved",
                "evidence_type": "document",
                "source": context_ref,
                "evidence_refs": evidence_refs,
                "record_refs": [overview_ref],
                "summary": overview.get("summary"),
                "pages": overview.get("pages"),
                "visual_observations": overview.get("observations"),
                "uncertainty": overview.get("uncertainty"),
            }
        ]
        authorized_evidence_refs = {"requirement_0001", *evidence_refs}
        candidate = await propose_and_validate_ontology_grounding(
            product_agent,
            interaction_root=root,
            tbox=tbox,
            abox=overview_merge.abox,
            workcell=workcell,
            evidence_catalog=evidence_catalog,
            authorized_evidence_refs=authorized_evidence_refs,
            tools=[],
            tool_executor=no_retrieval,
            max_tool_rounds=1,
            required_output_projection={
                "record_type": "RobotFrameLocationRecord",
                "target_frame": "configured resource reach frame",
                "purpose": "diagnostic schema projection only",
            },
        )
        if isinstance(candidate, OntologyGroundingInterruption):
            raise OntologyGroundingError(
                f"Document diagnostic did not return a proposal: {candidate.message}"
            )
        semantic_review = await review_target_feature_semantics(
            product_agent,
            interaction_root=root,
            candidate=candidate,
            product_requirement=product_requirement,
            evidence_catalog=evidence_catalog,
        )
        if semantic_review.verdict != "complete":
            reject_ontology_grounding_candidate(
                candidate,
                interaction_root=root,
                abox=overview_merge.abox,
                semantic_review=semantic_review,
            )
            raise OntologyGroundingError(
                semantic_review.gap or "Document target feature is semantically incomplete."
            )
        proposal = commit_ontology_grounding_candidate(
            candidate,
            interaction_root=root,
            tbox=tbox,
            abox=overview_merge.abox,
            workcell=workcell,
            authorized_evidence_refs=authorized_evidence_refs,
            semantic_review=semantic_review,
        )
        proposal_record = _read_mapping(
            proposal.proposal_path,
            "OntologyGroundingProposal",
        )
        proposal_stage = {
            "status": proposal_record["status"],
            "record_ref": str(proposal.proposal_path.relative_to(root)),
            "source_evidence_refs": evidence_refs,
            "PA_output": proposal_record["output"],
            "compiled_delta": proposal_record["compiled_delta"],
        }
    except (
        DocumentInterpretationError,
        OntologyContextError,
        OntologyGroundingError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        result = _failure_result(
            product_requirement,
            context_ref,
            record_path,
            "document_interpretation_rejected",
            f"{type(exc).__name__}: {exc}",
        )
        result["overview"] = overview_stage
        result["ontology_proposal"] = proposal_stage
        _write_record_if_absent(record_path, result)
        return result

    accepted_delta = _read_mapping(proposal.merge.delta_path, "accepted delta")
    result = {
        "status": "accepted",
        "product_requirement": product_requirement,
        "context_ref": context_ref,
        "overview": overview_stage,
        "ontology_proposal": proposal_stage,
        "accepted_assertions": accepted_delta["assertions"],
        "abox_path": str(proposal.merge.abox.abox_path),
        "diagnostic_record_path": str(record_path),
        "failure": None,
    }
    _write_record_if_absent(record_path, result)
    return result


def _failure_result(
    product_requirement: object,
    context_ref: object,
    record_path: Path,
    reason: str,
    message: str,
) -> dict[str, object]:
    return {
        "status": "rejected",
        "product_requirement": product_requirement,
        "context_ref": context_ref,
        "overview": None,
        "ontology_proposal": None,
        "accepted_assertions": [],
        "abox_path": None,
        "diagnostic_record_path": str(record_path),
        "failure": {"reason": reason, "message": message},
    }


def _read_mapping(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DocumentInterpretationError(f"{label} could not be read.") from exc
    if not isinstance(value, Mapping):
        raise DocumentInterpretationError(f"{label} must be an object.")
    return value


def _write_record_if_absent(path: Path, record: Mapping[str, object]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
