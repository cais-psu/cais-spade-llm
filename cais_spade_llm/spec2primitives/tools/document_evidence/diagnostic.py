"""Run Phase 4.1 against a separate diagnostic interaction ABox."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.config import DocumentVLMConfig
from cais_spade_llm.spec2primitives.ontology import OntologyContextError
from cais_spade_llm.spec2primitives.tools.document_evidence.interpreter import (
    DOCUMENT_CONTEXT_REF,
    DocumentInterpretationError,
    DocumentVisionRuntime,
    interpret_document_evidence,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    resolve_context_ref,
)

_DIAGNOSTIC_RECORD = Path("interaction_record/document_interpretation_diagnostic.json")


async def run_document_interpretation_diagnostic(
    *,
    interaction_root: Path,
    product_requirement: str,
    ontology_config: PAOntologyConfig,
    config: DocumentVLMConfig,
    vision_runtime: DocumentVisionRuntime,
) -> dict[str, object]:
    """Interpret the approved PDF and validate its delta in an isolated ABox.

    This diagnostic never invokes ProductAgent and never assesses context
    completeness. Its ABox is independent from the main Phase 3 interaction.
    """
    root = Path(interaction_root).resolve()
    record_path = root / _DIAGNOSTIC_RECORD
    if not isinstance(product_requirement, str) or not product_requirement.strip():
        return _failure_result(
            product_requirement,
            record_path,
            "invalid_product_requirement",
            "product_requirement must contain non-whitespace text.",
        )
    if record_path.exists():
        return _failure_result(
            product_requirement,
            record_path,
            "diagnostic_exists",
            "The diagnostic interaction already has a result record.",
        )

    try:
        tbox = ontology_config.load_tbox()
        abox = initialize_interaction_abox(root, product_requirement, tbox)
        resolved = resolve_context_ref({"context_ref": DOCUMENT_CONTEXT_REF})
        served_context = resolved.get("served_context")
        if not isinstance(served_context, Mapping):
            raise DocumentInterpretationError(
                "The exact approved NIST document could not be served."
            )
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
            f"{DOCUMENT_CONTEXT_REF}#page={page_number}"
            for page_number in range(1, interpretation.page_count + 1)
        ]
        merge = validate_and_merge_triple_delta(
            root,
            tbox,
            "document_evidence",
            interpretation.delta,
            authorized_evidence_refs=[DOCUMENT_CONTEXT_REF, *page_refs],
        )
    except (DocumentInterpretationError, OntologyContextError, OSError, RuntimeError) as exc:
        result = _failure_result(
            product_requirement,
            record_path,
            "document_interpretation_rejected",
            f"{type(exc).__name__}: {exc}",
        )
        trace_path = root / "products/grounding/document_evidence/interpretation_0001.json"
        abox_path = root / "products/grounding/ontology/interaction_abox.ttl"
        if trace_path.is_file():
            result["trace_path"] = str(trace_path)
        if abox_path.is_file():
            result["abox_path"] = str(abox_path)
        _write_record_if_absent(record_path, result)
        return result

    result = {
        "status": "accepted",
        "product_requirement": product_requirement,
        "context_ref": DOCUMENT_CONTEXT_REF,
        "page_count": interpretation.page_count,
        "provider": interpretation.provider,
        "model": interpretation.model,
        "response_id": interpretation.response_id,
        "assertion_count": interpretation.assertion_count,
        "supported_findings": interpretation.delta["assertions"],
        "uncertainty_count": interpretation.uncertainty_count,
        "uncertainty": interpretation.delta["uncertainty"],
        "unresolved_evidence_need_count": interpretation.unresolved_evidence_need_count,
        "unresolved_evidence_needs": interpretation.delta["unresolved_evidence_needs"],
        "abox_path": str(merge.abox.abox_path),
        "trace_path": str(interpretation.trace_path),
        "delta_path": str(merge.delta_path),
        "diagnostic_record_path": str(record_path),
        "failure": None,
    }
    _write_record_if_absent(record_path, result)
    return result


def _failure_result(
    product_requirement: object,
    record_path: Path,
    reason: str,
    message: str,
) -> dict[str, object]:
    return {
        "status": "rejected",
        "product_requirement": product_requirement,
        "context_ref": DOCUMENT_CONTEXT_REF,
        "page_count": None,
        "provider": None,
        "model": None,
        "response_id": None,
        "assertion_count": 0,
        "supported_findings": [],
        "uncertainty_count": 0,
        "uncertainty": [],
        "unresolved_evidence_need_count": 0,
        "unresolved_evidence_needs": [],
        "abox_path": None,
        "trace_path": None,
        "delta_path": None,
        "diagnostic_record_path": str(record_path),
        "failure": {"reason": reason, "message": message},
    }


def _write_record_if_absent(path: Path, record: Mapping[str, object]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
