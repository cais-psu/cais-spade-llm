"""OpenAI-backed document evidence interpretation boundary."""

from cais_spade_llm.spec2primitives.tools.document_evidence.diagnostic import (
    run_document_interpretation_diagnostic,
)
from cais_spade_llm.spec2primitives.tools.document_evidence.interpreter import (
    DOCUMENT_OVERVIEW_SCHEMA_VERSION,
    DocumentEvidenceRecord,
    DocumentInterpretationError,
    DocumentInterpretationResult,
    DocumentOverviewRecord,
    DocumentVisionRequest,
    DocumentVisionResponse,
    DocumentVisionRuntime,
    OpenAIDocumentVisionRuntime,
    TargetedDocumentVisionRequest,
    document_overview_cache_status,
    inspect_document_evidence,
    interpret_document_evidence,
    prepare_document_overview,
)

__all__ = [
    "DOCUMENT_OVERVIEW_SCHEMA_VERSION",
    "DocumentInterpretationError",
    "DocumentInterpretationResult",
    "DocumentEvidenceRecord",
    "DocumentOverviewRecord",
    "DocumentVisionRequest",
    "DocumentVisionResponse",
    "DocumentVisionRuntime",
    "OpenAIDocumentVisionRuntime",
    "TargetedDocumentVisionRequest",
    "document_overview_cache_status",
    "inspect_document_evidence",
    "interpret_document_evidence",
    "prepare_document_overview",
    "run_document_interpretation_diagnostic",
]
