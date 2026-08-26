"""OpenAI-backed document evidence interpretation boundary."""

from cais_spade_llm.spec2primitives.tools.document_evidence.diagnostic import (
    run_document_interpretation_diagnostic,
)
from cais_spade_llm.spec2primitives.tools.document_evidence.interpreter import (
    DOCUMENT_CONTEXT_REF,
    DocumentInterpretationError,
    DocumentInterpretationResult,
    DocumentVisionRequest,
    DocumentVisionResponse,
    DocumentVisionRuntime,
    OpenAIDocumentVisionRuntime,
    interpret_document_evidence,
)

__all__ = [
    "DOCUMENT_CONTEXT_REF",
    "DocumentInterpretationError",
    "DocumentInterpretationResult",
    "DocumentVisionRequest",
    "DocumentVisionResponse",
    "DocumentVisionRuntime",
    "OpenAIDocumentVisionRuntime",
    "interpret_document_evidence",
    "run_document_interpretation_diagnostic",
]
