from __future__ import annotations

"""Spec2Primitives PA context-interaction boundary."""

from cais_spade_llm.spec2primitives.agents.pa.context_assessment import (
    cancel_pa_context_interaction,
    continue_pa_context_interaction,
    submit_pa_clarification_reply,
)
from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
    ProductContextGroundingRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
    start_pa_context_interaction,
)
from cais_spade_llm.spec2primitives.agents.pa.context_serving import (
    serve_pa_requested_context,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingProducerDescriptor,
    PAContextGroundingCompletion,
    ProductContextView,
    TypedContextBinding,
    load_completed_product_context_view,
    load_pa_context_grounding_completion,
)
from cais_spade_llm.spec2primitives.agents.pa.presentation_records import (
    AllocationEvidenceEntry,
    AllocationEvidenceSource,
    AllocationPresentationRecord,
    AllocationResourceEntry,
    EvidencePresentationEntry,
    EvidencePresentationRecord,
    PresentationRecordError,
    load_allocation_presentation,
    load_evidence_presentation,
    load_or_create_allocation_presentation,
    load_or_create_evidence_presentation,
)
from cais_spade_llm.spec2primitives.agents.pa.production_grounding import (
    CameraToWorldCalibrationRuntime,
    ProductionProductContextGroundingRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    ReachabilityCheckRecord,
    ResourceGroundingError,
    ResourceSelectionRecord,
    RobotFrameLocationEvidenceError,
    candidate_resource_catalog,
    check_live_resource_reachability,
    commit_resource_assignment,
    persist_pa_resource_selection,
)

__all__ = [
    "ProductAgentContextRuntime",
    "CameraToWorldCalibrationRuntime",
    "ReachabilityCheckRecord",
    "GroundingProducerDescriptor",
    "AllocationEvidenceEntry",
    "AllocationEvidenceSource",
    "AllocationPresentationRecord",
    "AllocationResourceEntry",
    "EvidencePresentationEntry",
    "EvidencePresentationRecord",
    "PAOntologyConfig",
    "PAContextGroundingCompletion",
    "ProductContextView",
    "ProductContextGroundingRuntime",
    "ProductionProductContextGroundingRuntime",
    "PresentationRecordError",
    "ResourceGroundingError",
    "ResourceSelectionRecord",
    "RobotFrameLocationEvidenceError",
    "TypedContextBinding",
    "cancel_pa_context_interaction",
    "candidate_resource_catalog",
    "check_live_resource_reachability",
    "continue_pa_context_interaction",
    "commit_resource_assignment",
    "load_completed_product_context_view",
    "load_allocation_presentation",
    "load_evidence_presentation",
    "load_or_create_allocation_presentation",
    "load_or_create_evidence_presentation",
    "load_pa_context_grounding_completion",
    "persist_pa_resource_selection",
    "serve_pa_requested_context",
    "start_pa_context_interaction",
    "submit_pa_clarification_reply",
]
