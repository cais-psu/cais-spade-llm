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
    GroundingActionAttempt,
    GroundingNextAction,
    GroundingProducerDescriptor,
    GroundingSession,
    PAContextGroundingCompletionV2,
    ProductContextView,
    TypedContextBinding,
    TypedGroundingContract,
    load_pa_context_grounding_completion,
)
from cais_spade_llm.spec2primitives.agents.pa.production_grounding import (
    CameraToWorldCalibrationRuntime,
    ProductionProductContextGroundingRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    CandidateReachEvidence,
    ResourceAssignmentNeed,
    ResourceGroundingError,
    ResourceSelectionRecord,
    RobotFramePoseEvidenceError,
    commit_resource_assignment,
    derive_resource_assignment_need,
    select_predefined_resource,
)

__all__ = [
    "ProductAgentContextRuntime",
    "CameraToWorldCalibrationRuntime",
    "CandidateReachEvidence",
    "GroundingActionAttempt",
    "GroundingNextAction",
    "GroundingProducerDescriptor",
    "GroundingSession",
    "PAOntologyConfig",
    "PAContextGroundingCompletionV2",
    "ProductContextView",
    "ProductContextGroundingRuntime",
    "ProductionProductContextGroundingRuntime",
    "ResourceAssignmentNeed",
    "ResourceGroundingError",
    "ResourceSelectionRecord",
    "RobotFramePoseEvidenceError",
    "TypedContextBinding",
    "TypedGroundingContract",
    "cancel_pa_context_interaction",
    "continue_pa_context_interaction",
    "commit_resource_assignment",
    "derive_resource_assignment_need",
    "load_pa_context_grounding_completion",
    "serve_pa_requested_context",
    "select_predefined_resource",
    "start_pa_context_interaction",
    "submit_pa_clarification_reply",
]
