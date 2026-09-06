from __future__ import annotations

"""Spec2Primitives selected-RA integration boundary."""

from .context_handoff import (
    Phase51Diagnostic,
    PrimitiveCatalogSnapshot,
    RAContextHandoffError,
    RobotAgentCompositionRuntime,
    RobotStateSnapshot,
    SelectedRAAssignmentEnvelope,
    SelectedRAContextSnapshot,
    activate_selected_ra_context,
    load_selected_ra_context_snapshot,
    read_phase_5_1_diagnostic,
)
from .feasibility_validation import RobotAgentFeasibilityRuntime
from .primitive_draft import (
    Phase52Diagnostic,
    PrimitiveDraftError,
    PrimitiveProgramDraft,
    RobotAgentDraftRuntime,
    author_primitive_program_draft,
    read_phase_5_2_diagnostic,
)

__all__ = [
    "Phase51Diagnostic",
    "Phase52Diagnostic",
    "PrimitiveCatalogSnapshot",
    "PrimitiveDraftError",
    "PrimitiveProgramDraft",
    "RAContextHandoffError",
    "RobotAgentCompositionRuntime",
    "RobotAgentDraftRuntime",
    "RobotAgentFeasibilityRuntime",
    "RobotStateSnapshot",
    "SelectedRAAssignmentEnvelope",
    "SelectedRAContextSnapshot",
    "activate_selected_ra_context",
    "author_primitive_program_draft",
    "load_selected_ra_context_snapshot",
    "read_phase_5_1_diagnostic",
    "read_phase_5_2_diagnostic",
]
