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
    read_phase_5_1_diagnostic,
)

__all__ = [
    "Phase51Diagnostic",
    "PrimitiveCatalogSnapshot",
    "RAContextHandoffError",
    "RobotAgentCompositionRuntime",
    "RobotStateSnapshot",
    "SelectedRAAssignmentEnvelope",
    "SelectedRAContextSnapshot",
    "activate_selected_ra_context",
    "read_phase_5_1_diagnostic",
]
