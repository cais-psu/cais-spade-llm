"""Spec2Primitives selected-RA integration boundary."""

from .context_handoff import (
    PrimitiveCatalogSnapshot,
    RAContextHandoffError,
    RobotAgentCompositionRuntime,
    RobotStateSnapshot,
    SelectedRAAssignmentEnvelope,
    SelectedRAContextSnapshot,
    activate_selected_ra_context,
)

__all__ = [
    "PrimitiveCatalogSnapshot",
    "RAContextHandoffError",
    "RobotAgentCompositionRuntime",
    "RobotStateSnapshot",
    "SelectedRAAssignmentEnvelope",
    "SelectedRAContextSnapshot",
    "activate_selected_ra_context",
]
