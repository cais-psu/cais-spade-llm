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
from .primitive_composition import (
    PrimitiveCompositionError,
    PrimitiveProgramCandidate,
    RobotAgentProgramRuntime,
    author_primitive_program_candidate,
    read_primitive_composition_diagnostic,
)

__all__ = [
    "Phase51Diagnostic",
    "PrimitiveCatalogSnapshot",
    "PrimitiveCompositionError",
    "PrimitiveProgramCandidate",
    "RAContextHandoffError",
    "RobotAgentCompositionRuntime",
    "RobotAgentFeasibilityRuntime",
    "RobotAgentProgramRuntime",
    "RobotStateSnapshot",
    "SelectedRAAssignmentEnvelope",
    "SelectedRAContextSnapshot",
    "activate_selected_ra_context",
    "author_primitive_program_candidate",
    "load_selected_ra_context_snapshot",
    "read_phase_5_1_diagnostic",
    "read_primitive_composition_diagnostic",
]
