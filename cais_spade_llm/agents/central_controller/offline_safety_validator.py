"""Compatibility shim for the validator rename to plan_safety_validator."""

from __future__ import annotations

from cais_spade_llm.agents.central_controller.plan_safety_validator import (
    PlanSafetyValidator,
)


class OfflineSafetyValidator(PlanSafetyValidator):
    """Legacy alias retained during the plan validator migration."""

