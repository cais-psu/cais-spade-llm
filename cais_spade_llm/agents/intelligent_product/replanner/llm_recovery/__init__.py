"""LLM recovery helpers for intelligent-product replanning."""

from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_session import (
    RecoverySessionMixin,
)


class LlmRecoveryReplannerMixin(RecoverySessionMixin):
    """Active recovery mixin used by ``ProcessPlanner``."""

    pass


__all__ = ["LlmRecoveryReplannerMixin"]
