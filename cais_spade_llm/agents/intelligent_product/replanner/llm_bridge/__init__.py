"""LLM bridge helpers for intelligent-product replanning."""

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_session import (
    BridgeSessionMixin,
)


class LlmBridgeReplannerMixin(BridgeSessionMixin):
    """Active bridge mixin used by ``ProcessPlanner``."""

    pass


__all__ = ["LlmBridgeReplannerMixin"]
