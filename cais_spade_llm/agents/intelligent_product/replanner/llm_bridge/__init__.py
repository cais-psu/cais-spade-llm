"""LLM bridge helpers for intelligent-product replanning."""

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_compiler import (
    BridgeCompilerMixin,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_safety import (
    BridgeSafetyMixin,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_session import (
    BridgeSessionMixin,
)


class LlmBridgeReplannerMixin(BridgeSessionMixin, BridgeCompilerMixin, BridgeSafetyMixin):
    """Aggregate mixin for DES-guided LLM bridge replanning."""

    pass
